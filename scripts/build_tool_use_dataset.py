"""Build context-aware tool-use SFT data from the official CrossWOZ splits.

Unlike the previous single-turn conversion, this builder keeps recent
dialogue history and the previous structured state.  Targets are grounded in
CrossWOZ's dynamic ``user_state`` and the following system turn's faithful
``sys_state_init`` database query.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(ROOT / "src")
if SOURCE_ROOT in sys.path:
    sys.path.remove(SOURCE_ROOT)
sys.path.insert(0, SOURCE_ROOT)

from gemma_eval.contracts import validate_decision
from gemma_eval.tool_registry import DEFAULT_TOOL_REGISTRY


TOOL_REGISTRY = DEFAULT_TOOL_REGISTRY.as_dict()
QUERY_TOOL = {spec.domain: spec.name for spec in DEFAULT_TOOL_REGISTRY.values()}
PRONOUN_MARKERS = ("这个", "那个", "这家", "那里", "该景点", "该酒店", "该餐馆", "它")
CORRECTION_MARKERS = ("不要", "改成", "换成", "不是", "还是", "重新")


def load_split(data_dir: Path, split: str) -> dict[str, Any]:
    archive = data_dir / f"{split}.json.zip"
    with zipfile.ZipFile(archive) as handle:
        with handle.open(f"{split}.json") as source:
            return json.load(source)


def normalise_user_state(raw_state: list[list[Any]]) -> list[dict[str, Any]]:
    goals: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_goal_id, domain, slot, value, mentioned in raw_state:
        if not mentioned or domain not in QUERY_TOOL or not slot:
            continue
        goal_id = int(raw_goal_id)
        goal = goals.setdefault(
            (goal_id, domain),
            {"goal_id": goal_id, "domain": domain, "constraints": {}, "requested_fields": []},
        )
        if value in ("", None) or value == []:
            if slot not in goal["requested_fields"]:
                goal["requested_fields"].append(slot)
        else:
            goal["constraints"][slot] = value
    for goal in goals.values():
        goal["requested_fields"].sort()
    return sorted(goals.values(), key=lambda item: (item["goal_id"], item["domain"]))


def active_domain(dialog_act: list[list[str]]) -> str | None:
    counts = Counter(item[1] for item in dialog_act if len(item) == 4 and item[1] in QUERY_TOOL)
    return counts.most_common(1)[0][0] if counts else None


def requested_fields(dialog_act: list[list[str]], domain: str | None) -> list[str]:
    if not domain:
        return []
    return sorted(
        {
            slot
            for intent, act_domain, slot, _ in dialog_act
            if intent == "Request" and act_domain == domain and slot and slot != "none"
        }
    )


def missing_slots(system_message: dict[str, Any], domain: str | None) -> list[str]:
    if not domain:
        return []
    return sorted(
        {
            slot
            for intent, act_domain, slot, _ in system_message.get("dialog_act", [])
            if intent == "Request" and act_domain == domain and slot and slot != "none"
        }
    )


def required_slots_missing_from_state(
    state: list[dict[str, Any]], domain: str | None
) -> list[str]:
    """Find required tool arguments that the user has not stated yet.

    CrossWOZ sometimes lets the system infer a query even when the current
    user state is incomplete.  Our Agent contract is intentionally stricter:
    a side-effect or route tool may only be proposed after its required
    arguments are explicit in the accumulated user state.
    """

    if not domain:
        return []
    tool_name = QUERY_TOOL[domain]
    required = TOOL_REGISTRY[tool_name].required_arguments
    if not required:
        return []
    active_goals = [goal for goal in state if goal["domain"] == domain]
    constraints = active_goals[-1]["constraints"] if active_goals else {}
    return [slot for slot in required if not constraints.get(slot)]


def query_arguments(system_message: dict[str, Any], domain: str, requested: list[str]) -> dict[str, Any]:
    domain_state = system_message.get("sys_state_init", {}).get(domain, {})
    constraints = {
        slot: value
        for slot, value in domain_state.items()
        if slot != "selectedResults" and value not in ("", None, [])
    }
    if domain in {"地铁", "出租"}:
        return {slot: constraints[slot] for slot in ("出发地", "目的地") if slot in constraints}
    return {"constraints": constraints, "requested_fields": requested}


def make_output(user_message: dict[str, Any], system_message: dict[str, Any]) -> dict[str, Any]:
    state = normalise_user_state(user_message.get("user_state", []))
    domain = active_domain(user_message.get("dialog_act", []))
    requested = requested_fields(user_message.get("dialog_act", []), domain)
    missing = required_slots_missing_from_state(state, domain)
    if not missing:
        missing = missing_slots(system_message, domain)
    if missing:
        output = {"belief_state": state, "decision": "ask_user", "tool_call": None, "missing_slots": missing}
    elif domain:
        output = {
            "belief_state": state,
            "decision": "call_tool",
            "tool_call": {"name": QUERY_TOOL[domain], "arguments": query_arguments(system_message, domain, requested)},
            "missing_slots": [],
        }
    else:
        output = {"belief_state": state, "decision": "no_tool", "tool_call": None, "missing_slots": []}
    validate_decision(output)
    return output


def row_label(row: dict[str, Any]) -> str:
    output = row["output"]
    return output["tool_call"]["name"] if output["decision"] == "call_tool" else output["decision"]


def build_rows(dialogues: dict[str, Any], max_history_turns: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dialog_id, dialog in dialogues.items():
        messages = dialog["messages"]
        previous_state: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if message["role"] != "usr" or index + 1 >= len(messages) or messages[index + 1]["role"] != "sys":
                continue
            system_message = messages[index + 1]
            output = make_output(message, system_message)
            history = [
                {"role": "user" if item["role"] == "usr" else "assistant", "content": item["content"]}
                for item in messages[max(0, index - max_history_turns) : index]
            ]
            reference_names: dict[str, list[str]] = defaultdict(list)
            for domain, state in system_message.get("sys_state_init", {}).items():
                if domain not in QUERY_TOOL:
                    continue
                for name in state.get("selectedResults", []):
                    reference_names[domain].append(name)
                if state.get("名称"):
                    reference_names[domain].append(state["名称"])
                if domain in {"地铁", "出租"}:
                    for slot in ("出发地", "目的地"):
                        if state.get(slot):
                            reference_names[domain].append(state[slot])
            rows.append(
                {
                    "sample_id": f"{dialog_id}_{index}",
                    "history": history,
                    "previous_state": previous_state,
                    "current_user": message["content"],
                    "output": output,
                    "source": {"dialog_id": dialog_id, "turn_index": index},
                    "_reference_names": dict(reference_names),
                }
            )
            previous_state = output["belief_state"]
    return rows


def balanced_sample(rows: list[dict[str, Any]], per_label_cap: int, rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row_label(row)].append(row)
    sampled = []
    for label in sorted(groups):
        rng.shuffle(groups[label])
        sampled.extend(groups[label][:per_label_cap])
    rng.shuffle(sampled)
    return sampled


def natural_sample(rows: list[dict[str, Any]], cap: int, rng: random.Random) -> list[dict[str, Any]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:cap]


def controlled_missing_slot_examples(
    rows: list[dict[str, Any]], cap: int, rng: random.Random, split: str
) -> list[dict[str, Any]]:
    """Create a small, declared safety slice from real endpoint values.

    CrossWOZ almost always gives the system both endpoints, so its natural
    test split cannot measure whether a model refuses an incomplete route or
    taxi call.  These cases keep real entity values but ablate exactly one
    required argument.  They are marked as controlled cases in every row and
    in the manifest; they are not presented as natural CrossWOZ turns.
    """

    candidates: list[dict[str, Any]] = []
    for row in rows:
        output = row["output"]
        call = output.get("tool_call")
        if output["decision"] != "call_tool" or not call:
            continue
        if call["name"] not in {"query_metro_route", "request_taxi"}:
            continue
        arguments = call["arguments"]
        if not arguments.get("出发地") or not arguments.get("目的地"):
            continue
        domain = TOOL_REGISTRY[call["name"]].domain
        for missing in ("出发地", "目的地"):
            present = "目的地" if missing == "出发地" else "出发地"
            value = arguments[present]
            if domain == "地铁" and present == "出发地":
                utterance = f"我从{value}出发，帮我查一下地铁路线。"
            elif domain == "地铁":
                utterance = f"我想坐地铁去{value}，帮我查一下路线。"
            elif present == "出发地":
                utterance = f"我从{value}出发，帮我叫辆出租车。"
            else:
                utterance = f"我想打车去{value}。"
            goal = {
                "goal_id": 1,
                "domain": domain,
                "constraints": {present: value},
                "requested_fields": [],
            }
            controlled = {
                "sample_id": f"controlled_{split}_{row['sample_id']}_{missing}",
                "history": [],
                "previous_state": [],
                "current_user": utterance,
                "output": {
                    "belief_state": [goal],
                    "decision": "ask_user",
                    "tool_call": None,
                    "missing_slots": [missing],
                },
                "source": {
                    "type": "controlled_required_slot_ablation",
                    "derived_from": row["sample_id"],
                    "split": split,
                },
                "_reference_names": copy.deepcopy(row.get("_reference_names", {})),
            }
            validate_decision(controlled["output"])
            candidates.append(controlled)
    rng.shuffle(candidates)
    return candidates[:cap]


def challenge_sample(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        text = row["current_user"]
        domains = {goal["domain"] for goal in row["output"]["belief_state"]}
        label = row_label(row)
        if any(marker in text for marker in PRONOUN_MARKERS):
            buckets["coreference"].append(row)
        if any(marker in text for marker in CORRECTION_MARKERS):
            buckets["correction"].append(row)
        if len(domains) > 1:
            buckets["cross_domain"].append(row)
        if label in {"ask_user", "no_tool", "request_taxi"}:
            buckets[label].append(row)
    selected: dict[str, dict[str, Any]] = {}
    while len(selected) < size and any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category]:
                row = buckets[category].pop(0)
                copy = {**row, "challenge_categories": sorted({category, *row.get("challenge_categories", [])})}
                selected.setdefault(row["sample_id"], copy)
                if len(selected) >= size:
                    break
    return list(selected.values())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if key != "_reference_names"}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")


def write_reference_subset(database_dir: Path, output_dir: Path, rows: list[dict[str, Any]]) -> None:
    names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for domain, values in row.get("_reference_names", {}).items():
            names[domain].update(values)
        call = row["output"].get("tool_call")
        if call and call["name"] in {"query_metro_route", "request_taxi"}:
            names["地铁"].update(call["arguments"].values())
    file_domains = {
        "attraction_db.json": "景点",
        "hotel_db.json": "酒店",
        "restaurant_db.json": "餐馆",
        "metro_db.json": "地铁",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, domain in file_domains.items():
        entries = json.loads((database_dir / filename).read_text(encoding="utf-8"))
        subset = [entry for entry in entries if entry[0] in names[domain]]
        (output_dir / filename).write_text(
            json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CrossWOZ multi-turn tool-use data")
    parser.add_argument("--crosswoz-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tool_use"))
    parser.add_argument("--train-per-label", type=int, default=400)
    parser.add_argument("--validation-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=600)
    parser.add_argument("--challenge-size", type=int, default=100)
    parser.add_argument("--controlled-ask-train", type=int, default=160)
    parser.add_argument("--controlled-ask-validation", type=int, default=40)
    parser.add_argument("--controlled-ask-test", type=int, default=40)
    parser.add_argument("--max-history-turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    raw = {
        "train": build_rows(load_split(args.crosswoz_dir, "train"), args.max_history_turns),
        "validation": build_rows(load_split(args.crosswoz_dir, "val"), args.max_history_turns),
        "test": build_rows(load_split(args.crosswoz_dir, "test"), args.max_history_turns),
    }
    sampled = {
        "train": balanced_sample(raw["train"], args.train_per_label, rng),
        "validation": natural_sample(raw["validation"], args.validation_size, rng),
        "test": natural_sample(raw["test"], args.test_size, rng),
    }
    controlled_counts = {
        "train": args.controlled_ask_train,
        "validation": args.controlled_ask_validation,
        "test": args.controlled_ask_test,
    }
    for split, cap in controlled_counts.items():
        sampled[split].extend(controlled_missing_slot_examples(raw[split], cap, rng, split))
        rng.shuffle(sampled[split])
    challenge = challenge_sample(sampled["test"], args.challenge_size)
    for split, rows in sampled.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    write_jsonl(args.output_dir / "challenge.jsonl", challenge)
    all_sampled = sampled["train"] + sampled["validation"] + sampled["test"]
    write_reference_subset(args.crosswoz_dir / "database", args.output_dir / "reference", all_sampled)

    manifest = {
        "source": "CrossWOZ official train/val/test splits",
        "seed": args.seed,
        "max_history_turns": args.max_history_turns,
        "raw_turn_examples": {split: len(rows) for split, rows in raw.items()},
        "sampled_examples": {split: len(rows) for split, rows in sampled.items()},
        "challenge_examples": len(challenge),
        "controlled_missing_slot_examples": {
            split: sum(
                row.get("source", {}).get("type") == "controlled_required_slot_ablation"
                for row in rows
            )
            for split, rows in sampled.items()
        },
        "route_distribution": {
            split: dict(Counter(row_label(row) for row in rows)) for split, rows in sampled.items()
        },
        "label_provenance": {
            "belief_state": "CrossWOZ user_state entries whose mentioned flag is true",
            "tool_arguments": "following system turn sys_state_init",
            "ask_user": "natural missing arguments plus explicitly marked controlled required-slot ablations built from split-local endpoint values",
            "tool_name": "deterministic CrossWOZ domain-to-local-tool mapping",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
