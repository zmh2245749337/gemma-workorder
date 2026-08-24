"""Build a single-turn structured-extraction dataset from CrossWOZ.

CrossWOZ (THU-CoAI, TACL 2020, Apache-2.0, https://github.com/thu-coai/CrossWOZ)
is a real, human-annotated Chinese task-oriented dialogue corpus: 6,012
dialogues / ~102K utterances across five Beijing local-service domains
(attractions / hotels / restaurants / metro / taxi), each user turn already
labelled with dialogue acts (intent, domain, slot, value) by human annotators.
Real per-domain reference databases (real Beijing business listings) ship
with the corpus.

This script does NOT try to reproduce full multi-turn dialogue state
tracking — that is a different, much larger research problem. It converts
CrossWOZ into the same shape this repo already trains on: one Chinese
sentence -> one structured JSON target -> one safe, allow-listed next
action. Concretely, for every user turn it uses the turn's *own* human
labels to build:
  - domain: which of the five services the turn is about
  - constraints: slot/value pairs the user stated (dialogue-act "Inform")
  - requested_fields: slots the user is asking for (dialogue-act "Request")
  - next_action: which allow-listed tool this should route to

next_action is NOT part of the original CrossWOZ annotation -- CrossWOZ's
dialogue acts describe what was said, not which backend action a system
should safely take next. The mapping below (query the matching read-only
domain database, or -- for 出租/taxi -- require human confirmation before
dispatch, since that is the one domain here with a real-world side effect)
is this project's own design choice, layered on top of real data. It
mirrors the read-only-vs-confirm-before-write split the original
work-order version of this project already used.

CrossWOZ is multi-turn, and a later turn can refer back to something named
earlier ("该景点的评分是多少" -- "what's THIS attraction's rating") without
restating it. Training on that turn in isolation would ask the model to
answer from missing context. This script keeps a turn only if it is a
dialogue's first user turn (which CrossWOZ's own user-goal simulation
always makes a self-contained opening request) or if the turn itself names
at least one concrete constraint value -- both are interpretable without
the preceding history. Later turns that only request a field with nothing
named are dropped rather than mislabelled.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter
from pathlib import Path


DOMAINS = ("景点", "酒店", "餐馆", "地铁", "出租")

# A handful of "the user is actually specifying/asking about this domain"
# slots per domain, used only to decide whether a turn has enough signal to
# route somewhere instead of asking for clarification.
KEY_SLOTS = {
    "景点": {"名称", "门票", "评分", "游玩时间"},
    "酒店": {"名称", "价格", "酒店类型", "酒店设施"},
    "餐馆": {"名称", "推荐菜", "人均消费"},
    "地铁": {"出发地", "目的地"},
    "出租": {"出发地", "目的地"},
}

QUERY_TOOL = {
    "景点": "query_attraction_db",
    "酒店": "query_hotel_db",
    "餐馆": "query_restaurant_db",
    "地铁": "query_metro_route",
}


def _load_split(data_dir: Path, split: str) -> dict:
    zip_path = data_dir / f"{split}.json.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"{split}.json") as fh:
            return json.load(fh)


def _turn_records(dialog_act: list) -> list[tuple[str, str, str, str]]:
    """CrossWOZ acts are 4-tuples: [intent, domain, slot, value]."""
    return [tuple(item) for item in dialog_act]


def build_example(sample_id: str, text: str, acts: list[tuple[str, str, str, str]]) -> dict | None:
    domain_counts = Counter(domain for _, domain, _, _ in acts if domain in DOMAINS)
    if not domain_counts:
        return None  # pure greeting/thanks/bye turn, nothing to extract
    domain = domain_counts.most_common(1)[0][0]

    constraints = {
        slot: value
        for intent, d, slot, value in acts
        if d == domain and intent == "Inform" and slot and value and value not in ("none", "")
    }
    requested = sorted(
        {slot for intent, d, slot, _ in acts if d == domain and intent == "Request" and slot}
    )

    has_key_signal = bool(KEY_SLOTS[domain] & (set(constraints) | set(requested)))
    missing_fields: list[str] = [] if has_key_signal else [f"{domain}的具体查询条件"]

    if domain == "出租":
        if {"出发地", "目的地"} <= set(constraints):
            next_action = "request_taxi"  # real-world side effect -> human confirmation gate
        else:
            next_action = "ask_clarification"
    elif has_key_signal:
        next_action = QUERY_TOOL[domain]
    else:
        next_action = "ask_clarification"

    return {
        "sample_id": sample_id,
        "input_text": text,
        "output": {
            "domain": domain,
            "constraints": constraints,
            "requested_fields": requested,
            "missing_fields": missing_fields,
            "next_action": next_action,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a service-query JSONL dataset from real CrossWOZ dialogues")
    parser.add_argument("--crosswoz-dir", type=Path, required=True, help="path to CrossWOZ/data/crosswoz")
    parser.add_argument("--output-dir", type=Path, default=Path("data/service_query"))
    parser.add_argument(
        "--per-class-cap",
        type=int,
        default=None,
        help=(
            "stratified sampling: keep at most this many examples per next_action "
            "value in each split, so the six routes stay balanced instead of "
            "mirroring CrossWOZ's natural (attraction/hotel/restaurant-heavy) mix. "
            "Recommended for actual QLoRA training runs; omit to keep everything."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_splits = {}
    action_totals: dict[str, Counter] = {}

    for split, out_name in (("train", "train"), ("val", "validation"), ("test", "test")):
        dialogues = _load_split(args.crosswoz_dir, split)
        rows = []
        seen_text = set()
        for dialog_id, dialog in dialogues.items():
            for turn_idx, message in enumerate(dialog["messages"]):
                if message["role"] != "usr":
                    continue
                text = message["content"].strip()
                if not text or text in seen_text:
                    continue
                example = build_example(f"{dialog_id}_{turn_idx}", text, _turn_records(message["dialog_act"]))
                if example is None:
                    continue
                # A later turn that only *requests* a field ("该景点的评分是多少")
                # without naming or restating anything is only interpretable
                # with the preceding dialogue history, which this single-turn
                # dataset does not keep. Only keep a turn if it is either the
                # dialogue's self-contained opening request, or it names at
                # least one concrete constraint value itself -- both are
                # interpretable on their own.
                if turn_idx != 0 and not example["output"]["constraints"]:
                    continue
                seen_text.add(text)
                rows.append(example)
        rng.shuffle(rows)

        if args.per_class_cap:
            by_action: dict[str, list] = {}
            for row in rows:
                by_action.setdefault(row["output"]["next_action"], []).append(row)
            rows = []
            for action in sorted(by_action):
                rows.extend(by_action[action][: args.per_class_cap])
            rng.shuffle(rows)

        with (args.output_dir / f"{out_name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        action_totals[out_name] = Counter(row["output"]["next_action"] for row in rows)
        manifest_splits[out_name] = len(rows)

    manifest = {
        "source": "CrossWOZ (THU-CoAI, TACL 2020, Apache-2.0) -- real human-annotated Chinese task-oriented dialogue",
        "source_url": "https://github.com/thu-coai/CrossWOZ",
        "seed": args.seed,
        "splits": manifest_splits,
        "next_action_distribution": {split: dict(counts) for split, counts in action_totals.items()},
        "notice": (
            "input_text and the extracted domain/constraints/requested_fields come from CrossWOZ's own "
            "human annotation. next_action is this project's own routing design layered on top -- CrossWOZ "
            "does not label which backend action a system should take."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
