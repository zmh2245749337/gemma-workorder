"""Build a self-constructed dataset for Phase A/B experiments.

This generator is deliberately still synthetic and template-driven (see the
"适用边界" section of the README) — it does not claim to be production data.
Compared with the original 90-sample version, three things changed on
purpose:

1. Fault codes are no longer 1:1 with a specific asset. Each asset *type*
   has two possible codes, so the model has to read the code out of the
   text instead of memorising an asset-name -> code lookup table.
2. The ``query_asset_history`` route is now actually exercised. The
   original generator only ever produced ``query_fault_code`` and
   ``ask_clarification`` targets, even though both the JSON schema and the
   tool whitelist have supported a third route (asset-history lookup) since
   the beginning.
3. Sentence structure, symptom vocabulary, actions and replaced parts are
   sampled from larger pools with two sentence skeletons per scenario, so
   the dataset is not a single fixed sentence with three slots swapped in.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ASSET_INSTANCES = [
    ("A区3号风机", "风机"),
    ("D区5号风机", "风机"),
    ("B区2号水泵", "水泵"),
    ("E区1号水泵", "水泵"),
    ("C区1号空调机组", "空调机组"),
    ("G区2号空调机组", "空调机组"),
    ("F区2号配电柜", "配电柜"),
]

FAULT_CODES_BY_TYPE = {
    "风机": ["E07", "E09"],
    "水泵": ["E12", "E15"],
    "空调机组": ["E21", "E23"],
    "配电柜": ["E31", "E33"],
}

TIMES = ["今天上午", "今天下午", "今天凌晨", "昨晚", "昨天下午", "夜间", "刚才", "上一班交接时"]

SYMPTOMS = [
    "重启十分钟后再次停止", "运行两分钟后报警", "间歇停机", "面板持续告警",
    "运行时有明显异响", "启动后立即跳闸", "温度异常升高后自动停机",
    "输出压力波动很大", "指示灯不断闪烁", "远程复位没有效果",
]

ACTIONS = ["重启", "断电重启", "复位", "清理滤网"]
PARTS = ["风扇", "滤网", "传感器", "接触器"]

# (scenario, relative weight): "fault_with_code" and "fault_missing_code" /
# "fault_missing_time" cover the query_fault_code / ask_clarification routes
# the original generator had; "history_lookup" is new.
SCENARIOS = ["fault_with_code", "fault_missing_code", "fault_missing_time", "history_lookup"]
SCENARIO_WEIGHTS = [0.40, 0.20, 0.15, 0.25]

FAULT_TEMPLATES = [
    "{asset}{time_clause}{code_clause}{symptom}，{action_clause}。",
    "巡检发现{asset}{code_clause}{symptom}，{time_clause}{action_clause}，仍未恢复。",
]
HISTORY_TEMPLATE = "{asset}{time_clause}又出现{symptom}，{action_clause}，麻烦查一下这台设备之前有没有类似的维修记录。"


def _action_clause(actions: list[str], parts: list[str]) -> str:
    if actions and parts:
        return "已" + "、".join(actions) + "，并更换了" + "、".join(parts)
    if actions:
        return "已" + "、".join(actions)
    if parts:
        return "已更换" + "、".join(parts)
    return "尚未处理"


def build_record(sample_id: str, rng: random.Random) -> dict:
    asset_name, asset_type = rng.choice(ASSET_INSTANCES)
    fault_code = rng.choice(FAULT_CODES_BY_TYPE[asset_type])
    symptom = rng.choice(SYMPTOMS)
    time_value = rng.choice(TIMES)
    scenario = rng.choices(SCENARIOS, weights=SCENARIO_WEIGHTS)[0]

    is_history = scenario == "history_lookup"
    include_time = scenario != "fault_missing_time"
    include_code = scenario in ("fault_with_code",) or (scenario == "history_lookup" and False)
    # history_lookup never states the code in the text (see module docstring
    # point 2), so it must not be marked as included even though the asset
    # legitimately has one on file.
    if is_history:
        include_code = False

    actions = rng.sample(ACTIONS, k=rng.choice([0, 1, 1, 2]))
    parts = rng.sample(PARTS, k=rng.choice([0, 0, 0, 1]))

    time_clause = f"{time_value}，" if include_time else ""
    code_clause = f"面板显示{fault_code}，" if include_code else ""
    action_clause = _action_clause(actions, parts)

    if is_history:
        text = HISTORY_TEMPLATE.format(
            asset=asset_name, time_clause=time_clause, symptom=symptom, action_clause=action_clause
        )
    else:
        template = rng.choice(FAULT_TEMPLATES)
        text = template.format(
            asset=asset_name, time_clause=time_clause, code_clause=code_clause,
            symptom=symptom, action_clause=action_clause,
        )

    missing = []
    if not include_code:
        missing.append("故障码")
    if not include_time:
        missing.append("发生时间")

    if is_history:
        next_action = "query_asset_history"
    elif include_code:
        next_action = "query_fault_code"
    else:
        next_action = "ask_clarification"

    return {
        "sample_id": sample_id,
        "input_text": text,
        "output": {
            "asset_name": asset_name,
            "asset_id": None,
            "asset_type": asset_type,
            "fault_code": fault_code if include_code else None,
            "symptom": symptom,
            "occurred_at": time_value if include_time else None,
            "actions_taken": actions,
            "parts_replaced": parts,
            "missing_fields": missing,
            "next_action": next_action,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic work-order JSONL dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/workorder"))
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    seen_text: set[str] = set()
    attempts = 0
    max_attempts = max(args.samples * 30, 1000)
    while len(rows) < args.samples and attempts < max_attempts:
        attempts += 1
        candidate = build_record(f"wo_{len(rows):04d}", rng)
        if candidate["input_text"] in seen_text:
            continue
        seen_text.add(candidate["input_text"])
        rows.append(candidate)
    if len(rows) < args.samples:
        raise RuntimeError(
            f"only generated {len(rows)}/{args.samples} unique samples after {attempts} attempts; "
            "widen the vocabulary pools or lower --samples"
        )
    rng.shuffle(rows)

    total = len(rows)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    splits = {
        "train": rows[:train_end],
        "validation": rows[train_end:val_end],
        "test": rows[val_end:],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "samples": total,
        "splits": {name: len(items) for name, items in splits.items()},
        "next_action_distribution": {
            action: sum(1 for row in rows if row["output"]["next_action"] == action)
            for action in ("query_fault_code", "ask_clarification", "query_asset_history")
        },
        "notice": "Synthetic, self-constructed controlled dataset; not enterprise production data.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
