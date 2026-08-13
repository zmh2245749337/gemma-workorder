"""Build a small, deterministic self-constructed dataset for Phase A/B experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ASSETS = [
    ("A区3号风机", "风机", "E07", "过载保护触发"),
    ("B区2号水泵", "水泵", "E12", "进水压力异常"),
    ("C区1号空调机组", "空调机组", "E21", "温度传感器异常"),
]
TIMES = ["今天上午", "今天下午", "昨晚", "夜间"]
SYMPTOMS = ["重启十分钟后再次停止", "运行两分钟后报警", "间歇停机", "面板持续告警"]


def record(sample_id: str, asset: tuple[str, str, str, str], include_time: bool, include_code: bool, seed: int) -> dict:
    name, asset_type, code, _ = asset
    time = TIMES[seed % len(TIMES)] if include_time else None
    symptom = SYMPTOMS[seed % len(SYMPTOMS)]
    code_text = f"面板显示{code}，" if include_code else ""
    time_text = f"{time}，" if time else ""
    text = f"{name}{time_text}{code_text}{symptom}，已尝试重启。"
    missing = []
    if not include_code:
        missing.append("故障码")
    if not include_time:
        missing.append("发生时间")
    return {
        "sample_id": sample_id,
        "input_text": text,
        "output": {
            "asset_name": name,
            "asset_id": None,
            "asset_type": asset_type,
            "fault_code": code if include_code else None,
            "symptom": symptom,
            "occurred_at": time,
            "actions_taken": ["重启"],
            "parts_replaced": [],
            "missing_fields": missing,
            "next_action": "query_fault_code" if include_code else "ask_clarification",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic work-order JSONL dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/workorder"))
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    rows = []
    for index in range(args.samples):
        rows.append(record(f"wo_{index:04d}", ASSETS[index % len(ASSETS)], index % 5 != 0, index % 4 != 0, index))
    random.shuffle(rows)
    splits = {"train": rows[: int(len(rows) * 0.7)], "validation": rows[int(len(rows) * 0.7) : int(len(rows) * 0.85)], "test": rows[int(len(rows) * 0.85) :]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {"seed": args.seed, "samples": args.samples, "splits": {name: len(items) for name, items in splits.items()}, "notice": "Synthetic, self-constructed controlled dataset; not enterprise production data."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

