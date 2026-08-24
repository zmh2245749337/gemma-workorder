"""Prompt, SFT and metric helpers shared by QLoRA training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .workorder import WorkOrderFields


SYSTEM_PROMPT = """你是设备运维工单结构化助手。只输出一个JSON对象，不要输出Markdown或解释。
你不能诊断设备，也不能制定维修决策。只从用户文本抽取事实；不知道的字段必须为null或空列表。
next_action只能是query_fault_code、query_asset_history或ask_clarification。
JSON字段必须包含：asset_name, asset_id, asset_type, fault_code, symptom, occurred_at,
actions_taken, parts_replaced, missing_fields, next_action。"""


def workorder_prompt(text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n用户故障描述：\n{text}\n\nJSON："


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is invalid JSONL") from error
            if "input_text" not in row or "output" not in row:
                raise ValueError(f"{path}:{number} requires input_text and output")
            WorkOrderFields(**row["output"])
            rows.append(row)
    return rows


def compare_fields(prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Field-level exact-match summary for a small controlled extraction set."""
    keys = list(WorkOrderFields.__dataclass_fields__)
    matched = [key for key in keys if prediction.get(key) == target.get(key)]
    return {"matched_fields": matched, "total_fields": len(keys), "exact_match": len(matched) == len(keys)}


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(records)
    if not total:
        return {"samples": 0, "json_valid_rate": 0.0, "field_exact_rate": 0.0, "tool_accuracy": 0.0, "record_exact_rate": 0.0}
    json_valid = sum(item["json_valid"] for item in records)
    total_fields = sum(item.get("total_fields", 0) for item in records)
    matched_fields = sum(len(item.get("matched_fields", [])) for item in records)
    tool_correct = sum(item.get("tool_correct", False) for item in records)
    exact = sum(item.get("exact_match", False) for item in records)
    return {
        "samples": total,
        "json_valid_rate": round(json_valid / total, 4),
        "field_exact_rate": round(matched_fields / total_fields, 4) if total_fields else 0.0,
        "tool_accuracy": round(tool_correct / total, 4),
        "record_exact_rate": round(exact / total, 4),
    }

