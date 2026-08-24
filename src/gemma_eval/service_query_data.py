"""Prompt, SFT and metric helpers for the service-query task.

Direct successor to workorder_data.py -- same shape (prompt builder, JSONL
loader that validates against the dataclass, field-exact-match comparison,
aggregate metrics), generalised to ServiceQueryFields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .service_query import ServiceQueryFields


SYSTEM_PROMPT = """你是北京本地生活服务查询助手。只输出一个JSON对象，不要输出Markdown或解释。
你不能编造真实世界信息（比如具体地址、电话、评分）；这些事实只能来自后续的数据库查询结果，
你只负责从用户的话里抽取查询意图和查询条件。
domain只能是景点、酒店、餐馆、地铁、出租之一，或者为null（无法判断时）。
next_action只能是query_attraction_db、query_hotel_db、query_restaurant_db、query_metro_route、
request_taxi、ask_clarification之一。request_taxi会真实调度车辆，只有在出发地和目的地都明确的情况下
才能选择它，并且执行前必须经过人工确认，不能自动执行。
JSON字段必须包含：domain, constraints, requested_fields, missing_fields, next_action。"""


def service_query_prompt(text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n用户请求：\n{text}\n\nJSON："


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
            ServiceQueryFields(**row["output"])
            rows.append(row)
    return rows


def compare_fields(prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Field-level exact-match summary for a small controlled extraction set."""
    keys = list(ServiceQueryFields.__dataclass_fields__)
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
