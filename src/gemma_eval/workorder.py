"""Phase A foundations for the Gemma-WorkOrder prototype.

The module deliberately keeps the domain facts outside the model.  A model (or
the deterministic baseline in this file) only extracts fields and chooses a
small, allow-listed local tool.  Fault-code meanings and maintenance history
are read from SQLite so a 1B model is never presented as a diagnostic authority.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ALLOWED_TOOLS = {
    "query_asset",
    "query_asset_history",
    "query_fault_code",
    "query_manual",
    "draft_work_order",
    "ask_clarification",
}
REQUIRED_FIELDS = ("asset_name", "fault_code", "symptom", "occurred_at")


@dataclass
class WorkOrderFields:
    """The constrained output contract used by data generation and inference."""

    asset_name: str | None = None
    asset_id: str | None = None
    asset_type: str | None = None
    fault_code: str | None = None
    symptom: str | None = None
    occurred_at: str | None = None
    actions_taken: list[str] = field(default_factory=list)
    parts_replaced: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    next_action: str = "ask_clarification"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def rule_based_parse(text: str) -> WorkOrderFields:
    """A transparent Phase A baseline, *not* a replacement for Gemma fine-tuning.

    It provides a stable reference for schema validation, tool execution and
    future Base-vs-QLoRA comparisons.
    """

    asset = _find(r"(?:[A-Z]\s*区)?\s*\d+\s*号(?:风机|空调机组|水泵|配电柜)", text)
    if asset:
        asset = re.sub(r"\s+", "", asset)
    asset_type = next((kind for kind in ("风机", "空调机组", "水泵", "配电柜") if kind in text), None)
    # Python's Unicode \b treats Chinese characters as word characters, so
    # ``显示E07`` has no word boundary before E.  Use an ASCII-only boundary.
    fault_code = _find(r"(?<![A-Za-z0-9])E\s*[- ]?\d{2,3}(?![A-Za-z0-9])", text)
    if fault_code:
        fault_code = re.sub(r"[ -]", "", fault_code.upper())
    occurred_at = next((value for value in ("今天上午", "今天下午", "上午", "下午", "昨晚", "夜间") if value in text), None)
    actions = [item for item in ("重启", "断电重启", "复位", "清理滤网") if item in text]
    parts = [item for item in ("风扇", "滤网", "传感器", "接触器") if f"更换{item}" in text]
    symptom = text.strip().replace("\n", " ")

    fields = WorkOrderFields(
        asset_name=asset,
        asset_type=asset_type,
        fault_code=fault_code,
        symptom=symptom,
        occurred_at=occurred_at,
        actions_taken=_dedupe(actions),
        parts_replaced=_dedupe(parts),
    )
    missing: list[str] = []
    if not fields.asset_name:
        missing.append("设备名称或编号")
    if not fields.fault_code:
        missing.append("故障码")
    if not fields.occurred_at:
        missing.append("发生时间")
    fields.missing_fields = missing
    fields.next_action = "query_fault_code" if fields.fault_code else "ask_clarification"
    return fields


def validate_fields(payload: dict[str, Any]) -> WorkOrderFields:
    """Validate untrusted model JSON before any tool is considered."""

    allowed = set(WorkOrderFields.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown work-order fields: {unknown}")
    for key in ("actions_taken", "parts_replaced", "missing_fields"):
        if key in payload and not isinstance(payload[key], list):
            raise ValueError(f"{key} must be a list")
    next_action = payload.get("next_action", "ask_clarification")
    if next_action not in ALLOWED_TOOLS:
        raise ValueError(f"next_action is not allow-listed: {next_action}")
    return WorkOrderFields(**payload)


def extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object from an LLM answer and reject non-object values."""

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("model output must contain a JSON object")
    return value


def route_tool(fields: WorkOrderFields) -> ToolCall:
    """Turn validated fields into a safe, minimal tool call."""

    if fields.next_action == "query_fault_code" and fields.fault_code:
        return ToolCall("query_fault_code", {"code": fields.fault_code, "asset_type": fields.asset_type or "未知"})
    if fields.asset_name and fields.next_action == "query_asset_history":
        return ToolCall("query_asset_history", {"asset_name": fields.asset_name})
    return ToolCall("ask_clarification", {"missing_fields": "、".join(fields.missing_fields) or "请补充设备信息"})


def initialise_demo_database(path: Path) -> None:
    """Create small non-sensitive local reference data for a reproducible demo."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                asset_name TEXT UNIQUE NOT NULL,
                asset_type TEXT NOT NULL,
                model TEXT NOT NULL,
                location TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fault_codes (
                code TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                meaning TEXT NOT NULL,
                manual_ref TEXT NOT NULL,
                PRIMARY KEY (code, asset_type)
            );
            CREATE TABLE IF NOT EXISTS asset_history (
                asset_name TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                summary TEXT NOT NULL
            );
            """
        )
        conn.execute("DELETE FROM assets")
        conn.execute("DELETE FROM fault_codes")
        conn.execute("DELETE FROM asset_history")
        conn.executemany(
            "INSERT INTO assets VALUES (?, ?, ?, ?, ?)",
            [
                ("FAN-A-03", "A区3号风机", "风机", "FAN-X200", "A区机房"),
                ("PUMP-B-02", "B区2号水泵", "水泵", "PUMP-P80", "B区泵房"),
                ("AC-C-01", "C区1号空调机组", "空调机组", "AC-900", "C区控制室"),
            ],
        )
        conn.executemany(
            "INSERT INTO fault_codes VALUES (?, ?, ?, ?)",
            [
                ("E07", "风机", "过载保护触发", "FAN-X200 手册 4.2"),
                ("E12", "水泵", "进水压力异常", "PUMP-P80 手册 3.1"),
                ("E21", "空调机组", "温度传感器异常", "AC-900 手册 5.4"),
            ],
        )
        conn.executemany(
            "INSERT INTO asset_history VALUES (?, ?, ?)",
            [
                ("A区3号风机", "2026-06-18", "发生过一次滤网堵塞导致的过载告警"),
                ("B区2号水泵", "2026-07-03", "更换过进水压力传感器"),
            ],
        )


def execute_local_tool(path: Path, call: ToolCall) -> dict[str, Any]:
    """Execute only allow-listed, read-only SQLite lookups."""

    if call.name not in {"query_fault_code", "query_asset", "query_asset_history"}:
        return {"tool": call.name, "status": "not_executed", "reason": "clarification or drafting is not a database read"}
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        if call.name == "query_fault_code":
            row = conn.execute(
                "SELECT code, asset_type, meaning, manual_ref FROM fault_codes WHERE code = ? AND asset_type = ?",
                (call.arguments["code"], call.arguments["asset_type"]),
            ).fetchone()
        elif call.name == "query_asset":
            row = conn.execute("SELECT * FROM assets WHERE asset_name = ?", (call.arguments["asset_name"],)).fetchone()
        else:
            rows = conn.execute(
                "SELECT occurred_at, summary FROM asset_history WHERE asset_name = ? ORDER BY occurred_at DESC",
                (call.arguments["asset_name"],),
            ).fetchall()
            return {"tool": call.name, "status": "ok", "records": [dict(item) for item in rows]}
    return {"tool": call.name, "status": "ok" if row else "not_found", "record": dict(row) if row else None}


def draft_work_order(fields: WorkOrderFields, tool_result: dict[str, Any]) -> dict[str, Any]:
    """Create a human-reviewable draft; this function never submits a work order."""

    return {
        "status": "draft_requires_human_confirmation",
        "work_order": fields.to_dict(),
        "trusted_reference": tool_result,
        "safety_note": "故障码含义来自本地参考表；该草稿不构成维修决策。",
    }


def run_phase_a(text: str, db_path: Path) -> dict[str, Any]:
    """End-to-end executable Phase A baseline used by the demo script and tests."""

    fields = rule_based_parse(text)
    validated = validate_fields(fields.to_dict())
    call = route_tool(validated)
    result = execute_local_tool(db_path, call)
    return {
        "parser": "phase_a_rule_based_reference",
        "structured_fields": validated.to_dict(),
        "tool_call": call.to_dict(),
        "tool_result": result,
        "draft": draft_work_order(validated, result),
    }
