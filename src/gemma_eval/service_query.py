"""Phase A foundations for the CrossWOZ-derived service-query prototype.

This module is the direct successor to workorder.py, generalised from
"设备运维工单" to "北京本地生活服务查询" (attraction / hotel / restaurant /
metro / taxi). The design philosophy is unchanged: a model only extracts
structured fields and chooses a small, allow-listed local action; anything
the model is not explicitly allowed to do is refused, and anything with a
real-world side effect (dispatching a taxi) requires human confirmation
before ``execute_local_tool`` is ever called for it.

Data provenance matters here, so it is worth repeating in code, not just in
docs: the *input text* and the *domain / constraints / requested_fields*
labels used to train and evaluate this come from CrossWOZ (THU-CoAI, TACL
2020, Apache-2.0, https://github.com/thu-coai/CrossWOZ) -- a real,
human-annotated Chinese dialogue corpus. ``next_action`` is NOT part of
CrossWOZ's own annotation; it is this project's own routing design, built
on top of real data (see scripts/build_service_query_dataset.py for exactly
how it is derived).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DOMAINS = ("景点", "酒店", "餐馆", "地铁", "出租")

ALLOWED_TOOLS = {
    "query_attraction_db",
    "query_hotel_db",
    "query_restaurant_db",
    "query_metro_route",
    "request_taxi",
    "ask_clarification",
}

# Tools with a real-world side effect. execute_local_tool() will run them,
# but callers (run_service_query / the demo script / evaluation) must check
# requires_human_confirmation() first and only execute after confirmation.
CONFIRMATION_REQUIRED_TOOLS = {"request_taxi"}

_DOMAIN_QUERY_TOOL = {
    "景点": "query_attraction_db",
    "酒店": "query_hotel_db",
    "餐馆": "query_restaurant_db",
    "地铁": "query_metro_route",
}

_DOMAIN_TABLE = {"景点": "attractions", "酒店": "hotels", "餐馆": "restaurants"}


@dataclass
class ServiceQueryFields:
    """The constrained output contract used by data generation and inference."""

    domain: str | None = None
    constraints: dict[str, str] = field(default_factory=dict)
    requested_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    next_action: str = "ask_clarification"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_fields(payload: dict[str, Any]) -> ServiceQueryFields:
    """Validate untrusted model JSON before any tool is considered."""

    allowed = set(ServiceQueryFields.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown service-query fields: {unknown}")
    if "constraints" in payload and payload["constraints"] is not None and not isinstance(payload["constraints"], dict):
        raise ValueError("constraints must be an object")
    for key in ("requested_fields", "missing_fields"):
        if key in payload and payload[key] is not None and not isinstance(payload[key], list):
            raise ValueError(f"{key} must be a list")
    domain = payload.get("domain")
    if domain is not None and domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain}")
    next_action = payload.get("next_action", "ask_clarification")
    if next_action not in ALLOWED_TOOLS:
        raise ValueError(f"next_action is not allow-listed: {next_action}")
    return ServiceQueryFields(**payload)


def extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object from an LLM answer and reject non-object values."""
    import re

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("model output must contain a JSON object")
    return value


def route_tool(fields: ServiceQueryFields) -> ToolCall:
    """Turn validated fields into a safe, minimal tool call.

    The model's own ``next_action`` is not trusted blindly -- it is only
    honoured when it is actually consistent with what the tool needs:
    ``request_taxi`` requires both 出发地 and 目的地 to be present, and a
    domain query tool must match the extracted domain. Anything
    inconsistent falls back to ask_clarification instead of guessing.
    """

    if fields.next_action == "request_taxi":
        if {"出发地", "目的地"} <= set(fields.constraints):
            return ToolCall(
                "request_taxi",
                {"出发地": fields.constraints["出发地"], "目的地": fields.constraints["目的地"]},
            )
        return ToolCall("ask_clarification", {"missing_fields": "、".join(fields.missing_fields) or "出发地、目的地"})

    expected_tool = _DOMAIN_QUERY_TOOL.get(fields.domain)
    if expected_tool and fields.next_action == expected_tool:
        return ToolCall(
            expected_tool,
            {"domain": fields.domain, "constraints": fields.constraints, "requested_fields": fields.requested_fields},
        )
    return ToolCall("ask_clarification", {"missing_fields": "、".join(fields.missing_fields) or "请补充查询条件"})


def requires_human_confirmation(call: ToolCall) -> bool:
    return call.name in CONFIRMATION_REQUIRED_TOOLS


def initialise_demo_database(path: Path, reference_dir: Path) -> None:
    """Load a real, trimmed subset of the CrossWOZ reference databases.

    ``reference_dir`` must contain attraction_db.json / hotel_db.json /
    restaurant_db.json in CrossWOZ's own ``[[name, record], ...]`` format.
    Only entities that actually appear in data/service_query/*.jsonl are
    kept (see scripts/build_service_query_dataset.py), so the demo database
    stays small while remaining 100% real data, not invented records.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS attractions (name TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS hotels (name TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS restaurants (name TEXT PRIMARY KEY, record TEXT NOT NULL);
            """
        )
        conn.execute("DELETE FROM attractions")
        conn.execute("DELETE FROM hotels")
        conn.execute("DELETE FROM restaurants")
        file_to_table = {
            "attraction_db.json": "attractions",
            "hotel_db.json": "hotels",
            "restaurant_db.json": "restaurants",
        }
        for filename, table in file_to_table.items():
            file_path = reference_dir / filename
            if not file_path.exists():
                continue
            with file_path.open(encoding="utf-8") as handle:
                entries = json.load(handle)
            conn.executemany(
                f"INSERT OR REPLACE INTO {table} VALUES (?, ?)",
                [(name, json.dumps(record, ensure_ascii=False)) for name, record in entries],
            )


def _matches(record: dict[str, Any], constraints: dict[str, str]) -> bool:
    """Demo-grade constraint matching (substring compare on stringified
    field values) -- good enough to show a real record coming back, not a
    production search/ranking engine."""

    for slot, value in constraints.items():
        record_value = record.get(slot)
        if record_value is None:
            continue
        left, right = str(value), str(record_value)
        if left not in right and right not in left:
            return False
    return True


def execute_local_tool(path: Path, call: ToolCall) -> dict[str, Any]:
    """Execute only allow-listed, read-only lookups (plus the taxi dispatch
    stand-in). Callers must check ``requires_human_confirmation`` first and
    only reach this function for ``request_taxi`` after a human confirms.
    """

    if call.name == "request_taxi":
        return {
            "tool": call.name,
            "status": "ok",
            "record": {
                "出发地": call.arguments.get("出发地"),
                "目的地": call.arguments.get("目的地"),
                "调度状态": "人工确认后已派车",
            },
        }
    table = _DOMAIN_TABLE.get({"query_attraction_db": "景点", "query_hotel_db": "酒店", "query_restaurant_db": "餐馆"}.get(call.name, ""))
    if not table:
        return {"tool": call.name, "status": "not_executed", "reason": "clarification or an unsupported tool is not a database read"}
    constraints = call.arguments.get("constraints", {}) or {}
    name = constraints.get("名称")
    with sqlite3.connect(path) as conn:
        if name:
            row = conn.execute(f"SELECT record FROM {table} WHERE name = ?", (name,)).fetchone()
            if row:
                return {"tool": call.name, "status": "ok", "records": [json.loads(row[0])]}
            return {"tool": call.name, "status": "not_found", "records": []}
        rows = conn.execute(f"SELECT record FROM {table}").fetchall()
        candidates = [json.loads(item[0]) for item in rows]
        matched = [item for item in candidates if _matches(item, constraints)]
        return {"tool": call.name, "status": "ok" if matched else "not_found", "records": matched[:5]}


def draft_response(fields: ServiceQueryFields, call: ToolCall, tool_result: dict[str, Any] | None) -> dict[str, Any]:
    status = "requires_human_confirmation" if requires_human_confirmation(call) else "auto_executed"
    return {
        "status": status,
        "extracted_fields": fields.to_dict(),
        "tool_call": call.to_dict(),
        "tool_result": tool_result,
        "safety_note": "查询结果来自本地真实参考数据（CrossWOZ）；打车类调度动作必须经人工确认后才会真正执行。",
    }


def run_service_query(fields_payload: dict[str, Any], db_path: Path) -> dict[str, Any]:
    """End-to-end pipeline given already-extracted fields (e.g. the fine-tuned
    model's parsed JSON output). Mirrors workorder.py's run_phase_a, minus a
    rule-based text parser: CrossWOZ spans five open domains and dozens of
    slots, and a general-purpose Chinese rule-based NLU parser for all of
    them is a different, much larger project than this one -- tests and the
    demo script exercise this pipeline with representative structured
    payloads instead of a hand-rolled parser.
    """

    fields = validate_fields(fields_payload)
    call = route_tool(fields)
    if requires_human_confirmation(call):
        return {
            "structured_fields": fields.to_dict(),
            "tool_call": call.to_dict(),
            "tool_result": None,
            "draft": draft_response(fields, call, None),
        }
    result = execute_local_tool(db_path, call)
    return {
        "structured_fields": fields.to_dict(),
        "tool_call": call.to_dict(),
        "tool_result": result,
        "draft": draft_response(fields, call, result),
    }
