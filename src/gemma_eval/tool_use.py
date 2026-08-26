"""Context-aware tool use primitives for the current project mainline.

The model proposes a structured decision.  Deterministic code validates the
proposal, applies the tool policy, and only then executes a local tool.  Risk
is deliberately not predicted by the model: it belongs to the tool registry.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DOMAINS = ("景点", "酒店", "餐馆", "地铁", "出租")
DECISIONS = ("call_tool", "ask_user", "no_tool")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    domain: str
    risk: str
    required_arguments: tuple[str, ...] = ()


TOOL_REGISTRY = {
    "query_attraction_db": ToolSpec("query_attraction_db", "景点", "read_only"),
    "query_hotel_db": ToolSpec("query_hotel_db", "酒店", "read_only"),
    "query_restaurant_db": ToolSpec("query_restaurant_db", "餐馆", "read_only"),
    "query_metro_route": ToolSpec(
        "query_metro_route", "地铁", "read_only", ("出发地", "目的地")
    ),
    "request_taxi": ToolSpec(
        "request_taxi", "出租", "side_effect", ("出发地", "目的地")
    ),
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentDecision:
    belief_state: list[dict[str, Any]]
    decision: str
    tool_call: ToolCall | None
    missing_slots: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_call"] = self.tool_call.to_dict() if self.tool_call else None
        return payload


@dataclass(frozen=True)
class PolicyOutcome:
    status: str
    reason: str
    tool_call: ToolCall | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_call"] = self.tool_call.to_dict() if self.tool_call else None
        return payload


def strict_json_loads(text: str) -> dict[str, Any]:
    """Parse a response only when the complete response is one JSON object."""

    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("model output must be one JSON object")
    return value


def extract_json(text: str) -> dict[str, Any]:
    """Tolerant parser used for semantic evaluation after strict evaluation."""

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("model output does not contain a JSON object")
        candidate = text[start : end + 1]
    return strict_json_loads(candidate)


def _validate_belief_state(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("belief_state must be a list")
    goals: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"belief_state[{index}] must be an object")
        allowed = {"goal_id", "domain", "constraints", "requested_fields"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"belief_state[{index}] has unknown fields: {sorted(unknown)}")
        domain = raw.get("domain")
        if domain not in DOMAINS:
            raise ValueError(f"belief_state[{index}] has unknown domain: {domain}")
        constraints = raw.get("constraints", {})
        requested = raw.get("requested_fields", [])
        if not isinstance(constraints, dict):
            raise ValueError(f"belief_state[{index}].constraints must be an object")
        if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
            raise ValueError(f"belief_state[{index}].requested_fields must be a string list")
        goals.append(
            {
                "goal_id": int(raw.get("goal_id", index + 1)),
                "domain": domain,
                "constraints": constraints,
                "requested_fields": requested,
            }
        )
    return sorted(goals, key=lambda item: (item["goal_id"], item["domain"]))


def validate_decision(payload: dict[str, Any]) -> AgentDecision:
    """Validate untrusted model output without executing anything."""

    allowed = {"belief_state", "decision", "tool_call", "missing_slots"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown decision fields: {sorted(unknown)}")

    belief_state = _validate_belief_state(payload.get("belief_state", []))
    decision = payload.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    missing_slots = payload.get("missing_slots", [])
    if not isinstance(missing_slots, list) or not all(isinstance(item, str) for item in missing_slots):
        raise ValueError("missing_slots must be a string list")

    raw_call = payload.get("tool_call")
    call: ToolCall | None = None
    if decision == "call_tool":
        if not isinstance(raw_call, dict):
            raise ValueError("call_tool decision requires tool_call")
        if set(raw_call) - {"name", "arguments"}:
            raise ValueError("tool_call contains unknown fields")
        name, arguments = raw_call.get("name"), raw_call.get("arguments", {})
        if name not in TOOL_REGISTRY:
            raise ValueError(f"tool is not registered: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("tool_call.arguments must be an object")
        call = ToolCall(name, arguments)
        if missing_slots:
            raise ValueError("call_tool decision must not include missing_slots")
    elif raw_call is not None:
        raise ValueError(f"{decision} decision must not include tool_call")
    if decision == "ask_user" and not missing_slots:
        raise ValueError("ask_user decision requires at least one missing slot")
    if decision == "no_tool" and missing_slots:
        raise ValueError("no_tool decision must not include missing_slots")

    return AgentDecision(belief_state, decision, call, missing_slots)


def apply_policy(decision: AgentDecision) -> PolicyOutcome:
    """Convert a valid model proposal into an execution decision."""

    if decision.decision == "no_tool":
        return PolicyOutcome("no_tool", "the current turn does not require a tool", None)
    if decision.decision == "ask_user":
        return PolicyOutcome("needs_input", "required information is missing", None)

    assert decision.tool_call is not None
    call = decision.tool_call
    spec = TOOL_REGISTRY[call.name]
    missing = [key for key in spec.required_arguments if not call.arguments.get(key)]
    if missing:
        return PolicyOutcome("needs_input", f"missing required arguments: {', '.join(missing)}", None)
    if spec.risk == "side_effect":
        return PolicyOutcome("pending_confirmation", "side-effect tool requires human confirmation", call)
    return PolicyOutcome("auto_execute", "read-only tool passed validation", call)


def initialise_demo_database(path: Path, reference_dir: Path) -> None:
    """Build a small local database from the checked-in CrossWOZ subset."""

    path.parent.mkdir(parents=True, exist_ok=True)
    domain_files = {
        "景点": "attraction_db.json",
        "酒店": "hotel_db.json",
        "餐馆": "restaurant_db.json",
    }
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                domain TEXT NOT NULL,
                name TEXT NOT NULL,
                record TEXT NOT NULL,
                PRIMARY KEY (domain, name)
            );
            CREATE TABLE IF NOT EXISTS metro (
                place TEXT PRIMARY KEY,
                station TEXT NOT NULL
            );
            DELETE FROM entities;
            DELETE FROM metro;
            """
        )
        for domain, filename in domain_files.items():
            source = reference_dir / filename
            if not source.exists():
                continue
            entries = json.loads(source.read_text(encoding="utf-8"))
            connection.executemany(
                "INSERT OR REPLACE INTO entities VALUES (?, ?, ?)",
                [(domain, name, json.dumps(record, ensure_ascii=False)) for name, record in entries],
            )
        metro_source = reference_dir / "metro_db.json"
        if metro_source.exists():
            entries = json.loads(metro_source.read_text(encoding="utf-8"))
            connection.executemany(
                "INSERT OR REPLACE INTO metro VALUES (?, ?)",
                [(name, str(record.get("地铁", ""))) for name, record in entries if record.get("地铁")],
            )


def _matches(record: dict[str, Any], constraints: dict[str, Any]) -> bool:
    for slot, expected in constraints.items():
        if slot not in record:
            return False
        actual = record[slot]
        if isinstance(expected, list):
            if not all(str(item) in str(actual) for item in expected):
                return False
        elif str(expected) not in str(actual) and str(actual) not in str(expected):
            return False
    return True


def _execute_entity_query(path: Path, spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    constraints = arguments.get("constraints", {}) or {}
    if not isinstance(constraints, dict):
        return {"status": "invalid_arguments", "reason": "constraints must be an object"}
    requested = arguments.get("requested_fields", []) or []
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT record FROM entities WHERE domain = ?", (spec.domain,)
        ).fetchall()
    candidates = [json.loads(row[0]) for row in rows]
    matched = [record for record in candidates if _matches(record, constraints)]
    if requested:
        records = [
            {key: record.get(key) for key in ["名称", *requested] if key in record}
            for record in matched[:5]
        ]
    else:
        records = matched[:5]
    return {"status": "ok" if records else "not_found", "records": records}


def execute_tool(path: Path, call: ToolCall, *, confirmed: bool = False) -> dict[str, Any]:
    """Execute a validated call; side effects still require explicit confirmation."""

    spec = TOOL_REGISTRY[call.name]
    if spec.risk == "side_effect":
        if not confirmed:
            raise PermissionError("side-effect tool requires explicit confirmation")
        return {
            "status": "simulated",
            "message": "已记录人工确认；演示环境不会调用真实派车接口。",
            "arguments": call.arguments,
        }
    if call.name == "query_metro_route":
        start, destination = call.arguments["出发地"], call.arguments["目的地"]
        with sqlite3.connect(path) as connection:
            start_row = connection.execute("SELECT station FROM metro WHERE place = ?", (start,)).fetchone()
            destination_row = connection.execute(
                "SELECT station FROM metro WHERE place = ?", (destination,)
            ).fetchone()
        if not start_row or not destination_row:
            return {"status": "not_found", "route": None}
        return {
            "status": "ok",
            "route": {"出发地": start, "起点站": start_row[0], "目的地": destination, "终点站": destination_row[0]},
        }
    return _execute_entity_query(path, spec, call.arguments)


def run_decision(payload: dict[str, Any], db_path: Path) -> dict[str, Any]:
    decision = validate_decision(payload)
    policy = apply_policy(decision)
    result = None
    if policy.status == "auto_execute" and policy.tool_call:
        result = execute_tool(db_path, policy.tool_call)
    return {
        "decision": decision.to_dict(),
        "policy": policy.to_dict(),
        "tool_result": result,
    }
