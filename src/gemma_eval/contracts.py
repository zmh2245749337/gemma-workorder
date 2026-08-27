"""Output contract and strict validation for untrusted model decisions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


DOMAINS = ("景点", "酒店", "餐馆", "地铁", "出租")
DECISIONS = ("call_tool", "ask_user", "no_tool")


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


def strict_json_loads(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("model output must be one JSON object")
    return value


def extract_json(text: str) -> dict[str, Any]:
    """Tolerant parser used only after strict-format scoring."""

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


def validate_decision(
    payload: dict[str, Any], registry: ToolRegistry = DEFAULT_TOOL_REGISTRY
) -> AgentDecision:
    """Validate shape, decision invariants and tool allow-list membership."""

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
        if name not in registry:
            raise ValueError(f"tool is not registered: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("tool_call.arguments must be an object")
        registry[name].validate_argument_shape(arguments)
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
