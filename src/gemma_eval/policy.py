"""Deterministic execution policy applied after model-output validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import AgentDecision, ToolCall
from .tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


@dataclass(frozen=True)
class PolicyOutcome:
    status: str
    reason: str
    tool_call: ToolCall | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_call"] = self.tool_call.to_dict() if self.tool_call else None
        return payload


def apply_policy(
    decision: AgentDecision, registry: ToolRegistry = DEFAULT_TOOL_REGISTRY
) -> PolicyOutcome:
    if decision.decision == "no_tool":
        return PolicyOutcome("no_tool", "the current turn does not require a tool", None)
    if decision.decision == "ask_user":
        return PolicyOutcome("needs_input", "required information is missing", None)

    assert decision.tool_call is not None
    call = decision.tool_call
    spec = registry[call.name]
    missing = [key for key in spec.required_arguments if not call.arguments.get(key)]
    if missing:
        return PolicyOutcome(
            "needs_input", f"missing required arguments: {', '.join(missing)}", None
        )
    if spec.risk == "side_effect":
        return PolicyOutcome(
            "pending_confirmation", "side-effect tool requires human confirmation", call
        )
    return PolicyOutcome("auto_execute", "read-only tool passed validation", call)
