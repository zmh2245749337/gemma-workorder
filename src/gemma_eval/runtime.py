"""Guarded model -> policy -> tool orchestration with an auditable trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import extract_json, validate_decision
from .local_tools import execute_tool
from .policy import apply_policy
from .tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


@dataclass(frozen=True)
class RuntimeEvent:
    stage: str
    status: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRuntime:
    """Run one untrusted model proposal through deterministic guardrails."""

    def __init__(
        self, db_path: Path, registry: ToolRegistry = DEFAULT_TOOL_REGISTRY
    ) -> None:
        self.db_path = db_path
        self.registry = registry

    def run_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = uuid4().hex[:12]
        decision = validate_decision(payload, self.registry)
        events = [RuntimeEvent("contract", "accepted", decision.to_dict())]

        policy = apply_policy(decision, self.registry)
        events.append(RuntimeEvent("policy", policy.status, policy.to_dict()))

        result = None
        if policy.status == "auto_execute" and policy.tool_call:
            result = execute_tool(
                self.db_path, policy.tool_call, registry=self.registry
            )
            events.append(RuntimeEvent("tool", str(result.get("status", "ok")), result))
        else:
            events.append(
                RuntimeEvent(
                    "tool",
                    "skipped",
                    {"reason": "policy did not grant automatic execution"},
                )
            )

        return {
            "trace_id": trace_id,
            "decision": decision.to_dict(),
            "policy": policy.to_dict(),
            "tool_result": result,
            "events": [event.to_dict() for event in events],
        }

    def run_text(self, model_output: str) -> dict[str, Any]:
        return self.run_payload(extract_json(model_output))


def run_decision(payload: dict[str, Any], db_path: Path) -> dict[str, Any]:
    return AgentRuntime(db_path).run_payload(payload)
