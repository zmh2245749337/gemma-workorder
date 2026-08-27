"""Canonical Agent traces shared by data export, SFT and analysis."""

from __future__ import annotations

import json
from typing import Any

from .contracts import validate_decision
from .policy import apply_policy
from .tool_registry import DEFAULT_TOOL_REGISTRY


TRACE_SCHEMA_VERSION = "1.0"


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def row_to_agent_trace(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a labelled dialogue turn into an auditable training trace.

    The source dataset has no real tool response, so the trace stops at the
    expected policy outcome instead of inventing execution results.
    """

    decision = validate_decision(row["output"])
    policy = apply_policy(decision)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": str(row["sample_id"]),
        "context": {
            "history": row["history"],
            "previous_state": row.get("previous_state", []),
            "current_user": row["current_user"],
        },
        "tools": DEFAULT_TOOL_REGISTRY.to_function_schemas(),
        "events": [
            {"stage": "user", "payload": {"content": row["current_user"]}},
            {"stage": "model_target", "payload": decision.to_dict()},
            {"stage": "policy_target", "payload": policy.to_dict()},
        ],
        "expected": decision.to_dict(),
        "benchmark_tags": list(row.get("challenge_categories", [])),
        "provenance": row.get("source", {}),
    }


def render_sft_pair(row: dict[str, Any]) -> tuple[str, str]:
    """Render the canonical trace into the exact prompt/target used by SFT."""

    from .tool_use_data import tool_use_prompt

    trace = row_to_agent_trace(row)
    context = trace["context"]
    prompt = tool_use_prompt(
        context["history"], context["current_user"], context["previous_state"]
    )
    return prompt, canonical_json(trace["expected"])
