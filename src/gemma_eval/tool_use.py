"""Compatibility facade for the Agent tool-use mainline.

New code should import focused modules directly.  This facade keeps the
original imports stable for existing scripts, tests and saved notebooks.
"""

from .contracts import (
    DECISIONS,
    DOMAINS,
    AgentDecision,
    ToolCall,
    extract_json,
    strict_json_loads,
    validate_decision,
)
from .local_tools import execute_tool, initialise_demo_database
from .policy import PolicyOutcome, apply_policy
from .runtime import AgentRuntime, RuntimeEvent, run_decision
from .tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry, ToolSpec


TOOL_REGISTRY = DEFAULT_TOOL_REGISTRY.as_dict()

__all__ = [
    "DECISIONS",
    "DOMAINS",
    "TOOL_REGISTRY",
    "AgentDecision",
    "AgentRuntime",
    "PolicyOutcome",
    "RuntimeEvent",
    "ToolCall",
    "ToolRegistry",
    "ToolSpec",
    "apply_policy",
    "execute_tool",
    "extract_json",
    "initialise_demo_database",
    "run_decision",
    "strict_json_loads",
    "validate_decision",
]
