"""Gemma context-aware tool-use training and guarded execution utilities."""

from .decoding import GenerationResult, manual_generate
from .gemma3_core import (
    Gemma3CoreConfig,
    Gemma3CoreForCausalLM,
    Gemma3DecoderLayer,
    Gemma3RMSNorm,
    HybridKVCache,
)
from .modeling import DEFAULT_MODEL_ID, ModelBundle, load_model_bundle
from .contracts import AgentDecision, ToolCall, validate_decision
from .policy import apply_policy
from .runtime import AgentRuntime
from .tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry

__all__ = [
    "DEFAULT_MODEL_ID",
    "GenerationResult",
    "Gemma3CoreConfig",
    "Gemma3CoreForCausalLM",
    "Gemma3DecoderLayer",
    "Gemma3RMSNorm",
    "HybridKVCache",
    "ModelBundle",
    "load_model_bundle",
    "manual_generate",
    "AgentDecision",
    "AgentRuntime",
    "DEFAULT_TOOL_REGISTRY",
    "ToolCall",
    "ToolRegistry",
    "apply_policy",
    "validate_decision",
]
