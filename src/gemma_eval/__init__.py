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
from .tool_use import AgentDecision, ToolCall, apply_policy, validate_decision

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
    "ToolCall",
    "apply_policy",
    "validate_decision",
]
