"""White-box Gemma 3 text decoder implemented with plain PyTorch.

The implementation intentionally keeps the main data flow visible.  It covers
the parts that distinguish Gemma 3 from a generic decoder-only Transformer:

* Gemma-style RMSNorm with ``(1 + weight)`` scaling;
* Q/K normalization and multi-query attention (MQA);
* local/global RoPE bases and the 5:1 sliding/global attention pattern;
* GeGLU feed-forward blocks and Gemma 3's four-norm residual layout;
* a layer-aware hybrid KV cache for prefill/decode.

It is an educational inference implementation, not a replacement for optimized
Transformers kernels.  ``copy_from_hf`` is provided so its numerical output can
be checked against the official Hugging Face implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


def make_layer_types(num_hidden_layers: int, pattern: int = 6) -> tuple[str, ...]:
    """Return Gemma 3's repeated local/local/local/local/local/global pattern."""
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if pattern <= 1:
        raise ValueError("pattern must be greater than 1")
    return tuple(
        FULL_ATTENTION if (layer_idx + 1) % pattern == 0 else SLIDING_ATTENTION
        for layer_idx in range(num_hidden_layers)
    )


@dataclass(frozen=True)
class Gemma3CoreConfig:
    """Configuration for the Gemma 3 1B text decoder.

    Defaults match ``google/gemma-3-1b-it``.  Tests use smaller values while
    exercising the same tensor layout and cache behavior.
    """

    vocab_size: int = 262_144
    hidden_size: int = 1_152
    intermediate_size: int = 6_912
    num_hidden_layers: int = 26
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    query_pre_attn_scalar: float = 256.0
    attention_dropout: float = 0.0
    attention_bias: bool = False
    attn_logit_softcapping: float | None = None
    sliding_window: int = 512
    sliding_window_pattern: int = 6
    rope_local_theta: float = 10_000.0
    rope_theta: float = 1_000_000.0
    pad_token_id: int = 0
    layer_types: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        expected = self.num_hidden_layers
        if self.layer_types and len(self.layer_types) != expected:
            raise ValueError("layer_types length must equal num_hidden_layers")
        if not self.layer_types:
            object.__setattr__(
                self,
                "layer_types",
                make_layer_types(self.num_hidden_layers, self.sliding_window_pattern),
            )

    @classmethod
    def from_hf_config(cls, config: Any) -> "Gemma3CoreConfig":
        """Build from a Transformers Gemma 3 text or causal-LM config."""
        text_config = getattr(config, "text_config", config)
        layer_types = tuple(
            getattr(text_config, "layer_types", ())
            or make_layer_types(
                int(text_config.num_hidden_layers),
                int(getattr(text_config, "sliding_window_pattern", 6)),
            )
        )
        rope_parameters = getattr(text_config, "rope_parameters", None) or {}
        local_rope = rope_parameters.get(SLIDING_ATTENTION) or {}
        global_rope = rope_parameters.get(FULL_ATTENTION) or {}
        return cls(
            vocab_size=int(text_config.vocab_size),
            hidden_size=int(text_config.hidden_size),
            intermediate_size=int(text_config.intermediate_size),
            num_hidden_layers=int(text_config.num_hidden_layers),
            num_attention_heads=int(text_config.num_attention_heads),
            num_key_value_heads=int(text_config.num_key_value_heads),
            head_dim=int(text_config.head_dim),
            rms_norm_eps=float(text_config.rms_norm_eps),
            query_pre_attn_scalar=float(text_config.query_pre_attn_scalar),
            attention_dropout=float(getattr(text_config, "attention_dropout", 0.0)),
            attention_bias=bool(getattr(text_config, "attention_bias", False)),
            attn_logit_softcapping=getattr(text_config, "attn_logit_softcapping", None),
            sliding_window=int(text_config.sliding_window),
            sliding_window_pattern=int(getattr(text_config, "sliding_window_pattern", 6)),
            rope_local_theta=float(
                local_rope.get("rope_theta", getattr(text_config, "rope_local_base_freq", 10_000.0))
            ),
            rope_theta=float(global_rope.get("rope_theta", getattr(text_config, "rope_theta", 1_000_000.0))),
            pad_token_id=int(getattr(text_config, "pad_token_id", 0)),
            layer_types=layer_types,
        )


class Gemma3RMSNorm(nn.Module):
    """Gemma 3 RMSNorm: normalize in FP32, then multiply by ``1 + weight``."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


class Gemma3RotaryEmbedding(nn.Module):
    """RoPE with separate base frequencies for local and global layers."""

    def __init__(self, config: Gemma3CoreConfig) -> None:
        super().__init__()
        local_inv_freq = self._make_inv_freq(config.head_dim, config.rope_local_theta)
        global_inv_freq = self._make_inv_freq(config.head_dim, config.rope_theta)
        self.register_buffer("local_inv_freq", local_inv_freq, persistent=False)
        self.register_buffer("global_inv_freq", global_inv_freq, persistent=False)

    @staticmethod
    def _make_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
        return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.local_inv_freq if layer_type == SLIDING_ATTENTION else self.global_inv_freq
        inv_freq = inv_freq.to(device=x.device, dtype=torch.float32)
        expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        freqs = (expanded @ position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    """Expand one MQA key/value head to the number of query heads."""
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if repetitions == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        repetitions,
        seq_len,
        head_dim,
    )
    return expanded.reshape(batch, num_kv_heads * repetitions, seq_len, head_dim)


@dataclass
class LayerKVCache:
    """K/V tensors for one layer plus their absolute starting position."""

    key: torch.Tensor
    value: torch.Tensor
    start_position: int = 0

    @property
    def length(self) -> int:
        return int(self.key.shape[-2])

    @property
    def next_position(self) -> int:
        return self.start_position + self.length


@dataclass
class HybridKVCache:
    """Layer-aware cache: local layers are cropped, global layers keep history."""

    layers: list[LayerKVCache | None]
    seen_tokens: int = 0

    @classmethod
    def empty(cls, num_hidden_layers: int) -> "HybridKVCache":
        return cls(layers=[None] * num_hidden_layers, seen_tokens=0)


def _append_to_cache(
    past: LayerKVCache | None,
    key: torch.Tensor,
    value: torch.Tensor,
    max_cache_length: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int, LayerKVCache]:
    if past is None:
        combined_key = key
        combined_value = value
        key_start = 0
    else:
        combined_key = torch.cat((past.key, key), dim=-2)
        combined_value = torch.cat((past.value, value), dim=-2)
        key_start = past.start_position

    stored_key = combined_key
    stored_value = combined_value
    stored_start = key_start
    if max_cache_length is not None and stored_key.shape[-2] > max_cache_length:
        tokens_to_drop = int(stored_key.shape[-2] - max_cache_length)
        stored_key = stored_key[..., tokens_to_drop:, :]
        stored_value = stored_value[..., tokens_to_drop:, :]
        stored_start += tokens_to_drop

    return (
        combined_key,
        combined_value,
        key_start,
        LayerKVCache(stored_key, stored_value, stored_start),
    )


def build_attention_mask(
    *,
    position_ids: torch.Tensor,
    key_start_position: int,
    key_length: int,
    dtype: torch.dtype,
    sliding_window: int | None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create an additive causal mask from absolute query/key positions."""
    device = position_ids.device
    key_positions = torch.arange(
        key_start_position,
        key_start_position + key_length,
        device=device,
    )
    allowed = key_positions[None, None, :] <= position_ids[:, :, None]
    if sliding_window is not None:
        lower_bound = position_ids[:, :, None] - sliding_window + 1
        allowed = allowed & (key_positions[None, None, :] >= lower_bound)
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask[:, None, :].bool()
    min_value = torch.finfo(dtype).min
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, min_value)
    return mask[:, None, :, :]


class Gemma3MLP(nn.Module):
    """GeGLU feed-forward network used by Gemma 3."""

    def __init__(
        self,
        config: Gemma3CoreConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, **factory)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, **factory)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class Gemma3SelfAttention(nn.Module):
    """Eager MQA attention with explicit RoPE, mask and cache handling."""

    def __init__(
        self,
        config: Gemma3CoreConfig,
        layer_idx: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        factory = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
            **factory,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
            **factory,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
            **factory,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            **factory,
        )
        self.q_norm = Gemma3RMSNorm(config.head_dim, config.rms_norm_eps, **factory)
        self.k_norm = Gemma3RMSNorm(config.head_dim, config.rms_norm_eps, **factory)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, query_length, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)

        max_cache_length = self.config.sliding_window if self.layer_type == SLIDING_ATTENTION else None
        key, value, key_start, present = _append_to_cache(
            past_key_value,
            key,
            value,
            max_cache_length,
        )
        relevant_padding = None
        if key_padding_mask is not None:
            relevant_padding = key_padding_mask[:, key_start : key_start + key.shape[-2]]
        attention_mask = build_attention_mask(
            position_ids=position_ids,
            key_start_position=key_start,
            key_length=key.shape[-2],
            dtype=query.dtype,
            sliding_window=max_cache_length,
            key_padding_mask=relevant_padding,
        )

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        attention_scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        softcap = self.config.attn_logit_softcapping
        if softcap is not None:
            attention_scores = torch.tanh(attention_scores / softcap) * softcap
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if self.training and self.config.attention_dropout:
            attention_probs = F.dropout(attention_probs, p=self.config.attention_dropout)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)
        return self.o_proj(output), present if use_cache else None


class Gemma3DecoderLayer(nn.Module):
    """One Gemma 3 block with the official four-norm residual order."""

    def __init__(
        self,
        config: Gemma3CoreConfig,
        layer_idx: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.self_attn = Gemma3SelfAttention(config, layer_idx, **factory)
        self.mlp = Gemma3MLP(config, **factory)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps, **factory)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps, **factory)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps, **factory)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps, **factory)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            key_padding_mask=key_padding_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        return hidden_states, present


@dataclass
class CoreCausalLMOutput:
    logits: torch.Tensor
    past_key_values: HybridKVCache | None


class Gemma3CoreModel(nn.Module):
    """Complete Gemma 3 text backbone built from the white-box blocks above."""

    def __init__(
        self,
        config: Gemma3CoreConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "embed_scale",
            torch.tensor(config.hidden_size**0.5, device=device),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, i, device=device, dtype=dtype) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)
        self.rotary_emb = Gemma3RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: HybridKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, HybridKVCache | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, query_length = input_ids.shape
        if attention_mask is None:
            total_length = (past_key_values.seen_tokens if past_key_values else 0) + query_length
            attention_mask = torch.ones(batch, total_length, device=input_ids.device, dtype=torch.long)

        if use_cache and past_key_values is None:
            past_key_values = HybridKVCache.empty(self.config.num_hidden_layers)
        past_seen_tokens = past_key_values.seen_tokens if past_key_values is not None else 0
        position_ids = torch.arange(
            past_seen_tokens,
            past_seen_tokens + query_length,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch, -1)

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.embed_scale.to(dtype=hidden_states.dtype)
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        for layer_idx, layer in enumerate(self.layers):
            layer_past = past_key_values.layers[layer_idx] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                position_embeddings=position_embeddings[self.config.layer_types[layer_idx]],
                position_ids=position_ids,
                key_padding_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if past_key_values is not None and use_cache:
                past_key_values.layers[layer_idx] = present
        if past_key_values is not None and use_cache:
            past_key_values.seen_tokens += query_length
        return self.norm(hidden_states), past_key_values if use_cache else None


class Gemma3CoreForCausalLM(nn.Module):
    """White-box Gemma 3 backbone plus tied language-model head."""

    def __init__(
        self,
        config: Gemma3CoreConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma3CoreModel(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: HybridKVCache | None = None,
        use_cache: bool = False,
    ) -> CoreCausalLMOutput:
        hidden_states, cache = self.model(
            input_ids,
            attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return CoreCausalLMOutput(logits=self.lm_head(hidden_states), past_key_values=cache)

    @torch.no_grad()
    def copy_from_hf(self, hf_model: Any) -> None:
        """Copy official Gemma 3 weights into this independent implementation."""
        hf_backbone = hf_model.model
        self.model.embed_tokens.weight.copy_(hf_backbone.embed_tokens.weight)
        for ours, official in zip(self.model.layers, hf_backbone.layers, strict=True):
            ours.load_state_dict(official.state_dict(), strict=True)
        self.model.norm.load_state_dict(hf_backbone.norm.state_dict(), strict=True)
        self.lm_head.weight = self.model.embed_tokens.weight


def alignment_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    """Return interpretable numerical-alignment metrics for two tensors."""
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
    ref = reference.detach().float().reshape(-1)
    cand = candidate.detach().float().reshape(-1)
    difference = (ref - cand).abs()
    cosine = F.cosine_similarity(ref, cand, dim=0).item()
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "cosine_similarity": cosine,
        "allclose_atol_1e-2_rtol_1e-2": bool(torch.allclose(ref, cand, atol=1e-2, rtol=1e-2)),
    }
