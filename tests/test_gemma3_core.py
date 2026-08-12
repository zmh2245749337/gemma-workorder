import math

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from gemma_eval.gemma3_core import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    Gemma3CoreConfig,
    Gemma3CoreForCausalLM,
    Gemma3RMSNorm,
    build_attention_mask,
    make_layer_types,
    repeat_kv,
)


def tiny_config() -> Gemma3CoreConfig:
    return Gemma3CoreConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        query_pre_attn_scalar=4,
        sliding_window=4,
        layer_types=(SLIDING_ATTENTION, FULL_ATTENTION),
        pad_token_id=0,
    )


def test_gemma3_1b_defaults_are_documented_architecture() -> None:
    config = Gemma3CoreConfig()
    assert config.hidden_size == 1152
    assert config.intermediate_size == 6912
    assert config.num_hidden_layers == 26
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 1
    assert config.head_dim == 256
    assert config.sliding_window == 512


def test_layer_pattern_is_five_local_then_one_global() -> None:
    assert make_layer_types(12) == (
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        FULL_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        FULL_ATTENTION,
    )


def test_rms_norm_uses_one_plus_weight() -> None:
    norm = Gemma3RMSNorm(4, eps=1e-6)
    norm.weight.data.copy_(torch.tensor([0.0, 0.5, -0.25, 1.0]))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = expected * (1.0 + norm.weight)
    assert torch.allclose(norm(x), expected)


def test_repeat_kv_expands_one_head_without_changing_values() -> None:
    kv = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    repeated = repeat_kv(kv, repetitions=4)
    assert repeated.shape == (1, 4, 3, 4)
    for head in range(4):
        assert torch.equal(repeated[:, head], kv[:, 0])


def test_sliding_causal_mask_has_expected_window() -> None:
    position_ids = torch.arange(6).unsqueeze(0)
    mask = build_attention_mask(
        position_ids=position_ids,
        key_start_position=0,
        key_length=6,
        dtype=torch.float32,
        sliding_window=3,
    )[0, 0]
    assert torch.isfinite(mask[5, 3:6]).all()
    assert mask[5, :3].lt(-1e20).all()
    assert mask[2, 3:].lt(-1e20).all()


def test_mqa_projection_shapes_and_tied_embeddings() -> None:
    model = Gemma3CoreForCausalLM(tiny_config())
    attention = model.model.layers[0].self_attn
    assert attention.q_proj.out_features == 4 * 4
    assert attention.k_proj.out_features == 1 * 4
    assert attention.v_proj.out_features == 1 * 4
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(7)
    model = Gemma3CoreForCausalLM(tiny_config()).eval()
    input_ids = torch.tensor([[2, 5, 9, 3, 7]])
    full_mask = torch.ones_like(input_ids)

    full_logits = model(input_ids, full_mask, use_cache=False).logits[:, -1]
    prefill = model(input_ids[:, :-1], full_mask[:, :-1], use_cache=True)
    decoded = model(
        input_ids[:, -1:],
        full_mask,
        past_key_values=prefill.past_key_values,
        use_cache=True,
    )
    assert torch.allclose(full_logits, decoded.logits[:, -1], atol=1e-5, rtol=1e-4)
    assert decoded.past_key_values is not None
    assert decoded.past_key_values.seen_tokens == input_ids.shape[1]
    assert decoded.past_key_values.layers[0].length == 4
    assert decoded.past_key_values.layers[1].length == 5


def test_attention_scaling_uses_query_pre_attention_scalar() -> None:
    config = tiny_config()
    model = Gemma3CoreForCausalLM(config)
    assert model.model.layers[0].self_attn.scaling == 1 / math.sqrt(config.query_pre_attn_scalar)


def test_tiny_model_numerically_matches_transformers_reference() -> None:
    """Catch details that shape-only tests miss by aligning with official code."""
    torch.manual_seed(11)
    layer_types = [SLIDING_ATTENTION, FULL_ATTENTION]
    hf_config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        query_pre_attn_scalar=4,
        sliding_window=4,
        layer_types=layer_types,
        rope_parameters={
            SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 10_000.0},
            FULL_ATTENTION: {"rope_type": "default", "rope_theta": 1_000_000.0},
        },
        pad_token_id=0,
    )
    official = Gemma3ForCausalLM(hf_config).eval()
    official.config._attn_implementation = "eager"
    custom_config = Gemma3CoreConfig.from_hf_config(hf_config)
    custom = Gemma3CoreForCausalLM(custom_config).eval()
    custom.copy_from_hf(official)

    input_ids = torch.tensor([[2, 5, 9, 3, 7]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        reference_logits = official(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits
        custom_logits = custom(input_ids, attention_mask, use_cache=False).logits
    assert torch.allclose(reference_logits, custom_logits, atol=1e-5, rtol=1e-4)
