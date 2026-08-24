from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class GenerationResult:
    sequence_ids: torch.Tensor
    new_token_ids: torch.Tensor
    ttft_s: float
    decode_s: float
    total_s: float

    @property
    def generated_tokens(self) -> int:
        return int(self.new_token_ids.shape[-1])

    @property
    def total_tps(self) -> float:
        return self.generated_tokens / self.total_s if self.total_s > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        decode_tokens = max(self.generated_tokens - 1, 0)
        return decode_tokens / self.decode_s if self.decode_s > 0 else 0.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def filter_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Apply temperature, Top-K and nucleus filtering to a logits batch."""
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0。")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k 必须大于 0，或设为 None。")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p 必须位于 (0, 1]。")

    filtered = logits / temperature

    if top_k is not None and top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits,
        )

    return filtered


def sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    filtered = filter_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def _normalize_eos_ids(eos_token_id: int | Iterable[int] | None) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(token_id) for token_id in eos_token_id}


@torch.inference_mode()
def manual_generate(
    model: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    max_new_tokens: int = 64,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.9,
    seed: int = 42,
    use_cache: bool = True,
    stop_on_eos: bool = True,
    eos_token_id: int | Iterable[int] | None = None,
) -> GenerationResult:
    """Generate tokens without calling ``model.generate``.

    The first forward pass consumes the full prompt. With KV Cache enabled,
    later passes consume only the newest token while reusing past key/value
    tensors. With cache disabled, every step recomputes the full sequence.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0。")
    if input_ids.shape[0] != 1:
        raise ValueError("当前实验实现只支持 batch_size=1，便于解释和计时。")

    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    eos_ids = _normalize_eos_ids(
        eos_token_id
        if eos_token_id is not None
        else getattr(model.generation_config, "eos_token_id", None)
    )
    generated = input_ids.clone()
    full_attention_mask = attention_mask.clone()
    new_tokens: list[torch.Tensor] = []
    past_key_values = None

    _sync(device)
    total_start = time.perf_counter()
    first_start = total_start

    for step in range(max_new_tokens):
        if use_cache and past_key_values is not None:
            step_input_ids = generated[:, -1:]
        else:
            step_input_ids = generated

        outputs = model(
            input_ids=step_input_ids,
            attention_mask=full_attention_mask,
            past_key_values=past_key_values if use_cache else None,
            use_cache=use_cache,
            return_dict=True,
        )
        next_token = sample_next_token(
            outputs.logits[:, -1, :],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        if use_cache:
            past_key_values = outputs.past_key_values

        generated = torch.cat([generated, next_token], dim=-1)
        full_attention_mask = torch.cat(
            [full_attention_mask, torch.ones_like(next_token)],
            dim=-1,
        )
        new_tokens.append(next_token)

        if step == 0:
            _sync(device)
            first_end = time.perf_counter()

        if stop_on_eos and int(next_token.item()) in eos_ids:
            break

    _sync(device)
    total_end = time.perf_counter()
    new_token_ids = torch.cat(new_tokens, dim=-1)
    ttft_s = first_end - first_start
    return GenerationResult(
        sequence_ids=generated,
        new_token_ids=new_token_ids,
        ttft_s=ttft_s,
        decode_s=max(total_end - first_end, 0.0),
        total_s=total_end - total_start,
    )
