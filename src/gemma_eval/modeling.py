from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_ID = "google/gemma-3-1b-it"


@dataclass(frozen=True)
class ModelBundle:
    model: object
    tokenizer: object
    model_id: str
    precision: str
    compute_dtype: torch.dtype


def resolve_compute_dtype(precision: str) -> torch.dtype:
    """Choose a CUDA-safe compute dtype for the requested precision."""
    precision = precision.lower()
    if precision == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前设备不支持 BF16，请改用 --precision fp16。")
        return torch.bfloat16
    if precision in {"fp16", "8bit", "4bit"}:
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"不支持的精度: {precision}")


def load_model_bundle(
    model_id: str = DEFAULT_MODEL_ID,
    precision: str = "fp16",
) -> ModelBundle:
    """Load one model variant at a time so benchmark memory remains comparable."""
    precision = precision.lower()
    compute_dtype = resolve_compute_dtype(precision)
    token = os.getenv("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    load_kwargs: dict[str, object] = {
        "device_map": "auto",
        "token": token,
    }

    if precision == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        precision=precision,
        compute_dtype=compute_dtype,
    )


def model_device(model: object) -> torch.device:
    """Return the input device for a model loaded with or without device_map."""
    return next(model.parameters()).device


def build_chat_inputs(tokenizer: object, model: object, prompt: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = model_device(model)
    return {key: value.to(device) for key, value in encoded.items()}
