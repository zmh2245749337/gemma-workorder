"""Gemma inference adapter for the work-order contract."""

from __future__ import annotations

import torch

from .modeling import build_chat_inputs, load_model_bundle
from .workorder import extract_json, route_tool, validate_fields
from .workorder_data import workorder_prompt


class GemmaWorkOrderEngine:
    """Apply the same schema and tool boundary to every model response."""

    def __init__(self, model_id: str, precision: str = "4bit", max_new_tokens: int = 256):
        self.bundle = load_model_bundle(model_id, precision)
        self.max_new_tokens = max_new_tokens

    def parse(self, text: str) -> dict:
        inputs = build_chat_inputs(self.bundle.tokenizer, self.bundle.model, workorder_prompt(text))
        with torch.inference_mode():
            output = self.bundle.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
        start = inputs["input_ids"].shape[-1]
        generated = self.bundle.tokenizer.decode(output[0, start:], skip_special_tokens=True).strip()
        fields = validate_fields(extract_json(generated))
        return {"raw_model_output": generated, "structured_fields": fields.to_dict(), "tool_call": route_tool(fields).to_dict()}

