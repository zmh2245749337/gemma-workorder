"""Merge a verified QLoRA adapter into a standalone Hugging Face checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Gemma-WorkOrder LoRA adapter")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/workorder_qlora_adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/workorder_merged"))
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, token=True, torch_dtype=torch.float16, device_map="auto")
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
