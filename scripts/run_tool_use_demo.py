"""Run one end-to-end local tool-decision example with Base or QLoRA Gemma."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(ROOT / "src")
if SOURCE_ROOT in sys.path:
    sys.path.remove(SOURCE_ROOT)
sys.path.insert(0, SOURCE_ROOT)

from gemma_eval.tool_use import extract_json, initialise_demo_database, run_decision
from gemma_eval.tool_use_data import load_jsonl, tool_use_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Local context-aware Gemma tool-use demo")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--precision", choices=["fp16", "4bit"], default="4bit")
    parser.add_argument("--dataset", type=Path, default=Path("data/tool_use/challenge.jsonl"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--db", type=Path, default=Path("artifacts/demo_tools.sqlite3"))
    args = parser.parse_args()

    import torch
    from gemma_eval.modeling import build_chat_inputs, load_model_bundle

    rows = load_jsonl(args.dataset)
    if not 0 <= args.index < len(rows):
        raise IndexError(f"--index must be between 0 and {len(rows) - 1}")
    row = rows[args.index]
    prompt = tool_use_prompt(row["history"], row["current_user"], row["previous_state"])
    bundle = load_model_bundle(args.model_id, args.precision)
    model = bundle.model
    adapter = args.adapter
    if adapter and not adapter.exists():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter}")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    inputs = build_chat_inputs(bundle.tokenizer, model, prompt)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
        )
    prompt_length = inputs["input_ids"].shape[-1]
    raw = bundle.tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
    parsed = extract_json(raw)
    initialise_demo_database(args.db, Path("data/tool_use/reference"))
    outcome = run_decision(parsed, args.db)
    print(
        json.dumps(
            {
                "input": {
                    "history": row["history"],
                    "previous_state": row["previous_state"],
                    "current_user": row["current_user"],
                },
                "model": {"base": args.model_id, "adapter": str(adapter) if adapter else None},
                "raw_model_output": raw,
                "outcome": outcome,
                "reference_target": row["output"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
