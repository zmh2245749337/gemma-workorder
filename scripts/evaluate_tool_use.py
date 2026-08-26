"""Evaluate Base Gemma or a QLoRA adapter on the same fixed tool-use contract."""

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

from gemma_eval.tool_use import extract_json, strict_json_loads, validate_decision
from gemma_eval.tool_use_data import (
    aggregate_metrics,
    compare_decisions,
    load_jsonl,
    state_item_count,
    tool_use_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-turn state and tool decisions")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--precision", choices=["fp16", "4bit"], default="4bit")
    parser.add_argument("--dataset", type=Path, default=Path("data/tool_use/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/tool_use_model_eval.json"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    args = parser.parse_args()

    import torch
    from gemma_eval.modeling import build_chat_inputs, load_model_bundle

    rows = load_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    bundle = load_model_bundle(args.model_id, args.precision)
    model = bundle.model
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    records: list[dict] = []
    try:
        from tqdm import tqdm

        iterator = tqdm(rows, desc="evaluating", unit="sample")
    except ImportError:
        iterator = rows
    for row in iterator:
        target = row["output"]
        prompt = tool_use_prompt(row["history"], row["current_user"], row["previous_state"])
        inputs = build_chat_inputs(bundle.tokenizer, model, prompt)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
            )
        prompt_length = inputs["input_ids"].shape[-1]
        raw = bundle.tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        ).strip()
        item = {
            "sample_id": row.get("sample_id"),
            "source": row.get("source"),
            "target_decision": target["decision"],
            "raw_model_output": raw,
            "strict_json_valid": False,
            "schema_valid": False,
            "state_target_items": state_item_count(target["belief_state"]),
        }
        try:
            strict_json_loads(raw)
            item["strict_json_valid"] = True
        except (ValueError, json.JSONDecodeError):
            pass
        try:
            prediction = validate_decision(extract_json(raw))
            item.update({"schema_valid": True, "prediction": prediction.to_dict()})
            item.update(compare_decisions(prediction, target))
        except (ValueError, json.JSONDecodeError) as error:
            item["error"] = str(error)
        records.append(item)
    report = {
        "scope": "CrossWOZ natural examples plus declared controlled missing-slot safety cases",
        "model_id": args.model_id,
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": args.precision,
        "dataset": str(args.dataset),
        "metrics": aggregate_metrics(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
