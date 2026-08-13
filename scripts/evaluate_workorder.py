"""Evaluate Base Gemma or a QLoRA adapter against the controlled test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma_eval.workorder import extract_json, route_tool, validate_fields
from gemma_eval.workorder_data import aggregate_metrics, compare_fields, load_jsonl, workorder_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate constrained JSON extraction and safe tool routing")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--adapter", type=Path, default=None, help="optional QLoRA adapter directory")
    parser.add_argument("--precision", choices=["fp16", "4bit"], default="4bit")
    parser.add_argument("--dataset", type=Path, default=Path("data/workorder/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/workorder_model_eval.json"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    from gemma_eval.modeling import build_chat_inputs, load_model_bundle
    import torch

    bundle = load_model_bundle(args.model_id, args.precision)
    model = bundle.model
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    records = []
    for row in load_jsonl(args.dataset):
        inputs = build_chat_inputs(bundle.tokenizer, model, workorder_prompt(row["input_text"]))
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        prompt_length = inputs["input_ids"].shape[-1]
        raw = bundle.tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
        item = {"sample_id": row.get("sample_id"), "raw_model_output": raw, "json_valid": False, "tool_correct": False}
        try:
            parsed = validate_fields(extract_json(raw))
            prediction = parsed.to_dict()
            comparison = compare_fields(prediction, row["output"])
            item.update({"json_valid": True, "prediction": prediction, **comparison})
            item["tool_call"] = route_tool(parsed).to_dict()
            item["tool_correct"] = item["tool_call"]["name"] == row["output"]["next_action"]
        except (ValueError, json.JSONDecodeError) as error:
            item["error"] = str(error)
        records.append(item)
    report = {
        "scope": "Synthetic, self-constructed controlled evaluation; not a general industrial diagnosis benchmark.",
        "model_id": args.model_id,
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": args.precision,
        "metrics": aggregate_metrics(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
