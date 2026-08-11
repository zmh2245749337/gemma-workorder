from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma_eval.modeling import DEFAULT_MODEL_ID, build_chat_inputs, load_model_bundle
from gemma_eval.records import append_jsonl, environment_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 固定题集质量检查")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16", "8bit", "4bit"])
    parser.add_argument("--dataset", default="configs/prompts_zh.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", default="reports/quality.jsonl")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, object]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} 不是合法 JSONL") from error
    return cases


def contains_all(text: str, expected: list[str]) -> bool:
    normalized = text.casefold()
    return all(item.casefold() in normalized for item in expected)


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args.model_id, args.precision)
    cases = load_cases(Path(args.dataset))
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"{output_path} 已存在。请换一个输出名，避免覆盖证据。")

    passed = 0
    env = environment_record()
    for case in cases:
        prompt = str(case["prompt"])
        inputs = build_chat_inputs(bundle.tokenizer, bundle.model, prompt)
        with torch.inference_mode():
            generated = bundle.model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        new_ids = generated[:, prompt_length:]
        answer = bundle.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        expected = [str(item) for item in case.get("must_contain", [])]
        is_pass = contains_all(answer, expected)
        passed += int(is_pass)

        append_jsonl(
            output_path,
            {
                **env,
                "model_id": args.model_id,
                "precision": args.precision,
                "case_id": case["id"],
                "category": case["category"],
                "prompt": prompt,
                "must_contain": expected,
                "answer": answer,
                "pass": is_pass,
                "generated_tokens": int(new_ids.shape[-1]),
            },
        )
        print(f"[{case['id']}] {'PASS' if is_pass else 'FAIL'}")

    print(f"\n通过率: {passed}/{len(cases)} = {passed / len(cases):.1%}")
    print("注意：该通过率只代表当前固定题集，不代表通用模型能力。")


if __name__ == "__main__":
    main()
