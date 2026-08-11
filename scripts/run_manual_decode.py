from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma_eval.decoding import manual_generate
from gemma_eval.modeling import DEFAULT_MODEL_ID, build_chat_inputs, load_model_bundle
from gemma_eval.records import environment_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手写 Gemma 自回归解码实验")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16", "8bit", "4bit"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--check-hf-parity", action="store_true")
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args.model_id, args.precision)
    inputs = build_chat_inputs(bundle.tokenizer, bundle.model, args.prompt)

    result = manual_generate(
        bundle.model,
        inputs["input_ids"],
        inputs.get("attention_mask"),
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        use_cache=not args.no_cache,
    )
    manual_text = bundle.tokenizer.decode(result.new_token_ids[0], skip_special_tokens=True)
    print("\n[手写解码输出]\n" + manual_text)
    print(
        f"\n生成 {result.generated_tokens} tokens | "
        f"TTFT {result.ttft_s * 1000:.2f} ms | "
        f"总吞吐 {result.total_tps:.2f} tok/s"
    )

    parity = None
    if args.check_hf_parity:
        if not args.greedy:
            raise ValueError("--check-hf-parity 只用于 --greedy，避免采样器实现差异干扰。")
        with torch.inference_mode():
            hf_ids = bundle.model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=not args.no_cache,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        hf_new_ids = hf_ids[:, prompt_length:]
        parity = torch.equal(result.new_token_ids.cpu(), hf_new_ids.cpu())
        print(f"与 model.generate 贪心输出逐 token 一致: {parity}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **environment_record(),
            "model_id": args.model_id,
            "precision": args.precision,
            "prompt": args.prompt,
            "parameters": vars(args),
            "manual_text": manual_text,
            "generated_tokens": result.generated_tokens,
            "ttft_ms": result.ttft_s * 1000,
            "total_tps": result.total_tps,
            "hf_greedy_parity": parity,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
