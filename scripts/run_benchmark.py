from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma_eval.decoding import manual_generate
from gemma_eval.modeling import DEFAULT_MODEL_ID, build_chat_inputs, load_model_bundle
from gemma_eval.records import environment_record


DEFAULT_PROMPT = "请用三点说明大语言模型量化的优点和局限。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma KV Cache 与量化基准")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16", "8bit", "4bit"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/benchmark.csv")
    return parser.parse_args()


def reset_peak_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)


def model_footprint_mb(model: object) -> float | None:
    getter = getattr(model, "get_memory_footprint", None)
    if getter is None:
        return None
    return float(getter()) / (1024**2)


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_mode(bundle: object, inputs: dict[str, torch.Tensor], args: argparse.Namespace, use_cache: bool) -> dict[str, object]:
    for _ in range(args.warmups):
        manual_generate(
            bundle.model,
            inputs["input_ids"],
            inputs.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            seed=args.seed,
            use_cache=use_cache,
            stop_on_eos=False,
        )

    reset_peak_memory()
    results = []
    for _ in range(args.repeats):
        results.append(
            manual_generate(
                bundle.model,
                inputs["input_ids"],
                inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                seed=args.seed,
                use_cache=use_cache,
                stop_on_eos=False,
            )
        )

    env = environment_record()
    return {
        **env,
        "model_id": args.model_id,
        "precision": args.precision,
        "use_kv_cache": use_cache,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": args.max_new_tokens,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "ttft_ms_median": statistics.median(item.ttft_s for item in results) * 1000,
        "decode_tps_median": statistics.median(item.decode_tps for item in results),
        "total_tps_median": statistics.median(item.total_tps for item in results),
        "total_time_s_median": statistics.median(item.total_s for item in results),
        "peak_memory_mb": peak_memory_mb(),
        "model_footprint_mb": model_footprint_mb(bundle.model),
        "prompt": args.prompt,
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("repeats 至少为 1，warmups 不能小于 0。")

    bundle = load_model_bundle(args.model_id, args.precision)
    inputs = build_chat_inputs(bundle.tokenizer, bundle.model, args.prompt)
    output_path = Path(args.output)

    for use_cache in (True, False):
        row = run_mode(bundle, inputs, args, use_cache)
        append_csv(output_path, row)
        print(
            f"precision={args.precision} cache={use_cache} | "
            f"TTFT={row['ttft_ms_median']:.2f} ms | "
            f"decode={row['decode_tps_median']:.2f} tok/s | "
            f"peak={row['peak_memory_mb']} MB"
        )


if __name__ == "__main__":
    main()
