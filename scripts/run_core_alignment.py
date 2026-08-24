from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# Allow direct execution (Colab or local) without requiring ``pip install -e .``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma_eval.gemma3_core import Gemma3CoreConfig, Gemma3CoreForCausalLM, alignment_metrics
from gemma_eval.modeling import DEFAULT_MODEL_ID, build_chat_inputs, load_model_bundle, model_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load official Gemma 3 weights into the white-box implementation and compare outputs."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--prompt", default="请用两句话解释 KV Cache 的作用。")
    parser.add_argument("--layers", type=int, nargs="+", default=(0, 5, 25))
    parser.add_argument("--output", type=Path, default=Path("reports/core_alignment.json"))
    return parser.parse_args()


def _as_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"cannot extract tensor from hook output type {type(output)!r}")


def _capture_layers(model: Any, indices: list[int]) -> tuple[dict[int, torch.Tensor], list[Any]]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in indices:
        layer = model.model.layers[layer_idx]

        def hook(_module: Any, _inputs: Any, output: Any, *, idx: int = layer_idx) -> None:
            captured[idx] = _as_tensor(output).detach()

        handles.append(layer.register_forward_hook(hook))
    return captured, handles


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args.model_id, args.precision)
    official = bundle.model
    official.config._attn_implementation = "eager"
    device = model_device(official)
    dtype = next(official.parameters()).dtype

    config = Gemma3CoreConfig.from_hf_config(official.config)
    invalid_layers = [idx for idx in args.layers if idx < 0 or idx >= config.num_hidden_layers]
    if invalid_layers:
        raise ValueError(f"layer indices out of range: {invalid_layers}")

    print("正在构建独立 PyTorch 模型并复制官方权重……")
    white_box = Gemma3CoreForCausalLM(config, device=device, dtype=dtype).eval()
    white_box.copy_from_hf(official)

    inputs = build_chat_inputs(bundle.tokenizer, official, args.prompt)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    official_layers, official_handles = _capture_layers(official, list(args.layers))
    custom_layers, custom_handles = _capture_layers(white_box, list(args.layers))
    with torch.inference_mode():
        official_output = official(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        custom_output = white_box(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    for handle in official_handles + custom_handles:
        handle.remove()

    last_official = official_output.logits[:, -1, :]
    last_custom = custom_output.logits[:, -1, :]
    layer_results = {
        str(idx): {
            "layer_type": config.layer_types[idx],
            **alignment_metrics(official_layers[idx], custom_layers[idx]),
        }
        for idx in args.layers
    }
    logit_results = alignment_metrics(last_official, last_custom)
    official_top1 = int(last_official.argmax(dim=-1).item())
    custom_top1 = int(last_custom.argmax(dim=-1).item())
    logit_results.update(
        {
            "official_top1_token_id": official_top1,
            "custom_top1_token_id": custom_top1,
            "top1_match": official_top1 == custom_top1,
        }
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "prompt": args.prompt,
        "precision": args.precision,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "architecture": {
            "layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "query_heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "sliding_window": config.sliding_window,
        },
        "layer_alignment": layer_results,
        "last_token_logits": logit_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDecoder 层数值对齐：")
    for layer_idx, metrics in layer_results.items():
        print(
            f"layer={layer_idx:>2} type={metrics['layer_type']:<17} "
            f"max_abs={metrics['max_abs_error']:.6g} "
            f"mean_abs={metrics['mean_abs_error']:.6g} "
            f"cos={metrics['cosine_similarity']:.9f}"
        )
    print("\n最后一个 token 的 logits：")
    print(json.dumps(logit_results, ensure_ascii=False, indent=2))
    print(f"\n报告已保存：{args.output}")


if __name__ == "__main__":
    main()
