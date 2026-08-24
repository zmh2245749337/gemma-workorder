from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SERIES = [
    ("fp16", True, "FP16 / KV cache on", "#2563eb", "-", "o"),
    ("fp16", False, "FP16 / KV cache off", "#60a5fa", "--", "o"),
    ("4bit", True, "4-bit / KV cache on", "#dc2626", "-", "s"),
    ("4bit", False, "4-bit / KV cache off", "#f87171", "--", "s"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Gemma benchmark and quality results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("reports/t4_20260811"),
        help="Directory containing benchmark.csv and quality JSONL files",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["use_kv_cache"] = data["use_kv_cache"].astype(str).str.lower().eq("true")
    for column in ("generated_tokens", "decode_tps_median", "peak_memory_mb"):
        data[column] = pd.to_numeric(data[column])
    return data


def plot_lines(
    benchmark: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.1))
    for precision, cache, label, color, linestyle, marker in SERIES:
        subset = benchmark[
            (benchmark["precision"] == precision)
            & (benchmark["use_kv_cache"] == cache)
        ].sort_values("generated_tokens")
        axis.plot(
            subset["generated_tokens"],
            subset[metric],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.2,
            markersize=6,
        )

    lengths = sorted(benchmark["generated_tokens"].unique())
    axis.set_xticks(lengths)
    axis.set_xlabel("Generated tokens")
    axis.set_ylabel(ylabel)
    axis.set_title(title, weight="bold")
    axis.legend(frameon=True, ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_quality(results_dir: Path) -> None:
    fp16 = pd.read_json(results_dir / "quality_fp16.jsonl", lines=True)
    bit4 = pd.read_json(results_dir / "quality_4bit.jsonl", lines=True)
    rates = [float(fp16["pass"].mean()) * 100, float(bit4["pass"].mean()) * 100]
    counts = [int(fp16["pass"].sum()), int(bit4["pass"].sum())]
    totals = [len(fp16), len(bit4)]

    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    bars = axis.bar(["FP16", "4-bit"], rates, color=["#2563eb", "#dc2626"], width=0.58)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Pass rate (%)")
    axis.set_title("Fixed 20-case Chinese task set", weight="bold")
    axis.grid(axis="y", alpha=0.25)
    for bar, rate, count, total in zip(bars, rates, counts, totals, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 3,
            f"{count}/{total} ({rate:.0f}%)",
            ha="center",
            va="bottom",
            weight="bold",
        )
    figure.tight_layout()
    figure.savefig(results_dir / "quality_pass_rate.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    benchmark = load_benchmark(results_dir / "benchmark.csv")
    plot_lines(
        benchmark,
        metric="decode_tps_median",
        ylabel="Decode throughput (tokens/s)",
        title="Decode throughput on Tesla T4",
        output_path=results_dir / "decode_throughput.png",
    )
    plot_lines(
        benchmark,
        metric="peak_memory_mb",
        ylabel="Peak allocated GPU memory (MB)",
        title="Peak GPU memory on Tesla T4",
        output_path=results_dir / "peak_memory.png",
    )
    plot_quality(results_dir)
    print(f"Charts written to {results_dir}")


if __name__ == "__main__":
    main()
