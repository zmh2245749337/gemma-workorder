"""Generate a category-level Agent reliability benchmark report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(ROOT / "src")
if SOURCE_ROOT in sys.path:
    sys.path.remove(SOURCE_ROOT)
sys.path.insert(0, SOURCE_ROOT)

from gemma_eval.benchmark import load_enriched_records, slice_metrics


LABELS = {
    "all": "全部样本",
    "call_tool": "应调用工具",
    "ask_user": "缺参追问",
    "no_tool": "无需工具",
    "multi_turn_state": "多轮状态",
    "long_state": "长状态（≥11项）",
    "missing_argument_controlled": "受控缺参",
    "side_effect_tool": "副作用工具",
    "coreference": "指代消解",
    "correction": "条件修正",
    "cross_domain": "跨领域",
    "request_taxi": "打车调用",
}


def pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def scoped_pct(row: dict[str, Any], key: str, count_key: str) -> str:
    return pct(row.get(key)) if row.get(count_key, 0) else "—"


def table(title: str, metrics: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| 场景 | 样本 | Schema | 状态 F1 | 决策 | 工具 | 参数 F1 | 误调用率↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = [
        "all", "call_tool", "ask_user", "no_tool", "multi_turn_state", "long_state",
        "missing_argument_controlled", "side_effect_tool", "coreference", "correction",
        "cross_domain", "request_taxi",
    ]
    for tag in order:
        if tag not in metrics:
            continue
        row = metrics[tag]
        lines.append(
            f"| {LABELS.get(tag, tag)} | {row.get('samples', 0)} | "
            f"{pct(row.get('schema_valid_rate'))} | {pct(row.get('state_slot_f1'))} | "
            f"{pct(row.get('decision_accuracy'))} | "
            f"{scoped_pct(row, 'tool_accuracy', 'tool_target_samples')} | "
            f"{scoped_pct(row, 'argument_slot_f1', 'tool_target_samples')} | "
            f"{scoped_pct(row, 'no_tool_false_positive_rate', 'non_tool_target_samples')} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent tool-use scenario benchmark")
    parser.add_argument("--test-report", type=Path, default=Path("reports/tool_use_qlora_4bit.json"))
    parser.add_argument("--test-data", type=Path, default=Path("data/tool_use/test.jsonl"))
    parser.add_argument("--challenge-report", type=Path, default=Path("reports/tool_use_qlora_challenge_4bit.json"))
    parser.add_argument("--challenge-data", type=Path, default=Path("data/tool_use/challenge.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/BENCHMARK_REPORT.md"))
    args = parser.parse_args()

    test = slice_metrics(load_enriched_records(args.test_report, args.test_data))
    challenge = slice_metrics(load_enriched_records(args.challenge_report, args.challenge_data))
    lines = [
        "# Agent Tool-Use 分场景基准",
        "",
        "> 该报告由固定逐样本预测自动切片生成。同一条样本可以属于多个场景，因此各行样本数不可相加。",
        "",
        *table("Test 场景切片", test),
        "",
        *table("Challenge 场景切片", challenge),
        "",
        "## 如何解读",
        "",
        "- `ask_user` 与 `no_tool` 对应 When-to-Call：模型不仅要会调用，还要知道何时追问或停止。",
        "- `multi_turn_state`、`long_state` 检查历史状态累积；JGA 的严格分析仍见实验报告。",
        "- `side_effect_tool` 只衡量模型决策；运行时仍会把副作用动作拦截为人工确认。",
        "- challenge 是可解释难例切片，不等同于 BFCL 官方分数，也不宣称生产流量表现。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已保存：{args.output}")


if __name__ == "__main__":
    main()
