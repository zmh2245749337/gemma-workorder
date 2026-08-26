"""Recompute comparable metrics and write a compact experiment report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(ROOT / "src")
if SOURCE_ROOT in sys.path:
    sys.path.remove(SOURCE_ROOT)
sys.path.insert(0, SOURCE_ROOT)

from gemma_eval.tool_use import validate_decision
from gemma_eval.tool_use_data import (
    aggregate_metrics,
    compare_decisions,
    load_jsonl,
    state_item_count,
)


METRICS = (
    ("严格 JSON 合法率", "strict_json_valid_rate"),
    ("Schema 合法率", "schema_valid_rate"),
    ("状态联合准确率（JGA）", "state_joint_goal_accuracy"),
    ("状态 Slot F1", "state_slot_f1"),
    ("决策准确率", "decision_accuracy"),
    ("工具选择准确率", "tool_accuracy"),
    ("参数完全匹配率", "arguments_exact_rate"),
    ("参数 Slot F1", "argument_slot_f1"),
    ("必填参数准确率", "required_argument_accuracy"),
    ("非工具轮误调用率", "no_tool_false_positive_rate"),
    ("工具调用决策准确率", "call_tool_decision_accuracy"),
    ("追问决策准确率", "ask_user_decision_accuracy"),
    ("无工具决策准确率", "no_tool_decision_accuracy"),
)

CLOSURE_PATTERN = re.compile(r"谢谢|感谢|好的|好吧|行[！。,. ]|知道了|再见|哈哈|辛苦")


def enrich_report(report_path: Path, dataset_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = {str(row["sample_id"]): row for row in load_jsonl(dataset_path)}
    enriched: list[dict[str, Any]] = []
    for original in report["records"]:
        record = dict(original)
        row = dataset.get(str(record.get("sample_id")))
        if row is None:
            raise ValueError(f"{report_path}: sample {record.get('sample_id')} is absent from {dataset_path}")
        target = row["output"]
        record["target_decision"] = target["decision"]
        record["state_target_items"] = state_item_count(target["belief_state"])
        record["current_user"] = row["current_user"]
        record["target"] = target
        if record.get("schema_valid") and record.get("prediction"):
            prediction = validate_decision(record["prediction"])
            record.update(compare_decisions(prediction, target))
        enriched.append(record)
    return aggregate_metrics(enriched), enriched


def percent(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def metric_table(
    base: dict[str, Any], test: dict[str, Any], challenge: dict[str, Any]
) -> list[str]:
    rows = [
        "| 指标 | Base（test） | QLoRA（test） | QLoRA（challenge） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in METRICS:
        rows.append(
            f"| {label} | {percent(base.get(key))} | {percent(test.get(key))} | {percent(challenge.get(key))} |"
        )
    return rows


def count_errors(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "schema_invalid": sum(not row.get("schema_valid", False) for row in records),
        "state_mismatch": sum(
            row.get("schema_valid", False) and not row.get("state_exact", False) for row in records
        ),
        "decision_mismatch": sum(
            row.get("schema_valid", False) and not row.get("decision_correct", False)
            for row in records
        ),
        "tool_mismatch": sum(
            row.get("target_decision") == "call_tool" and not row.get("tool_correct", False)
            for row in records
        ),
        "argument_mismatch": sum(
            row.get("target_decision") == "call_tool" and not row.get("arguments_exact", False)
            for row in records
        ),
        "false_tool_call": sum(row.get("false_tool_call", False) for row in records),
        "closure_false_tool_call": sum(
            row.get("false_tool_call", False)
            and bool(CLOSURE_PATTERN.search(row.get("current_user", "")))
            for row in records
        ),
    }


def decision_confusion(records: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    confusion: Counter[tuple[str, str]] = Counter()
    for row in records:
        if not row.get("schema_valid") or not row.get("prediction"):
            continue
        confusion[(row["target_decision"], row["prediction"]["decision"])] += 1
    return confusion


def jga_buckets(records: list[dict[str, Any]]) -> list[tuple[str, int, float]]:
    buckets = (("1–5", 1, 5), ("6–10", 6, 10), ("11–20", 11, 20), ("21+", 21, 10**9))
    output = []
    for label, lower, upper in buckets:
        selected = [
            row
            for row in records
            if row.get("schema_valid") and lower <= row.get("state_target_items", 0) <= upper
        ]
        exact = sum(row.get("state_exact", False) for row in selected)
        output.append((label, len(selected), exact / len(selected) if selected else 0.0))
    return output


def sample_rows(
    records: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], limit: int = 3
) -> list[str]:
    lines = []
    for row in (item for item in records if predicate(item)):
        prediction = row.get("prediction") or {}
        tool_call = prediction.get("tool_call") or {}
        user = row.get("current_user", "").replace("\n", " ")
        if len(user) > 80:
            user = user[:77] + "…"
        lines.append(
            f"- `{row.get('sample_id')}`：{user}；预测 `{prediction.get('decision', 'invalid')}`"
            f" / `{tool_call.get('name', 'none')}`。"
        )
        if len(lines) == limit:
            break
    return lines or ["- 无。"]


def build_report(
    base_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    challenge_metrics: dict[str, Any],
    test_records: list[dict[str, Any]],
    challenge_records: list[dict[str, Any]],
) -> str:
    test_errors = count_errors(test_records)
    challenge_errors = count_errors(challenge_records)
    confusion = decision_confusion(test_records)
    test_required = sum(row.get("required_argument_case", False) for row in test_records)
    challenge_required = sum(row.get("required_argument_case", False) for row in challenge_records)
    lines = [
        "# Gemma Tool-Use 实验报告",
        "",
        "> 本报告由固定评测结果与对应数据集自动生成；Base 与 QLoRA 使用相同 Prompt、4bit 精度和贪心解码。",
        "",
        "## 核心结果",
        "",
        *metric_table(base_metrics, test_metrics, challenge_metrics),
        "",
        "test 与 challenge 在状态 Slot F1 和决策准确率上接近，说明主要能力提升能够迁移到难例切片。参数完全匹配率仍明显低于参数 Slot F1，应同时保留严格指标与分项指标，避免用单一数字掩盖错误。",
        f"必填参数准确率只覆盖路线/打车工具，test 为 {test_required} 条、challenge 为 {challenge_required} 条，样本量较小，不作为主要结论。",
        "",
        "## 测试集错误分布",
        "",
        "| 类型 | test（440） | challenge（80） |",
        "| --- | ---: | ---: |",
        f"| Schema 不合法 | {test_errors['schema_invalid']} | {challenge_errors['schema_invalid']} |",
        f"| 状态未完全匹配 | {test_errors['state_mismatch']} | {challenge_errors['state_mismatch']} |",
        f"| 决策错误 | {test_errors['decision_mismatch']} | {challenge_errors['decision_mismatch']} |",
        f"| 工具错误 | {test_errors['tool_mismatch']} | {challenge_errors['tool_mismatch']} |",
        f"| 参数未完全匹配 | {test_errors['argument_mismatch']} | {challenge_errors['argument_mismatch']} |",
        f"| 非工具轮误调用 | {test_errors['false_tool_call']} | {challenge_errors['false_tool_call']} |",
        f"| 其中结束语/致谢类 | {test_errors['closure_false_tool_call']} | {challenge_errors['closure_false_tool_call']} |",
        "",
        "## 决策混淆矩阵（test）",
        "",
        "| 目标\\预测 | call_tool | ask_user | no_tool |",
        "| --- | ---: | ---: | ---: |",
    ]
    for target in ("call_tool", "ask_user", "no_tool"):
        values = [str(confusion[(target, predicted)]) for predicted in ("call_tool", "ask_user", "no_tool")]
        lines.append(f"| {target} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "## 状态长度与联合准确率（test）",
            "",
            "JGA 要求一个样本的所有状态项同时正确，因此会随累计状态长度快速下降。",
            "",
            "| 目标状态项数 | Schema 合法样本 | JGA |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, samples, accuracy in jga_buckets(test_records):
        lines.append(f"| {label} | {samples} | {percent(accuracy)} |")
    lines.extend(
        [
            "",
            "## 典型错误样例",
            "",
            "### 非工具轮误调用",
            "",
            *sample_rows(test_records, lambda row: bool(row.get("false_tool_call"))),
            "",
            "### 工具或参数错误",
            "",
            *sample_rows(
                test_records,
                lambda row: row.get("target_decision") == "call_tool"
                and (not row.get("tool_correct", False) or not row.get("arguments_exact", False)),
            ),
            "",
            "## 结论与边界",
            "",
            "- QLoRA 显著提升了结构化输出、多轮状态维护、决策路由和工具选择能力。",
            "- challenge 的状态与决策指标没有明显崩塌，当前 Adapter 可作为主实验结果冻结。",
            "- 剩余主要问题是长状态全量匹配、参数精确生成和结束轮误调用；这些属于可解释的改进方向，不应表述为生产可用。",
            "- CrossWOZ 与受控缺参样本用于可复现实验，不代表真实线上流量。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved Base and QLoRA reports")
    parser.add_argument("--base", type=Path, default=Path("reports/tool_use_base_4bit.json"))
    parser.add_argument("--qlora", type=Path, default=Path("reports/tool_use_qlora_4bit.json"))
    parser.add_argument(
        "--challenge", type=Path, default=Path("reports/tool_use_qlora_challenge_4bit.json")
    )
    parser.add_argument("--test-data", type=Path, default=Path("data/tool_use/test.jsonl"))
    parser.add_argument(
        "--challenge-data", type=Path, default=Path("data/tool_use/challenge.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/EXPERIMENT_REPORT.md"))
    args = parser.parse_args()

    base_metrics, _ = enrich_report(args.base, args.test_data)
    test_metrics, test_records = enrich_report(args.qlora, args.test_data)
    challenge_metrics, challenge_records = enrich_report(args.challenge, args.challenge_data)
    output = build_report(
        base_metrics, test_metrics, challenge_metrics, test_records, challenge_records
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"报告已保存：{args.output}")
    print(json.dumps({"test": test_metrics, "challenge": challenge_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
