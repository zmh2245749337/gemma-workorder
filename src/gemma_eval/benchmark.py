"""Scenario slicing for Agent tool-use evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import validate_decision
from .tool_use_data import aggregate_metrics, compare_decisions, load_jsonl, state_item_count


def benchmark_tags(row: dict[str, Any]) -> set[str]:
    target = row["output"]
    tags = {"all"}
    decision = target["decision"]
    tags.add(decision)
    if row.get("history") or row.get("previous_state"):
        tags.add("multi_turn_state")
    if state_item_count(target["belief_state"]) >= 11:
        tags.add("long_state")
    source = row.get("source", {})
    if source.get("type") == "controlled_required_slot_ablation":
        tags.add("missing_argument_controlled")
    call = target.get("tool_call")
    if call and call.get("name") == "request_taxi":
        tags.add("side_effect_tool")
    tags.update(row.get("challenge_categories", []))
    return tags


def load_enriched_records(report_path: Path, dataset_path: Path) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = {str(row["sample_id"]): row for row in load_jsonl(dataset_path)}
    enriched: list[dict[str, Any]] = []
    for original in report["records"]:
        record = dict(original)
        row = rows.get(str(record.get("sample_id")))
        if row is None:
            raise ValueError(f"sample {record.get('sample_id')} missing from {dataset_path}")
        target = row["output"]
        record["target_decision"] = target["decision"]
        record["state_target_items"] = state_item_count(target["belief_state"])
        record["benchmark_tags"] = sorted(benchmark_tags(row))
        if record.get("schema_valid") and record.get("prediction"):
            record.update(compare_decisions(validate_decision(record["prediction"]), target))
        enriched.append(record)
    return enriched


def slice_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    tags = sorted({tag for record in records for tag in record.get("benchmark_tags", [])})
    output: dict[str, dict[str, float | int]] = {}
    for tag in tags:
        selected = [
            record for record in records if tag in record.get("benchmark_tags", [])
        ]
        metrics = aggregate_metrics(selected)
        metrics["tool_target_samples"] = sum(
            record.get("target_decision") == "call_tool" for record in selected
        )
        metrics["non_tool_target_samples"] = sum(
            record.get("target_decision") != "call_tool" for record in selected
        )
        output[tag] = metrics
    return output
