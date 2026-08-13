import tempfile
from pathlib import Path

import pytest

from gemma_eval.workorder import (
    ToolCall,
    execute_local_tool,
    extract_json,
    initialise_demo_database,
    route_tool,
    run_phase_a,
    validate_fields,
)
from gemma_eval.workorder_data import aggregate_metrics, compare_fields, workorder_prompt


@pytest.fixture()
def demo_db():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "workorder.sqlite"
        initialise_demo_database(path)
        yield path


def test_end_to_end_phase_a_routes_fault_code(demo_db):
    result = run_phase_a("A区3号风机今天上午显示E07，重启十分钟后再次停机。", demo_db)
    assert result["structured_fields"]["fault_code"] == "E07"
    assert result["tool_call"]["name"] == "query_fault_code"
    assert result["tool_result"]["record"]["meaning"] == "过载保护触发"
    assert result["draft"]["status"] == "draft_requires_human_confirmation"


def test_missing_fault_code_routes_to_clarification(demo_db):
    result = run_phase_a("A区3号风机今天上午反复停机，已经重启。", demo_db)
    assert "故障码" in result["structured_fields"]["missing_fields"]
    assert result["tool_call"]["name"] == "ask_clarification"


def test_model_json_schema_rejects_unknown_tool():
    payload = {
        "asset_name": "A区3号风机",
        "fault_code": "E07",
        "symptom": "停机",
        "occurred_at": "上午",
        "next_action": "delete_database",
    }
    with pytest.raises(ValueError, match="allow-listed"):
        validate_fields(payload)


def test_extract_json_from_fenced_model_response():
    payload = extract_json("分析结果：```json\n{\"asset_name\": \"A区3号风机\"}\n```")
    assert payload["asset_name"] == "A区3号风机"


def test_local_tool_not_found_is_safe(demo_db):
    result = execute_local_tool(demo_db, ToolCall("query_fault_code", {"code": "E99", "asset_type": "风机"}))
    assert result["status"] == "not_found"


def test_training_prompt_and_metric_helpers_are_deterministic():
    prompt = workorder_prompt("A区3号风机显示E07")
    assert "只输出一个JSON对象" in prompt
    target = {
        "asset_name": "A区3号风机",
        "asset_id": None,
        "asset_type": "风机",
        "fault_code": "E07",
        "symptom": "报警",
        "occurred_at": "今天上午",
        "actions_taken": [],
        "parts_replaced": [],
        "missing_fields": [],
        "next_action": "query_fault_code",
    }
    comparison = compare_fields(target, target)
    metrics = aggregate_metrics([{ "json_valid": True, "tool_correct": True, **comparison }])
    assert comparison["exact_match"] is True
    assert metrics["field_exact_rate"] == 1.0
    assert metrics["tool_accuracy"] == 1.0
