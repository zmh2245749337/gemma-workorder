import tempfile
from pathlib import Path

import pytest

from gemma_eval.service_query import (
    ToolCall,
    execute_local_tool,
    extract_json,
    initialise_demo_database,
    requires_human_confirmation,
    route_tool,
    run_service_query,
    validate_fields,
)
from gemma_eval.service_query_data import aggregate_metrics, compare_fields, service_query_prompt

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "service_query" / "reference"


@pytest.fixture()
def demo_db():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "service_query.sqlite"
        initialise_demo_database(path, REFERENCE_DIR)
        yield path


def test_restaurant_query_returns_a_real_record(demo_db):
    # "北京全聚德" is a real Beijing restaurant present in the trimmed
    # reference subset (see data/service_query/reference/restaurant_db.json,
    # which only keeps entities that actually appear in the generated
    # train/validation/test splits).
    payload = {
        "domain": "餐馆",
        "constraints": {"名称": "北京全聚德"},
        "requested_fields": ["评分"],
        "missing_fields": [],
        "next_action": "query_restaurant_db",
    }
    result = run_service_query(payload, demo_db)
    assert result["tool_call"]["name"] == "query_restaurant_db"
    assert result["tool_result"]["status"] == "ok"
    assert result["tool_result"]["records"][0]["名称"] == "北京全聚德"
    assert result["draft"]["status"] == "auto_executed"


def test_taxi_request_requires_human_confirmation_and_does_not_auto_execute(demo_db):
    payload = {
        "domain": "出租",
        "constraints": {"出发地": "乾清宫", "目的地": "姚记炒肝店（鼓楼店）"},
        "requested_fields": ["车型", "车牌"],
        "missing_fields": [],
        "next_action": "request_taxi",
    }
    result = run_service_query(payload, demo_db)
    assert result["tool_call"]["name"] == "request_taxi"
    assert requires_human_confirmation(ToolCall(**result["tool_call"])) is True
    assert result["tool_result"] is None  # never auto-executed
    assert result["draft"]["status"] == "requires_human_confirmation"


def test_taxi_request_without_both_endpoints_falls_back_to_clarification(demo_db):
    payload = {
        "domain": "出租",
        "constraints": {"出发地": "乾清宫"},
        "requested_fields": [],
        "missing_fields": ["目的地"],
        "next_action": "request_taxi",
    }
    result = run_service_query(payload, demo_db)
    assert result["tool_call"]["name"] == "ask_clarification"


def test_model_json_schema_rejects_unknown_tool():
    payload = {"domain": "餐馆", "next_action": "delete_database"}
    with pytest.raises(ValueError, match="allow-listed"):
        validate_fields(payload)


def test_model_json_schema_rejects_unknown_domain():
    payload = {"domain": "股票", "next_action": "ask_clarification"}
    with pytest.raises(ValueError, match="unknown domain"):
        validate_fields(payload)


def test_extract_json_from_fenced_model_response():
    payload = extract_json("好的：```json\n{\"domain\": \"景点\"}\n```")
    assert payload["domain"] == "景点"


def test_query_for_unknown_name_is_safe(demo_db):
    result = execute_local_tool(demo_db, ToolCall("query_hotel_db", {"constraints": {"名称": "不存在的酒店"}}))
    assert result["status"] == "not_found"


def test_training_prompt_and_metric_helpers_are_deterministic():
    prompt = service_query_prompt("帮我推荐一家吃烤鸭的餐馆")
    assert "只输出一个JSON对象" in prompt
    target = {
        "domain": "餐馆",
        "constraints": {"推荐菜": "烤鸭"},
        "requested_fields": ["名称"],
        "missing_fields": [],
        "next_action": "query_restaurant_db",
    }
    comparison = compare_fields(target, target)
    metrics = aggregate_metrics([{"json_valid": True, "tool_correct": True, **comparison}])
    assert comparison["exact_match"] is True
    assert metrics["field_exact_rate"] == 1.0
    assert metrics["tool_accuracy"] == 1.0
