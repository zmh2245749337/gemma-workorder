import json

import pytest

from gemma_eval.tool_use import (
    ToolCall,
    apply_policy,
    execute_tool,
    extract_json,
    initialise_demo_database,
    run_decision,
    validate_decision,
)
from gemma_eval.tool_use_data import aggregate_metrics, compare_decisions, tool_use_prompt
from gemma_eval.benchmark import benchmark_tags, slice_metrics
from gemma_eval.tool_registry import DEFAULT_TOOL_REGISTRY
from gemma_eval.traces import row_to_agent_trace


def payload(decision="no_tool", tool_call=None, missing_slots=None):
    return {
        "belief_state": [],
        "decision": decision,
        "tool_call": tool_call,
        "missing_slots": missing_slots or [],
    }


def write_reference_database(directory):
    directory.mkdir()
    (directory / "attraction_db.json").write_text(
        json.dumps([["故宫", {"名称": "故宫", "评分": "5分", "门票": "60元"}]], ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / "hotel_db.json").write_text("[]", encoding="utf-8")
    (directory / "restaurant_db.json").write_text("[]", encoding="utf-8")
    (directory / "metro_db.json").write_text(
        json.dumps(
            [["故宫", {"地铁": "天安门东站"}], ["颐和园", {"地铁": "北宫门站"}]],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_prompt_contains_history_and_previous_state():
    prompt = tool_use_prompt(
        [{"role": "user", "content": "想去故宫"}, {"role": "assistant", "content": "好的"}],
        "它附近有酒店吗？",
        [{"goal_id": 1, "domain": "景点", "constraints": {"名称": "故宫"}, "requested_fields": []}],
    )
    assert "想去故宫" in prompt
    assert "它附近有酒店吗" in prompt
    assert '"名称": "故宫"' in prompt


def test_schema_rejects_unknown_tool_and_empty_clarification():
    with pytest.raises(ValueError, match="not registered"):
        validate_decision(payload("call_tool", {"name": "run_shell", "arguments": {}}))
    with pytest.raises(ValueError, match="requires at least one"):
        validate_decision(payload("ask_user"))
    with pytest.raises(ValueError, match="unknown arguments"):
        validate_decision(
            payload(
                "call_tool",
                {"name": "query_hotel_db", "arguments": {"sql": "DROP TABLE hotels"}},
            )
        )


def test_missing_slot_routes_to_clarification():
    decision = validate_decision(payload("ask_user", missing_slots=["目的地"]))
    outcome = apply_policy(decision)
    assert outcome.status == "needs_input"
    assert outcome.tool_call is None


def test_taxi_never_auto_executes(tmp_path):
    decision = validate_decision(
        payload(
            "call_tool",
            {"name": "request_taxi", "arguments": {"出发地": "故宫", "目的地": "颐和园"}},
        )
    )
    assert apply_policy(decision).status == "pending_confirmation"
    with pytest.raises(PermissionError, match="explicit confirmation"):
        execute_tool(tmp_path / "demo.db", decision.tool_call)


def test_read_only_tools_execute_against_local_database(tmp_path):
    reference = tmp_path / "reference"
    write_reference_database(reference)
    database = tmp_path / "demo.db"
    initialise_demo_database(database, reference)
    query = payload(
        "call_tool",
        {
            "name": "query_attraction_db",
            "arguments": {"constraints": {"名称": "故宫"}, "requested_fields": ["门票"]},
        },
    )
    result = run_decision(query, database)
    assert result["policy"]["status"] == "auto_execute"
    assert result["tool_result"]["records"] == [{"名称": "故宫", "门票": "60元"}]

    metro = execute_tool(
        database, ToolCall("query_metro_route", {"出发地": "故宫", "目的地": "颐和园"})
    )
    assert metro["route"]["起点站"] == "天安门东站"
    assert metro["route"]["终点站"] == "北宫门站"


def test_unknown_entity_field_is_not_silently_ignored(tmp_path):
    reference = tmp_path / "reference"
    write_reference_database(reference)
    database = tmp_path / "demo.db"
    initialise_demo_database(database, reference)
    result = execute_tool(
        database,
        ToolCall(
            "query_attraction_db",
            {"constraints": {"不存在的字段": "随便"}, "requested_fields": []},
        ),
    )
    assert result["status"] == "not_found"


def test_tolerant_json_parser_and_metrics():
    target = payload("ask_user", missing_slots=["目的地"])
    prediction = validate_decision(extract_json("结果如下：```json\n" + json.dumps(target) + "\n```"))
    record = {
        "strict_json_valid": False,
        "schema_valid": True,
        "target_decision": "ask_user",
        **compare_decisions(prediction, target),
    }
    metrics = aggregate_metrics([record])
    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["strict_json_valid_rate"] == 0.0
    assert metrics["ask_user_decision_accuracy"] == 1.0


def test_argument_slot_metrics_keep_partial_credit():
    target = payload(
        "call_tool",
        {
            "name": "query_restaurant_db",
            "arguments": {
                "constraints": {"评分": "4.5分以上", "人均消费": "100-150元"},
                "requested_fields": ["名称"],
            },
        },
    )
    prediction = validate_decision(
        payload(
            "call_tool",
            {
                "name": "query_restaurant_db",
                "arguments": {
                    "constraints": {"评分": "4.5分以上"},
                    "requested_fields": ["名称", "地址"],
                },
            },
        )
    )
    record = {
        "strict_json_valid": True,
        "schema_valid": True,
        "target_decision": "call_tool",
        **compare_decisions(prediction, target),
    }
    metrics = aggregate_metrics([record])
    assert metrics["arguments_exact_rate"] == 0.0
    assert metrics["argument_slot_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["argument_slot_recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["argument_slot_f1"] == pytest.approx(2 / 3, abs=1e-4)


def test_required_argument_accuracy_is_reported_separately():
    target = payload(
        "call_tool",
        {"name": "query_metro_route", "arguments": {"出发地": "故宫", "目的地": "颐和园"}},
    )
    prediction = validate_decision(
        payload(
            "call_tool",
            {"name": "query_metro_route", "arguments": {"出发地": "故宫", "目的地": "颐和园"}},
        )
    )
    record = {
        "strict_json_valid": True,
        "schema_valid": True,
        "target_decision": "call_tool",
        **compare_decisions(prediction, target),
    }
    metrics = aggregate_metrics([record])
    assert metrics["required_argument_accuracy"] == 1.0


def test_registry_exports_function_schemas_and_risk_is_code_owned():
    schemas = DEFAULT_TOOL_REGISTRY.to_function_schemas()
    taxi = next(item for item in schemas if item["function"]["name"] == "request_taxi")
    assert taxi["function"]["parameters"]["required"] == ["出发地", "目的地"]
    assert DEFAULT_TOOL_REGISTRY["request_taxi"].risk == "side_effect"
    assert "risk" not in taxi["function"]["parameters"]["properties"]


def test_agent_trace_connects_context_target_and_policy():
    row = {
        "sample_id": "trace-1",
        "history": [{"role": "user", "content": "我从故宫出发"}],
        "previous_state": [],
        "current_user": "帮我叫车去颐和园",
        "output": payload(
            "call_tool",
            {
                "name": "request_taxi",
                "arguments": {"出发地": "故宫", "目的地": "颐和园"},
            },
        ),
        "source": {"dialog_id": "demo"},
    }
    trace = row_to_agent_trace(row)
    assert trace["schema_version"] == "1.0"
    assert trace["events"][1]["stage"] == "model_target"
    assert trace["events"][2]["payload"]["status"] == "pending_confirmation"


def test_benchmark_tags_and_slice_metrics():
    row = {
        "sample_id": "controlled-1",
        "history": [],
        "previous_state": [],
        "current_user": "我从故宫出发，帮我叫车",
        "output": payload("ask_user", missing_slots=["目的地"]),
        "source": {"type": "controlled_required_slot_ablation"},
    }
    assert {"ask_user", "missing_argument_controlled"} <= benchmark_tags(row)
    result = slice_metrics(
        [
            {
                "benchmark_tags": ["all", "ask_user"],
                "strict_json_valid": True,
                "schema_valid": True,
                "target_decision": "ask_user",
                "decision_correct": True,
            }
        ]
    )
    assert result["ask_user"]["decision_accuracy"] == 1.0
