"""Prompt, dataset and metrics helpers for multi-turn tool use."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import AgentDecision, validate_decision
from .tool_registry import DEFAULT_TOOL_REGISTRY


SYSTEM_PROMPT = """你是任务型Agent的工具决策模块。请根据完整对话历史维护用户当前状态，
并决定本轮是调用工具、继续追问，还是不需要工具。只输出一个JSON对象，不输出Markdown或解释。

输出字段固定为：
- belief_state: 已经在对话中明确提及的目标列表；每项包含goal_id、domain、constraints、requested_fields
- decision: 只能是call_tool、ask_user、no_tool之一
- tool_call: call_tool时包含name和arguments，其他情况必须为null
- missing_slots: 需要继续追问的字段列表

不要编造对话中没有出现的参数。工具风险由外部策略判断，模型不得自行宣称已执行有副作用的操作。"""


def tool_registry_prompt() -> str:
    rows = []
    return "\n".join(DEFAULT_TOOL_REGISTRY.prompt_lines())


def tool_use_prompt(
    history: list[dict[str, str]],
    current_user: str,
    previous_state: list[dict[str, Any]] | None = None,
) -> str:
    transcript = []
    for message in history:
        role = "用户" if message["role"] == "user" else "助手"
        transcript.append(f"{role}：{message['content']}")
    transcript.append(f"用户：{current_user}")
    return (
        f"{SYSTEM_PROMPT}\n\n可用工具：\n{tool_registry_prompt()}\n\n"
        f"上一轮已确认状态：\n{json.dumps(previous_state or [], ensure_ascii=False)}\n\n"
        f"最近对话：\n" + "\n".join(transcript) + "\n\nJSON："
    )


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is invalid JSONL") from error
            if (
                not isinstance(row.get("history"), list)
                or not isinstance(row.get("previous_state", []), list)
                or not isinstance(row.get("current_user"), str)
            ):
                raise ValueError(f"{path}:{number} requires history, previous_state and current_user")
            validate_decision(row.get("output", {}))
            rows.append(row)
    return rows


def _state_items(state: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    items: set[tuple[Any, ...]] = set()
    for goal in state:
        prefix = (goal["goal_id"], goal["domain"])
        for slot, value in goal.get("constraints", {}).items():
            items.add((*prefix, "constraint", slot, canonical_json(value) if isinstance(value, dict) else str(value)))
        for slot in goal.get("requested_fields", []):
            items.add((*prefix, "request", slot, ""))
    return items


def state_item_count(state: list[dict[str, Any]]) -> int:
    return len(_state_items(state))


def _argument_items(arguments: dict[str, Any]) -> set[tuple[str, ...]]:
    """Flatten tool arguments into order-independent semantic slot items."""

    items: set[tuple[str, ...]] = set()

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, str(key)))
        elif isinstance(value, list):
            for child in value:
                visit(child, path)
        else:
            items.add((*path, str(value)))

    visit(arguments, ())
    return items


def compare_decisions(prediction: AgentDecision, target_payload: dict[str, Any]) -> dict[str, Any]:
    target = validate_decision(target_payload)
    predicted_state = _state_items(prediction.belief_state)
    target_state = _state_items(target.belief_state)
    matched = predicted_state & target_state
    predicted_call = prediction.tool_call.to_dict() if prediction.tool_call else None
    target_call = target.tool_call.to_dict() if target.tool_call else None
    predicted_arguments = (
        _argument_items(prediction.tool_call.arguments) if prediction.tool_call else set()
    )
    target_arguments = _argument_items(target.tool_call.arguments) if target.tool_call else set()
    matched_arguments = predicted_arguments & target_arguments
    required_argument_case = bool(
        target.tool_call and DEFAULT_TOOL_REGISTRY[target.tool_call.name].required_arguments
    )
    required_arguments_correct = False
    if required_argument_case and target.tool_call and prediction.tool_call:
        required = DEFAULT_TOOL_REGISTRY[target.tool_call.name].required_arguments
        required_arguments_correct = (
            prediction.tool_call.name == target.tool_call.name
            and all(
                prediction.tool_call.arguments.get(key) == target.tool_call.arguments.get(key)
                for key in required
            )
        )
    return {
        "state_exact": predicted_state == target_state,
        "state_predicted_items": len(predicted_state),
        "state_target_items": len(target_state),
        "state_matched_items": len(matched),
        "decision_correct": prediction.decision == target.decision,
        "tool_correct": (
            prediction.tool_call is not None
            and target.tool_call is not None
            and prediction.tool_call.name == target.tool_call.name
        )
        if target.decision == "call_tool"
        else prediction.decision == target.decision,
        "arguments_exact": predicted_call == target_call if target.decision == "call_tool" else True,
        "argument_predicted_items": len(predicted_arguments),
        "argument_target_items": len(target_arguments),
        "argument_matched_items": len(matched_arguments),
        "required_argument_case": required_argument_case,
        "required_arguments_correct": required_arguments_correct,
        "false_tool_call": target.decision != "call_tool" and prediction.decision == "call_tool",
    }


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(records)
    if not total:
        return {"samples": 0}

    predicted_items = sum(item.get("state_predicted_items", 0) for item in records)
    target_items = sum(item.get("state_target_items", 0) for item in records)
    matched_items = sum(item.get("state_matched_items", 0) for item in records)
    precision = matched_items / predicted_items if predicted_items else 0.0
    recall = matched_items / target_items if target_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    argument_predicted_items = sum(item.get("argument_predicted_items", 0) for item in records)
    argument_target_items = sum(item.get("argument_target_items", 0) for item in records)
    argument_matched_items = sum(item.get("argument_matched_items", 0) for item in records)
    argument_precision = (
        argument_matched_items / argument_predicted_items if argument_predicted_items else 0.0
    )
    argument_recall = (
        argument_matched_items / argument_target_items if argument_target_items else 0.0
    )
    argument_f1 = (
        2 * argument_precision * argument_recall / (argument_precision + argument_recall)
        if argument_precision + argument_recall
        else 0.0
    )

    def rate(key: str, subset: list[dict[str, Any]] | None = None) -> float:
        selected = records if subset is None else subset
        return round(sum(bool(item.get(key, False)) for item in selected) / len(selected), 4) if selected else 0.0

    tool_records = [item for item in records if item.get("target_decision") == "call_tool"]
    non_tool_records = [item for item in records if item.get("target_decision") != "call_tool"]
    required_argument_records = [item for item in records if item.get("required_argument_case")]

    return {
        "samples": total,
        "strict_json_valid_rate": rate("strict_json_valid"),
        "schema_valid_rate": rate("schema_valid"),
        "state_joint_goal_accuracy": rate("state_exact"),
        "state_slot_precision": round(precision, 4),
        "state_slot_recall": round(recall, 4),
        "state_slot_f1": round(f1, 4),
        "decision_accuracy": rate("decision_correct"),
        "tool_accuracy": rate("tool_correct", tool_records),
        "arguments_exact_rate": rate("arguments_exact", tool_records),
        "argument_slot_precision": round(argument_precision, 4),
        "argument_slot_recall": round(argument_recall, 4),
        "argument_slot_f1": round(argument_f1, 4),
        "required_argument_accuracy": rate(
            "required_arguments_correct", required_argument_records
        ),
        "no_tool_false_positive_rate": rate("false_tool_call", non_tool_records),
        "call_tool_decision_accuracy": rate(
            "decision_correct", [item for item in records if item.get("target_decision") == "call_tool"]
        ),
        "ask_user_decision_accuracy": rate(
            "decision_correct", [item for item in records if item.get("target_decision") == "ask_user"]
        ),
        "no_tool_decision_accuracy": rate(
            "decision_correct", [item for item in records if item.get("target_decision") == "no_tool"]
        ),
    }
