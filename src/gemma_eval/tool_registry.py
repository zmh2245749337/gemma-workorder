"""Typed tool catalogue used by prompts, validation, policy and runtime.

The registry is the single source of truth for what the Agent is allowed to
propose.  Risk and execution metadata live in code rather than model output,
so a prompt or hallucinated field cannot grant additional permission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    domain: str
    risk: str
    parameters: Mapping[str, Any]
    required_arguments: tuple[str, ...] = ()
    executor: str = "local"

    def to_function_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function-tool declaration."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }

    def validate_argument_shape(self, arguments: dict[str, Any]) -> None:
        """Validate declared keys and basic JSON types before policy checks.

        Missing required values are deliberately handled by the Policy Engine,
        which can turn an incomplete call into ``needs_input`` instead of
        treating it as an executable request.
        """

        properties = self.parameters.get("properties", {})
        if self.parameters.get("additionalProperties") is False:
            unknown = set(arguments) - set(properties)
            if unknown:
                raise ValueError(f"{self.name} has unknown arguments: {sorted(unknown)}")
        python_types = {"object": dict, "array": list, "string": str}
        for key, value in arguments.items():
            expected = properties.get(key, {}).get("type")
            expected_type = python_types.get(expected)
            if expected_type and not isinstance(value, expected_type):
                raise ValueError(f"{self.name}.{key} must be {expected}")
            if expected == "array" and not all(isinstance(item, str) for item in value):
                raise ValueError(f"{self.name}.{key} must contain strings")


class ToolRegistry:
    """Small immutable-style registry with mapping and export helpers."""

    def __init__(self, specs: list[ToolSpec] | tuple[ToolSpec, ...]):
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("tool names must be unique")

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> ToolSpec:
        return self._specs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def values(self):
        return self._specs.values()

    def as_dict(self) -> dict[str, ToolSpec]:
        return dict(self._specs)

    def prompt_lines(self) -> list[str]:
        lines = []
        for spec in self.values():
            required = "、".join(spec.required_arguments) or "无固定必填字段"
            lines.append(
                f"- {spec.name}: {spec.description}; 领域={spec.domain}; 必填参数={required}"
            )
        return lines

    def to_function_schemas(self) -> list[dict[str, Any]]:
        return [spec.to_function_schema() for spec in self.values()]


ENTITY_PARAMETERS = {
    "type": "object",
    "properties": {
        "constraints": {
            "type": "object",
            "description": "用户已经明确给出的检索约束，不得补写未知条件",
            "additionalProperties": True,
        },
        "requested_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "用户希望工具返回的字段",
        },
    },
    "required": ["constraints", "requested_fields"],
    "additionalProperties": False,
}

ROUTE_PARAMETERS = {
    "type": "object",
    "properties": {
        "出发地": {"type": "string"},
        "目的地": {"type": "string"},
    },
    "required": ["出发地", "目的地"],
    "additionalProperties": False,
}


DEFAULT_TOOL_REGISTRY = ToolRegistry(
    [
        ToolSpec(
            "query_attraction_db",
            "按约束查询景点并返回用户请求的字段",
            "景点",
            "read_only",
            ENTITY_PARAMETERS,
        ),
        ToolSpec(
            "query_hotel_db",
            "按约束查询酒店并返回用户请求的字段",
            "酒店",
            "read_only",
            ENTITY_PARAMETERS,
        ),
        ToolSpec(
            "query_restaurant_db",
            "按约束查询餐馆并返回用户请求的字段",
            "餐馆",
            "read_only",
            ENTITY_PARAMETERS,
        ),
        ToolSpec(
            "query_metro_route",
            "根据明确的出发地和目的地查询地铁站点",
            "地铁",
            "read_only",
            ROUTE_PARAMETERS,
            ("出发地", "目的地"),
        ),
        ToolSpec(
            "request_taxi",
            "根据明确的出发地和目的地创建打车请求；执行前必须人工确认",
            "出租",
            "side_effect",
            ROUTE_PARAMETERS,
            ("出发地", "目的地"),
        ),
    ]
)
