"""Reproducible local executors used by the guarded Agent runtime."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import ToolCall
from .tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry, ToolSpec


def initialise_demo_database(path: Path, reference_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    domain_files = {
        "景点": "attraction_db.json",
        "酒店": "hotel_db.json",
        "餐馆": "restaurant_db.json",
    }
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                domain TEXT NOT NULL,
                name TEXT NOT NULL,
                record TEXT NOT NULL,
                PRIMARY KEY (domain, name)
            );
            CREATE TABLE IF NOT EXISTS metro (
                place TEXT PRIMARY KEY,
                station TEXT NOT NULL
            );
            DELETE FROM entities;
            DELETE FROM metro;
            """
        )
        for domain, filename in domain_files.items():
            source = reference_dir / filename
            if not source.exists():
                continue
            entries = json.loads(source.read_text(encoding="utf-8"))
            connection.executemany(
                "INSERT OR REPLACE INTO entities VALUES (?, ?, ?)",
                [
                    (domain, name, json.dumps(record, ensure_ascii=False))
                    for name, record in entries
                ],
            )
        metro_source = reference_dir / "metro_db.json"
        if metro_source.exists():
            entries = json.loads(metro_source.read_text(encoding="utf-8"))
            connection.executemany(
                "INSERT OR REPLACE INTO metro VALUES (?, ?)",
                [
                    (name, str(record.get("地铁", "")))
                    for name, record in entries
                    if record.get("地铁")
                ],
            )


def _matches(record: dict[str, Any], constraints: dict[str, Any]) -> bool:
    for slot, expected in constraints.items():
        if slot not in record:
            return False
        actual = record[slot]
        if isinstance(expected, list):
            if not all(str(item) in str(actual) for item in expected):
                return False
        elif str(expected) not in str(actual) and str(actual) not in str(expected):
            return False
    return True


def _execute_entity_query(
    path: Path, spec: ToolSpec, arguments: dict[str, Any]
) -> dict[str, Any]:
    constraints = arguments.get("constraints", {}) or {}
    if not isinstance(constraints, dict):
        return {"status": "invalid_arguments", "reason": "constraints must be an object"}
    requested = arguments.get("requested_fields", []) or []
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT record FROM entities WHERE domain = ?", (spec.domain,)
        ).fetchall()
    candidates = [json.loads(row[0]) for row in rows]
    matched = [record for record in candidates if _matches(record, constraints)]
    if requested:
        records = [
            {key: record.get(key) for key in ["名称", *requested] if key in record}
            for record in matched[:5]
        ]
    else:
        records = matched[:5]
    return {"status": "ok" if records else "not_found", "records": records}


def execute_tool(
    path: Path,
    call: ToolCall,
    *,
    confirmed: bool = False,
    registry: ToolRegistry = DEFAULT_TOOL_REGISTRY,
) -> dict[str, Any]:
    """Execute only a previously validated call under explicit policy control."""

    spec = registry[call.name]
    if spec.risk == "side_effect":
        if not confirmed:
            raise PermissionError("side-effect tool requires explicit confirmation")
        return {
            "status": "simulated",
            "message": "已记录人工确认；演示环境不会调用真实派车接口。",
            "arguments": call.arguments,
        }
    if call.name == "query_metro_route":
        start, destination = call.arguments["出发地"], call.arguments["目的地"]
        with sqlite3.connect(path) as connection:
            start_row = connection.execute(
                "SELECT station FROM metro WHERE place = ?", (start,)
            ).fetchone()
            destination_row = connection.execute(
                "SELECT station FROM metro WHERE place = ?", (destination,)
            ).fetchone()
        if not start_row or not destination_row:
            return {"status": "not_found", "route": None}
        return {
            "status": "ok",
            "route": {
                "出发地": start,
                "起点站": start_row[0],
                "目的地": destination,
                "终点站": destination_row[0],
            },
        }
    return _execute_entity_query(path, spec, call.arguments)
