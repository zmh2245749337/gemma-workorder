"""Export labelled turns as portable Agent Trace JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(ROOT / "src")
if SOURCE_ROOT in sys.path:
    sys.path.remove(SOURCE_ROOT)
sys.path.insert(0, SOURCE_ROOT)

from gemma_eval.tool_use_data import load_jsonl
from gemma_eval.traces import row_to_agent_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical Agent Trace JSONL")
    parser.add_argument("--input", type=Path, default=Path("data/tool_use/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/traces/train.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row_to_agent_trace(row), ensure_ascii=False) + "\n")
    print(json.dumps({"input": str(args.input), "output": str(args.output), "traces": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
