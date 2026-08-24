"""Run the reproducible service-query safety pipeline without downloading model weights.

Unlike run_workorder_demo.py, this does not include a rule-based Chinese
text parser: CrossWOZ spans five open domains and dozens of slots, and a
general-purpose rule-based Chinese NLU parser for all of them is a
different, larger project than this one. This demo instead takes an
already-extracted fields payload -- exactly the shape the fine-tuned model
is trained to output -- and shows the safe-execution side of the pipeline:
schema validation, tool routing, and the human-confirmation gate for
request_taxi, running against the real CrossWOZ reference data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma_eval.service_query import initialise_demo_database, run_service_query

DEFAULT_PAYLOAD = {
    "domain": "餐馆",
    "constraints": {"名称": "北京全聚德", "推荐菜": "烤鸭"},
    "requested_fields": ["地址", "评分"],
    "missing_fields": [],
    "next_action": "query_restaurant_db",
}

TAXI_PAYLOAD = {
    "domain": "出租",
    "constraints": {"出发地": "乾清宫", "目的地": "姚记炒肝店（鼓楼店）"},
    "requested_fields": ["车型", "车牌"],
    "missing_fields": [],
    "next_action": "request_taxi",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma-ServiceQuery Phase A local demo")
    parser.add_argument("--example", choices=("restaurant", "taxi"), default="restaurant")
    parser.add_argument("--db", type=Path, default=Path("artifacts/service_query_demo.sqlite"))
    parser.add_argument("--reference-dir", type=Path, default=Path("data/service_query/reference"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    initialise_demo_database(args.db, args.reference_dir)
    payload = TAXI_PAYLOAD if args.example == "taxi" else DEFAULT_PAYLOAD
    result = run_service_query(payload, args.db)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
