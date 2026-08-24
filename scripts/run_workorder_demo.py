"""Run the reproducible Phase A work-order baseline without downloading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permit the documented script to run before editable installation as well.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma_eval.workorder import initialise_demo_database, run_phase_a


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma-WorkOrder Phase A local demo")
    parser.add_argument(
        "--text",
        default="A区3号风机今天上午频繁停机，面板显示E07，重启后运行十分钟再次停止，暂时没有更换零件。",
    )
    parser.add_argument("--db", type=Path, default=Path("artifacts/workorder_demo.sqlite"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    initialise_demo_database(args.db)
    payload = run_phase_a(args.text, args.db)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
