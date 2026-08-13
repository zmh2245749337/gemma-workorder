"""Local review-gated API for the Gemma-WorkOrder demonstration.

This service does not submit work orders and exposes no write tool.  It is
only a local demo endpoint for structured extraction and trusted lookups.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the review-gated Gemma-WorkOrder local demo")
    parser.add_argument("--mode", choices=["baseline", "gemma"], default="baseline")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--precision", choices=["fp16", "4bit"], default="4bit")
    parser.add_argument("--db", type=Path, default=Path("artifacts/workorder_demo.sqlite"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    from gemma_eval.workorder import draft_work_order, execute_local_tool, initialise_demo_database, route_tool, validate_fields

    engine = None
    if args.mode == "gemma":
        from gemma_eval.workorder_inference import GemmaWorkOrderEngine

        engine = GemmaWorkOrderEngine(args.model_id, args.precision)
    initialise_demo_database(args.db)
    app = FastAPI(title="Gemma-WorkOrder", version="0.1.0")

    class ParseRequest(BaseModel):
        text: str

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "mode": args.mode, "safety": "drafts require human confirmation; local tools are read-only"}

    @app.post("/api/workorders/parse")
    def parse_work_order(request: ParseRequest) -> dict:
        if not request.text.strip():
            raise HTTPException(status_code=422, detail="text must not be empty")
        if args.mode == "baseline":
            from gemma_eval.workorder import run_phase_a

            return run_phase_a(request.text, args.db)
        assert engine is not None
        try:
            result = engine.parse(request.text)
            fields = validate_fields(result["structured_fields"])
            call = route_tool(fields)
            tool_result = execute_local_tool(args.db, call)
            return {
                "parser": "gemma_constrained_json",
                "raw_model_output": result["raw_model_output"],
                "structured_fields": fields.to_dict(),
                "tool_call": call.to_dict(),
                "tool_result": tool_result,
                "draft": draft_work_order(fields, tool_result),
            }
        except ValueError as error:
            raise HTTPException(status_code=422, detail=f"model output rejected by schema/safety boundary: {error}") from error

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
