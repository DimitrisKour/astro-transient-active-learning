#!/usr/bin/env python
"""Run one frozen diagnostic task; no active-learning loop or bootstrap."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from al_pipeline.astronomy.training_diagnostics import load_plan, run_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        plan = load_plan(args.plan)
        print(json.dumps(plan["tasks"], indent=2))
        return
    if args.task_index is None:
        parser.error("--task-index is required unless --describe is used")
    result_path = run_task(args.plan, args.task_index, device=args.device)
    result = json.loads(result_path.read_text())
    print(json.dumps({
        "task": result["task"], "result_path": str(result_path),
        "object_metrics": {s["split"]: s["object_metrics"] for s in result["evaluation"]},
    }, indent=2))


if __name__ == "__main__":
    main()
