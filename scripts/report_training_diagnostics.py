#!/usr/bin/env python
"""Report paired retraining and full-data order tests from small saved results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from al_pipeline.astronomy.training_diagnostic_reporting import report_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = report_diagnostics(args.plan, args.output, plots=not args.no_plots)
    print(json.dumps({
        "planned_tasks": result["planned_tasks"], "completed_tasks": result["completed_tasks"],
        "all_tasks_complete": result["all_tasks_complete"],
        "incomplete_task_indices": result["incomplete_task_indices"], "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
