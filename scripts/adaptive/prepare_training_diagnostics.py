#!/usr/bin/env python
"""Preflight a training-only diagnostic suite; never acquire or train here."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from al_pipeline.astronomy.training_diagnostics import prepare_diagnostics, verify_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--diagnostic-id", default="astrominn_training_diagnostics_v1")
    parser.add_argument("--source-suite", type=Path)
    parser.add_argument("--source-artifact-root", type=Path)
    parser.add_argument("--persistent-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--trajectory-strategy", default="entropy_balanced_explore")
    parser.add_argument("--trajectory-seed", type=int, default=135)
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument("--first-repeat-seed", type=int, default=1001)
    parser.add_argument("--repeat-loader-seed", type=int)
    parser.add_argument("--profile", default="astrominn_ztf")
    parser.add_argument("--legacy-code-dir", type=Path,
                        default=Path("/projects/bcrv/ffontinelenunes/things_for_hailey"))
    parser.add_argument("--verify-plan", type=Path)
    args = parser.parse_args()
    if args.verify_plan:
        plan_path = args.verify_plan.resolve()
        plan = verify_plan(plan_path)
    else:
        if args.registry is None:
            parser.error("--registry is required for preparation")
        kwargs = vars(args).copy()
        kwargs.pop("verify_plan")
        kwargs["registry_path"] = kwargs.pop("registry")
        kwargs["profile_name"] = kwargs.pop("profile")
        plan_path = prepare_diagnostics(**kwargs)
        plan = verify_plan(plan_path)
    tasks = plan["tasks"]
    print(json.dumps({
        "plan": str(plan_path), "fits": len(tasks),
        "anchor_fits": sum(t["anchor"] for t in tasks),
        "fresh_paired_fits": sum(t["kind"] == "transition" and not t["anchor"] for t in tasks),
        "full_data_fits": sum(t["kind"] == "order" for t in tasks),
        "fresh_training_seeds": plan["repeat_seeds"],
        "acquisition_executed": False, "full_data": plan["full_data"],
        "artifact_root": plan["artifact_root"],
        "interpretation": plan["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
