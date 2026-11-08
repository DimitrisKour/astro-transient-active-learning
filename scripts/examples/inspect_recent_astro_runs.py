#!/usr/bin/env python3
"""Read-only inventory of recent AstroMiNN closed-loop policy experiments.

The teaching historical replay should use completed runs with compatible
training configurations. This tool discovers what exists; it does not claim
cross-suite comparability or export candidate IDs, labels, or predictions.
Requires only Python's standard library. It does not access sample images.

Example:
    python scripts/examples/inspect_recent_astro_runs.py \
        --persistent-root /projects/bcrv/$USER/applecider_adaptive \
        --artifact-root /work/hdd/bcrv/$USER/applecider_adaptive \
        --output ~/astro_recent_inventory.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_SUITES = (
    "astrominn_closed_loop_warm_v1",
    "astrominn_closed_loop_warm_f1best_085_v1",
    "astrominn_task11_advanced_acquisition_v1",
    "astrominn_task12_bald_mc20_v1",
    "astrominn_task13_tailor_pilot_v1",
    "astrominn_task13_tailor_extension_v1",
    "astrominn_task14_similar_facility_v1",
    "astrominn_task15_direct_vreduce_v1",
    "astrominn_task16_batchsize_sensitivity_v1",
    "astrominn_task17_random_l0_v1",
    "astrominn_task18_anchoral_v1",
    "astrominn_bald_tailor_robust_random_l0_l0_135_v1",
    "astrominn_bald_tailor_robust_random_l0_l0_246_v1",
    "astrominn_bald_tailor_robust_random_l0_l0_357_v1",
    "astrominn_bald_tailor_robust_stratified_l0_135_v1",
    "astrominn_bald_tailor_robust_stratified_l0_246_v1",
    "astrominn_bald_tailor_robust_stratified_l0_357_v1",
)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return {"_read_error": type(exc).__name__}


def inspect_suite(name: str, persistent_root: Path, artifact_root: Path) -> dict:
    state_root = persistent_root / "runs" / name
    artifact_suite = artifact_root / "runs" / name
    found = []
    if state_root.is_dir():
        for policy_dir in sorted(state_root.iterdir()):
            if not policy_dir.is_dir() or policy_dir.name == "shared":
                continue
            for seed_dir in sorted(policy_dir.glob("seed_*")):
                if not seed_dir.is_dir():
                    continue
                run_state = read_json(seed_dir / "run_state.json")
                completed = run_state.get("last_completed_cycle")
                if not isinstance(completed, int):
                    completed = None
                artifact_run = artifact_suite / policy_dir.name / seed_dir.name
                cycle_list = (list(range(completed + 1)) if completed is not None and completed >= 0 else [])
                available_evals = []
                missing_evals = []
                sample_keys = None
                for cycle in cycle_list:
                    ev = artifact_run / f"cycle_{cycle:03d}" / "evaluation" / "evaluation_summary.json"
                    if ev.is_file():
                        available_evals.append(cycle)
                        if sample_keys is None:
                            sample = read_json(ev)
                            sample_keys = sorted(sample) if isinstance(sample, dict) else [type(sample).__name__]
                    else:
                        missing_evals.append(cycle)
                found.append({
                    "policy": policy_dir.name, "seed": seed_dir.name,
                    "last_completed_cycle": completed,
                    "active_cycle": run_state.get("active_cycle"),
                    "state_read_error": run_state.get("_read_error"),
                    "evaluation_cycles": available_evals,
                    "missing_evaluation_cycles": missing_evals,
                    "evaluation_summary_keys": sample_keys,
                })
    return {"suite": name, "state_directory_found": state_root.is_dir(),
            "artifact_directory_found": artifact_suite.is_dir(), "runs": found}


def inventory(persistent_root: Path, artifact_root: Path, suites=DEFAULT_SUITES) -> dict:
    return {"purpose": "read-only inventory for teaching replay selection",
            "persistent_root": str(persistent_root), "artifact_root": str(artifact_root),
            "suites": [inspect_suite(s, persistent_root, artifact_root) for s in suites]}


def print_summary(data):
    for suite in data["suites"]:
        runs = suite["runs"]
        if not runs and not suite["state_directory_found"]:
            print(f"{suite['suite']}: no persistent run directory")
            continue
        print(f"\n{suite['suite']}: {len(runs)} run(s); archive={'yes' if suite['artifact_directory_found'] else 'no'}")
        by_policy = defaultdict(list)
        for run in runs:
            by_policy[run["policy"]].append(run)
        for policy, group in sorted(by_policy.items()):
            tags = []
            for item in group:
                cycle = item["last_completed_cycle"]
                available = len(item["evaluation_cycles"])
                expected = (cycle + 1) if cycle is not None and cycle >= 0 else "?"
                tag = f"{item['seed']}:cycle={cycle},eval={available}/{expected}"
                if item["active_cycle"] is not None:
                    tag += ",ACTIVE"
                if item["state_read_error"]:
                    tag += f",state_error={item['state_read_error']}"
                tags.append(tag)
            print(f"  {policy} | " + " | ".join(tags))
        schema = next((r["evaluation_summary_keys"] for r in runs if r["evaluation_summary_keys"]), None)
        if schema:
            print("  evaluation_summary keys:", ", ".join(schema[:25]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--persistent-root", required=True, type=Path)
    p.add_argument("--artifact-root", required=True, type=Path)
    p.add_argument("--suite", action="append", help="Use a specific suite; repeatable")
    p.add_argument("--output", type=Path, help="Optional local JSON inventory for detailed review")
    a = p.parse_args()
    data = inventory(a.persistent_root, a.artifact_root, tuple(a.suite) if a.suite else DEFAULT_SUITES)
    print_summary(data)
    if a.output:
        output = a.output.expanduser()
        if output.exists():
            p.error(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved run inventory: {output}")


if __name__ == "__main__":
    main()