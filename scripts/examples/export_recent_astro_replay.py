#!/usr/bin/env python3
"""Export small, read-only validation histories from recent closed-loop AstroMiNN runs.

One-time instructor tool; no images, checkpoints, candidate IDs, unrevealed
labels or per-object predictions are loaded or distributed. The archive is a
REVIEW artifact, not an assertion that all experiment suites are comparable.

Usage:
  python scripts/examples/export_recent_astro_replay.py \
    --persistent-root /projects/bcrv/$USER/applecider_adaptive \
    --output ~/astro_recent_school_pilot.zip
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# No dependencies beyond the standard library. Include all key methods, while
# preserving suite identities: many differ in L0/checkpoint/training settings.
DEFAULT_SUITES = {
    "astrominn_closed_loop_warm_v1": ("random", "entropy", "entropy_diversity", "entropy_balanced_explore"),
    "astrominn_task11_advanced_acquisition_v1": ("badge", "core_set", "fisher_batch"),
    "astrominn_task12_bald_mc20_v1": ("bald",),
    "astrominn_task13_tailor_pilot_v1": ("tailor_diversity", "uniform_mixture"),
    "astrominn_task14_similar_facility_v1": ("similar_fl", "similar_flcg", "similar_flqmi_rare"),
    "astrominn_task15_direct_vreduce_v1": ("direct",),
    "astrominn_task17_random_l0_v1": ("random", "entropy"),
    "astrominn_task18_anchoral_v1": ("anchoral",),
}
SEEDS = (135, 246, 357)
METRICS = ("f1_macro", "balanced_accuracy")
CSV_FIELDS = ("suite", "strategy", "seed", "cycle", "split", "metric", "value", "n_labeled", "n_acquired_total")


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 18), b""):
            h.update(chunk)
    return h.hexdigest()


def gather_run(root: Path, suite: str, strategy: str, seed: int, *, expected_cycles=9):
    location = root / "runs" / suite / strategy / f"seed_{seed}"
    state_file, history_file = location / "run_state.json", location / "evaluation_history.json"
    label = f"{suite}/{strategy}/seed_{seed}"
    state = load_json(state_file)
    if state.get("last_completed_cycle") != expected_cycles or state.get("active_cycle") is not None:
        raise ValueError(f"{label}: not completed {expected_cycles} cycles: {state.get('last_completed_cycle')}, active={state.get('active_cycle')}")
    history = load_json(history_file)
    if not isinstance(history, list):
        raise ValueError(f"{label}: expected a list of evaluation records")
    by_cycle = {}
    for record in history:
        if not isinstance(record, dict) or record.get("split") != "validation":
            continue
        cycle = record.get("cycle")
        if not isinstance(cycle, int) or cycle not in range(expected_cycles + 1):
            raise ValueError(f"{label}: invalid validation cycle {cycle!r}")
        if cycle in by_cycle:
            raise ValueError(f"{label}: duplicate validation cycle {cycle}")
        by_cycle[cycle] = record
    expected = set(range(expected_cycles + 1))
    if set(by_cycle) != expected:
        raise ValueError(f"{label}: missing validation cycles {sorted(expected - set(by_cycle))}")
    n0 = by_cycle[0].get("n_labeled")
    if not isinstance(n0, int) or n0 <= 0:
        raise ValueError(f"{label}: invalid initial labeled count {n0!r}")
    last_n = n0
    result = []
    for cycle in range(expected_cycles + 1):
        record = by_cycle[cycle]
        n = record.get("n_labeled")
        if not isinstance(n, int) or n < last_n:
            raise ValueError(f"{label}: invalid/nonmonotonic n_labeled at cycle {cycle}: {n!r}")
        last_n = n
        recorded_strategy = record.get("strategy")
        if recorded_strategy is not None and recorded_strategy != strategy:
            raise ValueError(f"{label}: inconsistent history strategy {recorded_strategy!r}")
        recorded_seed = record.get("seed")
        if recorded_seed is not None and int(recorded_seed) != seed:
            raise ValueError(f"{label}: inconsistent history seed {recorded_seed!r}")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{label}: invalid metrics at cycle {cycle}")
        for metric in METRICS:
            if metric not in metrics:
                raise ValueError(f"{label}: missing {metric} at cycle {cycle}; found {sorted(metrics)}")
            value = float(metrics[metric])
            if not math.isfinite(value) or not (0 <= value <= 1):
                raise ValueError(f"{label}: invalid {metric}={value} at cycle {cycle}")
            result.append({"suite": suite, "strategy": strategy, "seed": seed, "cycle": cycle,
                           "split": "validation", "metric": metric, "value": value,
                           "n_labeled": n, "n_acquired_total": n - n0})
    return result, {"suite": suite, "strategy": strategy, "seed": seed,
                    "completed_cycles": expected_cycles, "initial_labeled": n0,
                    "final_labeled": last_n, "history_sha256": file_hash(history_file)}


def csv_text(rows, fields):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def collect(root: Path, suites=DEFAULT_SUITES, seeds=SEEDS, *, expected_cycles=9):
    rows, run_list = [], []
    for suite, strategies in suites.items():
        for policy in strategies:
            for seed in seeds:
                records, info = gather_run(root, suite, policy, seed, expected_cycles=expected_cycles)
                rows.extend(records)
                run_list.append(info)
    baseline = defaultdict(list)
    for row in rows:
        if row["cycle"] == 0 and row["metric"] == "f1_macro":
            baseline[row["seed"]].append({"suite": row["suite"], "strategy": row["strategy"],
                                          "f1_macro": row["value"], "n_labeled": row["n_labeled"]})
    audit = {str(seed): {
        "distinct_initial_labeled_counts": sorted(set(v["n_labeled"] for v in values)),
        "min_baseline_macro_f1": min(v["f1_macro"] for v in values),
        "max_baseline_macro_f1": max(v["f1_macro"] for v in values),
        "baseline_spread": max(v["f1_macro"] for v in values) - min(v["f1_macro"] for v in values),
        "per_policy": values,
    } for seed, values in baseline.items()}
    provenance = {"format": "astro_recent_validation_review_v1",
                  "status": "provisional_not_a_comparability_or_redistribution_audit",
                  "metrics": list(METRICS), "split": "validation",
                  "expected_cycles": expected_cycles,
                  "seeds": list(seeds), "suites": {k: list(v) for k, v in suites.items()},
                  "run_count": len(run_list), "record_count": len(rows),
                  "run_histories": run_list,
                  "baseline_audit": audit,
                  "warning": "Suite and policy labels remain separate. Matching cycle-zero scores is necessary but insufficient to establish experimental comparability. Review L0, data split, training objective, checkpoint selection and budgets before pooling curves.",
                  "excluded": ["test metrics", "test labels", "candidate IDs", "revealed object IDs",
                               "raw training samples", "checkpoints", "private paths"]}
    return rows, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--persistent-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="New ZIP file; never automatically writes into examples/data")
    p.add_argument("--suite", action="append", choices=tuple(DEFAULT_SUITES),
                   help="Optional specific suite; repeat for several")
    args = p.parse_args()
    suites = ({name: DEFAULT_SUITES[name] for name in args.suite} if args.suite else DEFAULT_SUITES)
    rows, provenance = collect(args.persistent_root.expanduser(), suites)
    out = args.output.expanduser()
    if out.exists():
        p.error(f"Refusing to overwrite existing review archive: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out, "x", ZIP_DEFLATED) as z:
        z.writestr("astro_recent_validation.csv", csv_text(rows, CSV_FIELDS))
        z.writestr("astro_recent_protocol_audit.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"archive": str(out), "run_count": provenance["run_count"],
                      "validation_records": len(rows),
                      "cycle_zero_f1_spreads_by_seed": {seed: round(a["baseline_spread"], 6)
                                                       for seed, a in provenance["baseline_audit"].items()},
                      "initial_labeled_counts_by_seed": {seed: a["distinct_initial_labeled_counts"]
                                                        for seed, a in provenance["baseline_audit"].items()},
                      "note": "This is a review export; do not publish or mix suites without checking protocols."}, indent=2))


if __name__ == "__main__":
    main()