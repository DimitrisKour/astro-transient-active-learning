#!/usr/bin/env python
"""Summarize early-budget curves and the full-data reference, without tuning."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def curve_summary(records, total_objects, *, metric, early_fraction=0.5, gain_fraction=0.95):
    """No extrapolation, and no full-reference claim for incomplete curves."""
    records = sorted(records, key=lambda r: r["cycle"])
    x = np.asarray([r["n_labeled"] / total_objects for r in records])
    y = np.asarray([r["metrics"][metric] for r in records])
    if len(set(x.tolist())) != len(x) or np.any(np.diff(x) <= 0):
        raise ValueError("Curve budgets must be unique and strictly increasing.")
    limit = min(early_fraction, float(x[-1]))
    mask = x <= limit
    xx, yy = x[mask], y[mask]
    if len(xx) and xx[-1] < limit:
        xx = np.append(xx, limit)
        yy = np.append(yy, np.interp(limit, x, y))
    area = float(np.trapezoid(yy, xx) if hasattr(np, "trapezoid") else np.trapz(yy, xx))
    width = limit - float(x[0])
    full = records[-1]["n_labeled"] == total_objects
    threshold = first = stable = None
    if full and y[-1] > y[0]:
        threshold = float(y[0] + gain_fraction * (y[-1] - y[0]))
        crossing = np.flatnonzero(y >= threshold)
        first = int(records[int(crossing[0])]["n_labeled"])
        persistent = [i for i in crossing if np.all(y[i:] >= threshold)]
        stable = int(records[int(persistent[0])]["n_labeled"])
    return {
        "last_cycle": records[-1]["cycle"],
        "last_n_labeled": records[-1]["n_labeled"],
        "baseline_metric": float(y[0]),
        "last_metric": float(y[-1]),
        "full_data_reference_available": full,
        "early_fraction_limit_observed": limit,
        "early_auc_normalized": area / width if width > 0 else None,
        "gain_fraction": gain_fraction,
        "own_full_reference_threshold": threshold,
        "first_n_labeled_at_threshold": first,
        "persistent_n_labeled_at_threshold": stable,
    }


def canonical_manifest_hash(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = sorted(reader.fieldnames)
        rows = sorted(tuple(row[field] for field in fields) for row in reader)
    return hashlib.sha256(json.dumps([fields, rows]).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--metric", default="f1_macro")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    total = json.loads(
        (args.suite_root / "shared/historical_replay_split/split_summary.json").read_text()
    )["historical_objects"]
    summaries = []
    curves = []
    for path in sorted(args.suite_root.glob("*/seed_*/evaluation_history.json")):
        records = [r for r in json.loads(path.read_text()) if r["split"] == args.split]
        if not records:
            continue
        summary = curve_summary(records, total, metric=args.metric)
        summary.update(strategy=records[0]["strategy"], seed=records[0]["seed"])
        if summary["full_data_reference_available"]:
            manifest = path.parent / "manifests" / f"train_cycle_{summary['last_cycle']:03d}.csv"
            summary["full_manifest_content_sha256"] = canonical_manifest_hash(manifest)
        summaries.append(summary)
        for r in records:
            curves.append(
                {
                    "strategy": r["strategy"],
                    "seed": r["seed"],
                    "cycle": r["cycle"],
                    "n_labeled": r["n_labeled"],
                    "labeled_fraction": r["n_labeled"] / total,
                    "metric": args.metric,
                    "value": r["metrics"][args.metric],
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "metric": args.metric,
        "total_objects": total,
        "runs": summaries,
        "notes": [
            "Compare early AUC only over the same observed budget interval.",
            "Thresholds use each run's own full-data reference, not a theoretical optimum.",
            "A single first crossing can be transient; persistent crossing is also reported.",
            "No threshold is reported when the full-data metric does not exceed baseline.",
            "Equal final manifest hashes ignore row order; training order may still differ.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "seed",
                "cycle",
                "n_labeled",
                "labeled_fraction",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(curves)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
