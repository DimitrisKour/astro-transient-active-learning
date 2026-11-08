#!/usr/bin/env python3
"""Extract a tiny, non-sensitive summary of completed AstroMiNN AL runs.

Input is the existing reporting.py CSV output, not raw images/checkpoints.
Supply a vetted report directory containing evaluation_records.csv and,
optionally, acquisition_records.csv, selection_overlap_records.csv, and
per_class_records.csv. No fake/incomplete trajectories are constructed.

Example:
  python scripts/examples/export_astro_school_replay.py \
      --report-directory /path/to/verified/multiseed/report \
      --output-directory examples/data

The resulting files still require a research/publication review before commit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

TABLES = {
    "evaluation_records.csv": ("astro_historical_evaluation.csv", (
        "suite", "initialization", "strategy", "seed", "cycle", "split", "metric", "value",
        "n_labeled", "n_acquired_total")),
    "acquisition_records.csv": ("astro_historical_acquisition.csv", (
        "suite", "initialization", "strategy", "seed", "cycle", "label", "count", "cumulative_count")),
    "selection_overlap_records.csv": ("astro_historical_overlap.csv", (
        "suite", "initialization", "seed", "cycle", "strategy_a", "strategy_b", "jaccard")),
    "per_class_records.csv": ("astro_historical_per_class.csv", (
        "suite", "initialization", "strategy", "seed", "cycle", "split", "level", "class_name",
        "metric", "value", "support")),
}


def read_table(path, required):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def export_reports(report_directory: Path, output_directory: Path, *, overwrite=False):
    report_directory = report_directory.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    required = report_directory / "evaluation_records.csv"
    if not required.exists():
        raise FileNotFoundError(f"No evaluation_records.csv in {report_directory}")
    outputs = {}
    # First read and validate everything; do not create half-written exports.
    for source_name, (target_name, columns) in TABLES.items():
        src = report_directory / source_name
        if not src.exists():
            if source_name == "evaluation_records.csv":
                raise FileNotFoundError(src)
            continue
        rows = read_table(src, columns)
        if source_name == "evaluation_records.csv":
            rows = [r for r in rows if r["split"] == "validation" and
                    r["metric"] in {"f1_macro", "balanced_accuracy"}]
            if not rows:
                raise ValueError("No validation macro-F1/balanced-accuracy records")
        elif source_name == "per_class_records.csv":
            rows = [r for r in rows if r["split"] == "validation" and
                    r["level"] == "object" and r["metric"] in {"recall", "f1"}]
        outputs[target_name] = (columns, rows, src)
    for target in outputs:
        dest = output_directory / target
        if dest.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {dest}")
    output_directory.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "source": "historical AstroMiNN multi-seed report CSVs",
               "tables": {}}
    for target, (columns, rows, src) in outputs.items():
        dest = output_directory / target
        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({col: row[col] for col in columns} for row in rows)
        summary["tables"][target] = {
            "rows": len(rows), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "size_bytes": dest.stat().st_size,
        }
    provenance = output_directory / "astro_historical_provenance.json"
    provenance.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Exported small CSVs; audit completeness and release rights before committing.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-directory", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    export_reports(args.report_directory, args.output_directory, overwrite=args.overwrite)


if __name__ == "__main__":
    main()