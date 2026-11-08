#!/usr/bin/env python3
"""Prepare and audit small, CPU-only astronomy teaching exports.

This is an INSTRUCTOR preparation tool. It reads existing local manifests
through the registry and never modifies the historical sources.

Example:
    python scripts/examples/prepare_astro_school_exports.py \
      --registry /path/to/research-checkout/configs/delta.toml \
      --output-directory examples/data \
      --report-root /path/to/research/reports

Part A exports image/metadata summaries, NOT neural embeddings or checkpoints.
Part B is exported only when --report-directory explicitly identifies a
reviewed reporting directory. --report-root merely discovers candidates.

Outputs are provisional. Inspect them and confirm redistribution rights
before adding any generated file to Git.
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from export_astro_school_features import export as export_features
from export_astro_school_replay import export_reports


def read_dataset_registry(path: Path, name: str) -> dict[str, object]:
    """Load only the required dataset fields from a local TOML registry."""
    if not path.is_file():
        raise FileNotFoundError(f"Registry not found: {path}")
    with path.open("rb") as stream:
        registry = tomllib.load(stream)
    datasets = registry.get("datasets", {})
    if name not in datasets:
        raise KeyError(f"Dataset {name!r} not found; available: {sorted(datasets)}")
    record = datasets[name]
    keys = ("train_manifest", "validation_manifest", "test_manifest", "data_root",
            "object_id_column", "sample_file_column", "label_column")
    missing = [key for key in keys if not record.get(key)]
    if missing:
        raise ValueError(f"Dataset {name!r} missing required keys: {missing}")
    for key in ("train_manifest", "validation_manifest", "test_manifest", "data_root"):
        if not Path(record[key]).exists():
            raise FileNotFoundError(f"Configured {key} is inaccessible: {record[key]}")
    return record


def find_reports(root: Path) -> list[Path]:
    """List possible report directories without selecting one automatically."""
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.rglob("evaluation_records.csv") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--dataset", default="astrominn_ztf")
    parser.add_argument("--output-directory", type=Path, default=Path("examples/data"))
    parser.add_argument("--train-objects", type=int, default=700)
    parser.add_argument("--validation-objects", type=int, default=200)
    parser.add_argument("--test-objects", type=int, default=200)
    parser.add_argument("--seed", type=int, default=135)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--report-directory", type=Path,
                        help="Reviewed report containing evaluation_records.csv")
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()

    out = args.output_directory.expanduser().resolve()
    if not args.skip_features:
        spec = read_dataset_registry(args.registry.expanduser(), args.dataset)
        export_features(argparse.Namespace(
            train_manifest=Path(spec["train_manifest"]),
            validation_manifest=Path(spec["validation_manifest"]),
            test_manifest=Path(spec["test_manifest"]),
            data_root=Path(spec["data_root"]),
            object_id_column=str(spec["object_id_column"]),
            file_column=str(spec["sample_file_column"]),
            label_column=str(spec["label_column"]),
            train_objects=args.train_objects,
            validation_objects=args.validation_objects,
            test_objects=args.test_objects,
            seed=args.seed,
            output=out / "astro_real_v1.npz",
            overwrite=False,
        ))
    if args.report_root is not None:
        candidates = find_reports(args.report_root.expanduser())
        print("\nHistorical report candidates (review suite/seeds before selection):")
        for p in candidates[:40]:
            print("  ", p)
        if len(candidates) > 40:
            print(f"  ... {len(candidates)-40} additional directories")
        if not candidates:
            print("  None found. You may need to create a report with write_report_tables.")
    if args.report_directory is not None:
        print("\nExporting the explicitly selected archived report:")
        export_reports(args.report_directory.expanduser(), out)
    else:
        print("\nPart B not exported: pass --report-directory after reviewing a candidate.")
    print("\nFiles in", out, "are for review; do not commit without checking provenance and redistribution rights.")


if __name__ == "__main__":
    main()