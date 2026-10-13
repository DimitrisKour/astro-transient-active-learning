#!/usr/bin/env python3
"""Prove Task 4 reuses the original historical object split and bootstraps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalized_csv(path: Path, *, ignored: frozenset[str] = frozenset()) -> list[tuple[tuple[str, str], ...]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return sorted(tuple(sorted((k, v) for k, v in row.items() if k not in ignored))
                      for row in reader)


def check_equivalent_splits(root: Path, base_suite: str, new_suite: str) -> None:
    base = root / "runs" / base_suite / "shared" / "historical_replay_split"
    new = root / "runs" / new_suite / "shared" / "historical_replay_split"
    if base == new:
        raise ValueError("Base and new replay must use distinct suites")
    left = json.loads((base / "split_summary.json").read_text())
    right = json.loads((new / "split_summary.json").read_text())
    keys = [
        "split_seed", "initial_fraction", "minimum_initial_per_class",
        "historical_objects", "historical_samples", "initial_objects", "initial_samples",
        "candidate_objects", "candidate_samples", "maximum_alert",
        "historical_object_label_counts", "initial_object_label_counts",
        "candidate_object_label_counts", "initial_sample_label_counts",
        "candidate_sample_label_counts", "overlap_checks",
    ]
    for key in keys:
        if left.get(key) != right.get(key):
            raise ValueError(f"Original and Task 4 split differ at {key}: {left.get(key)} vs {right.get(key)}")
    for name in ("initial_labeled_manifest.csv", "hidden_labels.csv"):
        if normalized_csv(base / name) != normalized_csv(new / name):
            raise ValueError(f"Task4 split mismatch: {name}")
    relative = Path("candidate_pool") / "candidate_pool.csv"
    if normalized_csv(base / relative, ignored=frozenset({"pool_id"})) != normalized_csv(
        new / relative, ignored=frozenset({"pool_id"})
    ):
        raise ValueError("Task 4 candidate pool differs from original replay")
    print("L0, hidden labels, and full U0 candidate rows match the original replay")


def check_bootstraps(data: Path, base_suite: str, seeds: tuple[int, ...]) -> None:
    for seed in seeds:
        folder = data / "runs" / base_suite / "shared" / f"seed_{seed}" / "bootstrap_training"
        if not (folder / "best_vit_tower.pth").is_file():
            raise FileNotFoundError(f"Missing baseline L0 checkpoint: {folder}")
        if not (folder / "training_summary.json").is_file():
            raise FileNotFoundError(f"Missing baseline L0 summary: {folder}")
    print("All original seed-specific L0 checkpoints and training summaries exist")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--persistent", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base-suite", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[135, 246, 357])
    args = p.parse_args()
    check_equivalent_splits(args.persistent, args.base_suite, args.suite)
    check_bootstraps(args.data, args.base_suite, tuple(args.seeds))


if __name__ == "__main__":
    main()
