#!/usr/bin/env python
"""Preflight or run the BOOM/TNS simulated-label AstroMiNN experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
from al_pipeline.astronomy.experiments.simulated import (
    load_hidden_labels_csv,
    stratified_holdout_ids,
)
from al_pipeline.astronomy.experiments.astrominn import build_astrominn_simulated_suite
from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a resumable AstroMiNN active-learning suite. Without "
            "--run, only validate and print the plan."
        )
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--hidden-labels", type=Path, required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=("random", "margin", "entropy"),
        choices=("random", "margin", "entropy", "fisher"),
        help=(
            "Acquisition strategies to compare. Fisher is available but "
            "is intentionally not part of the default three-strategy pilot."
        ),
    )
    parser.add_argument("--acquisition-size", type=int, default=50)
    parser.add_argument("--max-cycles", type=int, default=6)
    parser.add_argument("--seed", type=int, default=135)
    parser.add_argument(
        "--training-initialization",
        choices=(
            "warm_start",
            "restart_from_baseline",
            "restart_from_random",
        ),
        default="warm_start",
    )
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=2_048,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--new-object-weighting",
        choices=("per_sample", "equal_object"),
        default="per_sample",
        help=(
            "Use equal_object to give every newly acquired object the "
            "same total loss weight despite different alert-row counts."
        ),
    )
    parser.add_argument("--train-workers", type=int, default=0)
    parser.add_argument("--validation-workers", type=int, default=0)
    parser.add_argument("--inference-workers", type=int, default=0)
    parser.add_argument("--evaluation-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.0)
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=135,
        help=("Seed used only for the fixed holdout. Keep this constant across training seeds."),
    )
    parser.add_argument("--allow-partial-final-cycle", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start GPU inference/training after a successful preflight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_hidden_labels_csv(args.hidden_labels)
    manifest = CandidatePoolManifest.from_csv(args.pool / "candidate_pool.csv")
    unknown_labels = set(manifest.object_ids).difference(labels)

    if unknown_labels:
        raise ValueError(
            f"Candidate pool objects are missing hidden labels: {sorted(unknown_labels)[:10]}."
        )

    pool_labels = {object_id: labels[object_id] for object_id in manifest.object_ids}
    holdout_ids = (
        stratified_holdout_ids(
            pool_labels,
            fraction=args.holdout_fraction,
            seed=args.holdout_seed,
        )
        if args.holdout_fraction > 0
        else ()
    )
    policy = AstroMiNNViTTrainingConfig(
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        train_num_workers=args.train_workers,
        validation_num_workers=args.validation_workers,
        augment_training=True,
        augment_validation=False,
        legacy_frozen_expert_train_mode=False,
        legacy_seed_sequence=False,
        new_object_weighting=args.new_object_weighting,
    )
    suite = build_astrominn_simulated_suite(
        suite_id=args.suite_id,
        registry_path=args.registry,
        pool_path=args.pool,
        hidden_labels_path=args.hidden_labels,
        training_policy=policy,
        strategies=args.strategies,
        acquisition_size=args.acquisition_size,
        max_cycles=args.max_cycles,
        seed=args.seed,
        training_initialization=args.training_initialization,
        allow_partial_final_cycle=args.allow_partial_final_cycle,
        holdout_ids=holdout_ids,
        device=args.device,
        inference_num_workers=args.inference_workers,
        evaluation_num_workers=args.evaluation_workers,
    )
    acquisition_count = len(manifest.object_ids) - len(holdout_ids)
    holdout_set = set(holdout_ids)
    holdout_label_counts = Counter(pool_labels[object_id] for object_id in holdout_ids)
    acquisition_label_counts = Counter(
        label for object_id, label in pool_labels.items() if object_id not in holdout_set
    )
    plan = {
        "suite_id": args.suite_id,
        "strategies": list(args.strategies),
        "pool_id": manifest.pool_id,
        "candidate_objects": len(manifest.object_ids),
        "candidate_samples": len(manifest.records),
        "samples_per_object": len(manifest.records) / len(manifest.object_ids),
        "target_label_counts": dict(sorted(Counter(map(str, pool_labels.values())).items())),
        "holdout_objects": len(holdout_ids),
        "holdout_seed": args.holdout_seed,
        "holdout_label_counts": dict(sorted(holdout_label_counts.items())),
        "acquisition_objects": acquisition_count,
        "acquisition_label_counts": dict(sorted(acquisition_label_counts.items())),
        "acquisition_size": args.acquisition_size,
        "max_cycles": args.max_cycles,
        "maximum_requested_objects": (args.acquisition_size * args.max_cycles),
        "training_initialization": args.training_initialization,
        "training_policy": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "validation_batch_size": args.validation_batch_size,
            "learning_rate": args.learning_rate,
            "new_object_weighting": args.new_object_weighting,
            "train_workers": args.train_workers,
            "validation_workers": args.validation_workers,
        },
        "runs": {
            strategy: {
                "state_dir": str(experiment.run_dir),
                "artifact_dir": str(experiment.artifact_dir),
            }
            for strategy, experiment in suite.experiments.items()
        },
        "will_run": bool(args.run),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    if not args.run:
        print("\nPRE-FLIGHT ONLY: add --run after the printed policy has been reviewed.")
        return

    results = suite.run()
    print(
        json.dumps(
            {strategy: [result.cycle for result in cycles] for strategy, cycles in results.items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
