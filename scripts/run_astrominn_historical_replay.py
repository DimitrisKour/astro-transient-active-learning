#!/usr/bin/env python
"""Prepare or run Experiment B: historical AstroMiNN AL replay."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.experiments.astrominn import build_astrominn_simulated_suite
from al_pipeline.astronomy.experiments.historical_replay import (
    materialize_astrominn_historical_replay_split,
)
from al_pipeline.astronomy.astrominn.inference.astrominn_vit import DEFAULT_LEGACY_CODE_DIR
from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainer, AstroMiNNViTTrainingConfig
from al_pipeline.core.training.base import TrainingRequest
from al_pipeline.core.experiments.batch_policy import BatchAcquisitionConfig
from al_pipeline.core.experiments.budget import plan_cycle_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic historical L0/U0 replay and optionally "
            "run seed-specific random-ViT bootstrap and AL cycles."
        )
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--profile", default="astrominn_ztf")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=("random", "margin", "entropy"),
        choices=(
            "random",
            "margin",
            "entropy",
            "fisher",
            "entropy_diversity",
            "entropy_balanced_explore",
        ),
    )
    parser.add_argument("--seed", type=int, default=135)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=135,
        help="Fixed data-split seed; keep unchanged across model seeds.",
    )
    parser.add_argument("--initial-fraction", type=float, default=0.10)
    parser.add_argument(
        "--minimum-initial-per-class",
        type=int,
        default=5,
    )
    acquisition_group = parser.add_mutually_exclusive_group()
    acquisition_group.add_argument("--acquisition-size", type=int)
    acquisition_group.add_argument("--acquisition-fraction", type=float)
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--cycle", type=int, help="Run exactly this one round across strategies.")
    parser.add_argument("--allow-partial-final-cycle", action="store_true")
    parser.add_argument(
        "--complete-pool",
        action="store_true",
        help="Compute enough rounds to acquire all U0, including the remainder.",
    )
    parser.add_argument("--shortlist-multiplier", type=float, default=3.0)
    parser.add_argument("--balance-alpha", type=float, default=1.0)
    parser.add_argument("--balance-beta", type=float, default=0.5)
    parser.add_argument("--balance-strength", type=float, default=1.0)
    parser.add_argument("--balance-cap", type=float, default=3.0)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-workers", type=int, default=0)
    parser.add_argument("--validation-workers", type=int, default=0)
    parser.add_argument("--inference-workers", type=int, default=0)
    parser.add_argument("--evaluation-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--assert-excluded-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Object manifest used only to prove zero overlap with external "
            "labelled data. May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="With --run, train L0 and stop before any acquisition.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Train the L0 bootstrap if needed and run/resume AL cycles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_only and not args.run:
        raise ValueError("--bootstrap-only requires --run.")
    if args.cycle is not None and (not args.run or args.bootstrap_only or args.cycle < 1):
        raise ValueError("--cycle requires --run, a positive cycle, and no --bootstrap-only.")
    batch_policy = BatchAcquisitionConfig(
        shortlist_multiplier=args.shortlist_multiplier,
        balance_alpha=args.balance_alpha,
        balance_beta=args.balance_beta,
        balance_strength=args.balance_strength,
        balance_cap=args.balance_cap,
        exploration_fraction=args.exploration_fraction,
    )

    registry = DatasetRegistry.from_toml(args.registry)
    resolved = registry.resolve_profile(args.profile)
    registry.validate_paths(
        resolved.dataset.name,
        require_historical_splits=True,
    )
    dataset = resolved.dataset
    profile = resolved.profile

    configured_checkpoint = profile.options.get("checkpoint")
    if configured_checkpoint is None:
        raise ValueError(
            "The AstroMiNN profile must define its published checkpoint so "
            "the frozen expert-tower artifact directory can be located."
        )
    published_checkpoint = Path(configured_checkpoint)
    if not published_checkpoint.exists():
        raise FileNotFoundError(published_checkpoint)
    model_artifact_dir = published_checkpoint.parent

    split_dir = (
        resolved.storage.persistent_root
        / "runs"
        / args.suite_id
        / "shared"
        / "historical_replay_split"
    )
    replay = materialize_astrominn_historical_replay_split(
        dataset=dataset,
        destination=split_dir,
        initial_fraction=args.initial_fraction,
        minimum_initial_per_class=args.minimum_initial_per_class,
        split_seed=args.split_seed,
        pool_id=f"{args.suite_id}_historical_u0",
        excluded_object_manifests=args.assert_excluded_manifest,
    )
    replay_dataset = replace(
        dataset,
        train_manifest=replay.initial_manifest,
    )
    candidate_manifest = CandidatePoolManifest.from_csv(
        replay.candidate_pool_dir / "candidate_pool.csv"
    )
    missing_candidate_files = candidate_manifest.missing_files()
    if missing_candidate_files:
        raise FileNotFoundError(
            "Historical replay candidate pool references missing samples: "
            + ", ".join(missing_candidate_files[:10])
        )
    bootstrap_dir = (
        resolved.storage.data_root
        / "runs"
        / args.suite_id
        / "shared"
        / f"seed_{args.seed}"
        / "bootstrap_training"
    )
    bootstrap_checkpoint = bootstrap_dir / "best_vit_tower.pth"

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
        loader_seed=args.seed,
        new_object_weighting="per_sample",
        class_weight_source="training_manifest",
    )

    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))
    historical_objects = int(split_summary["historical_objects"])
    candidate_objects = len(candidate_manifest.object_ids)
    acquisition_fraction = (
        0.10
        if args.acquisition_fraction is None and args.acquisition_size is None
        else args.acquisition_fraction
    )
    if args.acquisition_size is not None:
        if args.acquisition_size <= 0:
            raise ValueError("--acquisition-size must be positive.")
        acquisition_size = args.acquisition_size
        acquisition_rule = "explicit_object_count"
    else:
        if acquisition_fraction is None or not 0 < acquisition_fraction < 1:
            raise ValueError("--acquisition-fraction must be between zero and one.")
        acquisition_size = math.ceil(historical_objects * acquisition_fraction)
        acquisition_rule = "fraction_of_historical_training_objects"

    if args.max_cycles <= 0:
        raise ValueError("--max-cycles must be positive.")
    budget_plan = plan_cycle_budget(
        candidate_objects=candidate_objects,
        acquisition_size=acquisition_size,
        max_cycles=args.max_cycles,
        allow_partial_final_cycle=args.allow_partial_final_cycle,
        complete_pool=args.complete_pool,
    )
    args.max_cycles = budget_plan.max_cycles
    args.allow_partial_final_cycle = budget_plan.allow_partial_final_cycle
    maximum_requested_objects = budget_plan.total_acquisitions
    if args.cycle is not None and args.cycle > args.max_cycles:
        raise ValueError("--cycle exceeds the planned number of rounds.")

    excluded_audits = split_summary["excluded_manifest_audits"]
    final_labeled_objects = int(split_summary["initial_objects"]) + maximum_requested_objects
    plan = {
        "experiment": "B_historical_active_learning_replay",
        "suite_id": args.suite_id,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "strategies": list(args.strategies),
        "initial_fraction_requested": args.initial_fraction,
        "minimum_initial_per_class": args.minimum_initial_per_class,
        "initial_objects": split_summary["initial_objects"],
        "initial_samples": split_summary["initial_samples"],
        "initial_fraction_actual": (int(split_summary["initial_objects"]) / historical_objects),
        "initial_object_label_counts": split_summary["initial_object_label_counts"],
        "candidate_objects": candidate_objects,
        "candidate_samples": len(candidate_manifest.records),
        "missing_candidate_files": 0,
        "candidate_object_label_counts": split_summary["candidate_object_label_counts"],
        "acquisition_rule": acquisition_rule,
        "acquisition_fraction_requested": acquisition_fraction,
        "acquisition_size": acquisition_size,
        "max_cycles": args.max_cycles,
        "allow_partial_final_cycle": args.allow_partial_final_cycle,
        "requested_single_cycle": args.cycle,
        "final_cycle_size": budget_plan.final_cycle_size,
        "batch_acquisition_policy": {
            field: getattr(batch_policy, field) for field in batch_policy.__dataclass_fields__
        },
        "maximum_requested_objects": maximum_requested_objects,
        "final_labeled_objects_planned": final_labeled_objects,
        "final_labeled_fraction_planned": (final_labeled_objects / historical_objects),
        "validation_manifest_unchanged": str(dataset.validation_manifest),
        "test_manifest_unchanged": str(dataset.test_manifest),
        "initial_manifest": str(replay.initial_manifest),
        "candidate_pool": str(replay.candidate_pool_dir),
        "hidden_labels": str(replay.hidden_labels),
        "split_summary": str(replay.summary_path),
        "bootstrap_checkpoint": str(bootstrap_checkpoint),
        "bootstrap_complete": bootstrap_checkpoint.exists(),
        "published_vit_checkpoint_loaded": False,
        "frozen_expert_artifact_dir": str(model_artifact_dir),
        "historical_training_source_only": True,
        "external_appended_samples_loaded": False,
        "external_exclusion_audits": excluded_audits,
        "cycle_training_initialization": "restart_from_random",
        "expert_towers": "published_pretrained_frozen",
        "vit_tower": "random_initialization_each_training_stage",
        "class_weight_source": "current_L_t_training_manifest_only",
        "sample_weighting": "per_sample_historical_compatibility",
        "training_policy": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "validation_batch_size": args.validation_batch_size,
            "learning_rate": args.learning_rate,
            "loader_seed": args.seed,
            "train_workers": args.train_workers,
            "validation_workers": args.validation_workers,
        },
        "will_run": bool(args.run),
        "bootstrap_only": bool(args.bootstrap_only),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    if not args.run:
        print(
            "\nPRE-FLIGHT ONLY: manifests were materialized and audited; "
            "add --run --bootstrap-only for the first small training check."
        )
        return

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    bootstrap_trainer = AstroMiNNViTTrainer(
        dataset=replay_dataset,
        policy=policy,
        device=device,
        initialization="random_vit",
        legacy_code_dir=DEFAULT_LEGACY_CODE_DIR,
        artifact_dir=model_artifact_dir,
    )
    bootstrap_request = TrainingRequest(
        train_manifest=replay.initial_manifest,
        validation_manifest=dataset.validation_manifest,
        output_dir=bootstrap_dir,
        run_id=f"{args.suite_id}_bootstrap_seed{args.seed}",
        cycle=0,
        strategy="bootstrap",
        seed=args.seed,
        initial_checkpoint=None,
    )
    bootstrap = bootstrap_trainer.completed_result(bootstrap_request)
    if bootstrap is None:
        bootstrap = bootstrap_trainer.train(bootstrap_request)

    print(
        json.dumps(
            {
                "bootstrap_checkpoint": str(bootstrap.checkpoint_path),
                "bootstrap_best_epoch": bootstrap.best_epoch,
                "bootstrap_epochs_completed": bootstrap.epochs_completed,
                "bootstrap_best_validation_loss": (bootstrap.best_validation_loss),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.bootstrap_only:
        return

    suite = build_astrominn_simulated_suite(
        suite_id=args.suite_id,
        registry_path=args.registry,
        pool_path=replay.candidate_pool_dir,
        hidden_labels_path=replay.hidden_labels,
        training_policy=policy,
        profile_name=args.profile,
        strategies=args.strategies,
        acquisition_size=acquisition_size,
        max_cycles=args.max_cycles,
        allow_partial_final_cycle=args.allow_partial_final_cycle,
        batch_policy=batch_policy,
        seed=args.seed,
        training_initialization="restart_from_random",
        holdout_ids=(),
        device=device,
        inference_num_workers=args.inference_workers,
        evaluation_num_workers=args.evaluation_workers,
        baseline_checkpoint=bootstrap.checkpoint_path,
        training_manifest_override=replay.initial_manifest,
        artifact_dir=model_artifact_dir,
    )
    results = suite.run() if args.cycle is None else suite.run_cycle(args.cycle)
    print(
        json.dumps(
            {strategy: [result.cycle for result in cycles] for strategy, cycles in results.items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
