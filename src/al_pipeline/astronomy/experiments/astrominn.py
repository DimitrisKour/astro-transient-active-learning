"""High-level construction of AstroMiNN simulated-label AL runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence, cast

import torch

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.astrominn.evaluation import AstroMiNNViTEvaluator
from al_pipeline.astronomy.experiments.simulated import (
    CycleResult,
    SimulatedALConfig,
    SimulatedActiveLearningExperiment,
    StrategyName,
    TrainingInitialization,
    load_hidden_labels_csv,
    materialize_labeled_candidate_manifest,
)
from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (
    DEFAULT_LEGACY_CODE_DIR,
    AstroMiNNViTPoolPredictor,
)
from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainer, AstroMiNNViTTrainingConfig


@dataclass(frozen=True, slots=True)
class AstroMiNNSimulatedSuite:
    """Independent strategy runs sharing only immutable source data."""

    suite_id: str
    experiments: dict[str, SimulatedActiveLearningExperiment]

    def evaluate_baselines(self) -> None:
        """Materialize cycle-zero evaluation for every strategy run."""

        for experiment in self.experiments.values():
            experiment.evaluate_baseline()

    def run(self) -> dict[str, tuple[CycleResult, ...]]:
        """Run strategies sequentially so one GPU is never oversubscribed."""

        return {strategy: experiment.run() for strategy, experiment in self.experiments.items()}


def build_astrominn_simulated_suite(
    *,
    suite_id: str,
    registry_path: str | Path,
    pool_path: str | Path,
    hidden_labels_path: str | Path,
    training_policy: AstroMiNNViTTrainingConfig,
    profile_name: str = "astrominn_ztf",
    strategies: Sequence[StrategyName] = (
        "random",
        "margin",
        "entropy",
    ),
    acquisition_size: int = 50,
    max_cycles: int = 6,
    seed: int = 135,
    training_initialization: TrainingInitialization = "warm_start",
    allow_partial_final_cycle: bool = False,
    holdout_ids: Sequence[str] = (),
    device: str | torch.device | None = None,
    inference_batch_size: int = 64,
    evaluation_batch_size: int = 2_048,
    inference_num_workers: int = 0,
    evaluation_num_workers: int = 0,
    baseline_checkpoint: str | Path | None = None,
    training_manifest_override: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    legacy_code_dir: str | Path = DEFAULT_LEGACY_CODE_DIR,
) -> AstroMiNNSimulatedSuite:
    """Build isolated AstroMiNN acquisition-strategy experiments.

    The returned suite does not start GPU work. Calling ``suite.run()`` is
    the explicit launch point. ``training_policy`` is mandatory so epoch,
    patience, worker, and augmentation choices are visible in the launch
    code reviewed with the supervisors.
    """

    if not suite_id:
        raise ValueError("suite_id must be a non-empty string.")

    strategy_names = tuple(str(value) for value in strategies)

    if not strategy_names:
        raise ValueError("At least one acquisition strategy is required.")

    if len(strategy_names) != len(set(strategy_names)):
        raise ValueError("Acquisition strategies must not be duplicated.")

    unknown_strategies = set(strategy_names).difference({"random", "margin", "entropy", "fisher"})

    if unknown_strategies:
        raise ValueError(f"Unsupported acquisition strategies: {sorted(unknown_strategies)}.")

    registry = DatasetRegistry.from_toml(registry_path)
    resolved = registry.resolve_profile(profile_name)
    dataset = resolved.dataset
    profile = resolved.profile
    registry.validate_paths(
        dataset.name,
        require_historical_splits=True,
    )

    if training_manifest_override is not None:
        replay_manifest = Path(training_manifest_override)
        if not replay_manifest.exists():
            raise FileNotFoundError(replay_manifest)
        dataset = replace(
            dataset,
            train_manifest=replay_manifest,
        )

    configured_checkpoint = profile.options.get("checkpoint")
    checkpoint = (
        Path(baseline_checkpoint if baseline_checkpoint is not None else configured_checkpoint)
        if (baseline_checkpoint is not None or configured_checkpoint is not None)
        else None
    )

    if checkpoint is None:
        raise ValueError(
            "The profile must define checkpoint or baseline_checkpoint must be supplied."
        )

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model_artifact_dir = Path(artifact_dir) if artifact_dir is not None else checkpoint.parent
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    labels = load_hidden_labels_csv(hidden_labels_path)
    candidate_manifest = CandidatePoolManifest.from_csv(Path(pool_path) / "candidate_pool.csv")
    evaluation_manifests = {
        "validation": dataset.validation_manifest,
        "test": dataset.test_manifest,
    }

    if holdout_ids:
        holdout_manifest = materialize_labeled_candidate_manifest(
            candidate_manifest=candidate_manifest,
            dataset=dataset,
            labels=labels,
            object_ids=holdout_ids,
            destination=(
                resolved.storage.persistent_root
                / "runs"
                / suite_id
                / "shared"
                / "new_holdout_manifest.csv"
            ),
        )
        evaluation_manifests["new_holdout"] = holdout_manifest

    experiments: dict[str, SimulatedActiveLearningExperiment] = {}

    for strategy_value in strategy_names:
        strategy = cast(StrategyName, strategy_value)
        run_id = f"{suite_id}_{strategy}_seed{seed}"
        relative_run = Path(suite_id) / strategy / f"seed_{seed}"
        state_dir = resolved.storage.persistent_root / "runs" / relative_run
        heavy_dir = resolved.storage.data_root / "runs" / relative_run
        predictor = AstroMiNNViTPoolPredictor(
            pool_path=pool_path,
            device=resolved_device,
            batch_size=inference_batch_size,
            num_workers=inference_num_workers,
            legacy_code_dir=legacy_code_dir,
            artifact_dir=model_artifact_dir,
        )
        trainer = AstroMiNNViTTrainer(
            dataset=dataset,
            policy=training_policy,
            device=resolved_device,
            initialization=(
                "random_vit" if training_initialization == "restart_from_random" else "pretrained"
            ),
            legacy_code_dir=legacy_code_dir,
            artifact_dir=model_artifact_dir,
        )
        evaluator = AstroMiNNViTEvaluator(
            dataset=dataset,
            manifests=evaluation_manifests,
            device=resolved_device,
            batch_size=evaluation_batch_size,
            num_workers=evaluation_num_workers,
            legacy_code_dir=legacy_code_dir,
            artifact_dir=model_artifact_dir,
        )
        experiments[strategy] = SimulatedActiveLearningExperiment(
            config=SimulatedALConfig(
                run_id=run_id,
                strategy=strategy,
                acquisition_size=acquisition_size,
                max_cycles=max_cycles,
                seed=seed,
                training_initialization=training_initialization,
                allow_partial_final_cycle=allow_partial_final_cycle,
            ),
            dataset=dataset,
            pool_path=pool_path,
            hidden_labels=labels,
            run_dir=state_dir,
            artifact_dir=heavy_dir,
            baseline_checkpoint=checkpoint,
            predictor=predictor,
            trainer=trainer,
            evaluator=evaluator,
            holdout_ids=holdout_ids,
        )

    return AstroMiNNSimulatedSuite(
        suite_id=suite_id,
        experiments=experiments,
    )


__all__ = [
    "AstroMiNNSimulatedSuite",
    "build_astrominn_simulated_suite",
]
