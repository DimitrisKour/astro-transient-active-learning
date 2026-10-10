"""Resumable simulated-label active-learning experiments.

This module joins the independently tested adaptive components into the
transaction used by retrospective experiments:

    infer U_t -> acquire A_t -> reveal hidden labels -> materialize L_t
    -> train model_t -> evaluate model_t -> advance the run state

It deliberately does not know how astronomical labels were obtained. A
simulated run receives an audited hidden-label mapping and reveals only the
entries selected during each cycle.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from al_pipeline.core.scoring.entropy import EntropyScorer
from al_pipeline.core.scoring.class_balanced import ClassBalancedInput
from al_pipeline.core.batch_selection import DiversitySelector, ExplorationSelector
from al_pipeline.core.scoring.fisher import FisherScorer, FisherScoringInput
from al_pipeline.core.scoring.margin import MarginScorer
from al_pipeline.core.scoring.random import RandomScorer
from al_pipeline.core.selection import TopKSelector
from al_pipeline.astronomy.data.labeled import AcquisitionRecord, LabeledSetManager
from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
from al_pipeline.astronomy.data.registry import DatasetSpec
from al_pipeline.core.state.run_state import RunState
from al_pipeline.core.evaluation.history import EvaluationHistory
from al_pipeline.core.evaluation.base import ModelEvaluationResult, ModelEvaluator
from al_pipeline.core.oracle.existing_labels import ExistingLabelsOracle
from al_pipeline.core.experiments.config import (
    SimulatedALConfig,
    StrategyName,
    TrainingInitialization,
)
from al_pipeline.core.pool import Pool
from al_pipeline.core.training.base import TrainingBackend, TrainingRequest, TrainingResult
from al_pipeline.astronomy.astrominn.types import ClassificationInferenceResult
from al_pipeline.core.types import ClassificationPredictions, LabeledBatch, SelectionBatch


class PoolPredictor(Protocol):
    """Inference boundary required by the experiment coordinator."""

    def predict(
        self,
        *,
        object_ids: Sequence[str],
        checkpoint_path: Path,
    ) -> ClassificationInferenceResult:
        """Return one object-level prediction for every requested ID."""

        ...


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Completed artifacts for one active-learning cycle."""

    cycle: int
    selection: SelectionBatch
    labeled: LabeledBatch
    training: TrainingResult
    evaluation: ModelEvaluationResult
    cumulative_manifest: Path
    summary_path: Path


class SimulatedActiveLearningExperiment:
    """Run one isolated acquisition strategy against hidden known labels.

    Every strategy must receive its own ``run_dir`` and trainer output. The
    immutable physical candidate pool may be shared. State is persisted after
    every durable stage, so relaunching the same object resumes rather than
    selecting or training the cycle twice.
    """

    def __init__(
        self,
        *,
        config: SimulatedALConfig,
        dataset: DatasetSpec,
        pool_path: str | Path,
        hidden_labels: Mapping[str, object],
        run_dir: str | Path,
        artifact_dir: str | Path | None = None,
        baseline_checkpoint: str | Path,
        predictor: PoolPredictor,
        trainer: TrainingBackend,
        evaluator: ModelEvaluator,
        holdout_ids: Sequence[str] = (),
        class_names: Sequence[str] = (),
        label_mapping: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.pool_path = Path(pool_path)
        self.run_dir = Path(run_dir)
        self.artifact_dir = self.run_dir if artifact_dir is None else Path(artifact_dir)
        self.baseline_checkpoint = Path(baseline_checkpoint)
        self.predictor = predictor
        self.trainer = trainer
        self.evaluator = evaluator
        self.hidden_labels = {str(object_id): label for object_id, label in hidden_labels.items()}
        self.holdout_ids = tuple(str(value) for value in holdout_ids)
        self.class_names = tuple(class_names)
        self.label_mapping = dict(label_mapping or {})
        if config.strategy == "entropy_balanced_explore":
            if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
                raise ValueError("Class-balanced acquisition requires ordered unique class_names.")
        self.manifest_path = self.pool_path / "candidate_pool.csv"
        self.manifest = CandidatePoolManifest.from_csv(self.manifest_path)
        self.labeled_sets = LabeledSetManager(
            dataset=dataset,
            run_dir=self.run_dir,
        )
        self.state_path = self.run_dir / "run_state.json"
        self.history_path = self.run_dir / "evaluation_history.json"
        self.identity_path = self.run_dir / "experiment.json"

        holdout_set = set(self.holdout_ids)
        self.acquisition_ids = tuple(
            object_id for object_id in self.manifest.object_ids if object_id not in holdout_set
        )

        self._validate_static_inputs()

    @classmethod
    def from_hidden_labels_csv(
        cls,
        *,
        hidden_labels_path: str | Path,
        object_id_column: str = "object_id",
        label_column: str = "target_label",
        **kwargs: object,
    ) -> SimulatedActiveLearningExperiment:
        """Build an experiment from the notebook-12 hidden-label table."""

        hidden_labels = load_hidden_labels_csv(
            hidden_labels_path,
            object_id_column=object_id_column,
            label_column=label_column,
        )

        return cls(
            hidden_labels=hidden_labels,
            **kwargs,
        )

    def initialize(self) -> RunState:
        """Validate provenance and initialize persistent run-owned state."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.labeled_sets.initialize()
        self._validate_no_historical_overlap()
        identity = self._identity_payload()

        if self.identity_path.exists():
            observed = json.loads(self.identity_path.read_text(encoding="utf-8"))

            if observed != identity:
                raise ValueError(
                    "Existing run directory has a different experiment "
                    "identity. Use a new run_id/run_dir."
                )
        else:
            _json_dump_atomic(identity, self.identity_path)

        if self.state_path.exists():
            return RunState.load(self.state_path)

        return RunState.create(self.state_path)

    def evaluate_baseline(self) -> ModelEvaluationResult:
        """Evaluate cycle zero once using the immutable baseline checkpoint."""

        self.initialize()
        result = self._evaluate_or_resume(
            checkpoint_path=self.baseline_checkpoint,
            output_dir=self.artifact_dir / "cycle_000" / "evaluation",
        )
        self._record_evaluation(
            cycle=0,
            n_acquired=0,
            result=result,
        )
        return result

    def run(self) -> tuple[CycleResult, ...]:
        """Evaluate cycle zero and run or resume all configured cycles."""

        state = self.initialize()
        self.evaluate_baseline()
        completed: list[CycleResult] = []

        while state.last_completed_cycle < self.config.max_cycles:
            if state.active_cycle is not None:
                completed.append(self.run_next_cycle())
                state = RunState.load(self.state_path)
                continue

            available = self.available_ids()

            if not available:
                break

            if (
                len(available) < self.config.acquisition_size
                and not self.config.allow_partial_final_cycle
                and state.active_cycle is None
            ):
                break

            completed.append(self.run_next_cycle())
            state = RunState.load(self.state_path)

        return tuple(completed)

    def run_next_cycle(self) -> CycleResult:
        """Run or resume exactly one acquisition-training-evaluation cycle."""

        state = self.initialize()
        cycle = state.start_next_cycle()

        if cycle > self.config.max_cycles:
            raise RuntimeError(
                f"Configured maximum of {self.config.max_cycles} cycles has already been reached."
            )

        cycle_dir = self.run_dir / f"cycle_{cycle:03d}"
        artifact_cycle_dir = self.artifact_dir / f"cycle_{cycle:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        artifact_cycle_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cycle_dir / "cycle_summary.json"

        if summary_path.exists():
            return self._finalize_completed_cycle(
                cycle=cycle,
                state=state,
                cycle_dir=cycle_dir,
                artifact_cycle_dir=artifact_cycle_dir,
                summary_path=summary_path,
            )

        available_after_recorded = self.available_ids()
        recorded_current = self.labeled_sets.get_cycle(cycle)

        if recorded_current:
            before_set = set(available_after_recorded).union(
                record.object_id for record in recorded_current
            )
            available_ids = tuple(
                object_id for object_id in self.acquisition_ids if object_id in before_set
            )
            budget = len(recorded_current)
        else:
            available_ids = available_after_recorded
            budget = min(
                self.config.acquisition_size,
                len(available_ids),
            )

        if budget < self.config.acquisition_size:
            if not self.config.allow_partial_final_cycle:
                raise RuntimeError(
                    "Fewer candidates remain than acquisition_size and "
                    "allow_partial_final_cycle is false."
                )

        inference_checkpoint = self._inference_checkpoint(cycle)
        inference = self._infer_or_resume(
            object_ids=available_ids,
            checkpoint_path=inference_checkpoint,
            path=artifact_cycle_dir / "inference.npz",
        )
        selection = self._select_or_resume(
            inference=inference,
            budget=budget,
            cycle=cycle,
            path=cycle_dir / "selection.json",
        )
        state.mark_pending(cycle)
        labeled = self._labels_or_resume(
            selection=selection,
            path=cycle_dir / "revealed_labels.json",
        )
        state.mark_active(cycle)
        acquisition_records, training_rows = self._training_rows(
            selection=selection,
            labeled=labeled,
        )
        cumulative_manifest = self.labeled_sets.ensure_cycle(
            cycle,
            acquisitions=acquisition_records,
            training_rows=training_rows,
        )
        training_request = TrainingRequest(
            train_manifest=cumulative_manifest,
            validation_manifest=self._validation_manifest(),
            output_dir=artifact_cycle_dir / "training",
            run_id=self.config.run_id,
            cycle=cycle,
            strategy=self.config.strategy,
            seed=self.config.seed,
            initial_checkpoint=self._training_checkpoint(cycle),
        )
        training = self._train_or_resume(training_request)
        evaluation = self._evaluate_or_resume(
            checkpoint_path=training.checkpoint_path,
            output_dir=artifact_cycle_dir / "evaluation",
        )
        self._record_evaluation(
            cycle=cycle,
            n_acquired=len(selection),
            result=evaluation,
        )
        summary = {
            "run_id": self.config.run_id,
            "cycle": cycle,
            "strategy": self.config.strategy,
            "acquisition_size": len(selection),
            "selected_ids": list(selection.ids),
            "revealed_labels": {
                object_id: str(label)
                for object_id, label in zip(
                    labeled.ids,
                    labeled.labels,
                    strict=True,
                )
            },
            "available_before_selection": len(available_ids),
            "available_after_selection": (len(available_ids) - len(selection)),
            "cumulative_labeled_objects": len(self.labeled_sets.object_ids_through(cycle)),
            "cumulative_manifest": str(cumulative_manifest),
            "inference_checkpoint": str(inference_checkpoint),
            "training_initial_checkpoint": (
                str(training_request.initial_checkpoint)
                if training_request.initial_checkpoint is not None
                else None
            ),
            "trained_checkpoint": str(training.checkpoint_path),
            "training_summary": str(training.summary_path),
            "evaluation_summary": str(evaluation.summary_path),
        }
        _json_dump_atomic(summary, summary_path)
        state.complete_cycle(cycle)

        return CycleResult(
            cycle=cycle,
            selection=selection,
            labeled=labeled,
            training=training,
            evaluation=evaluation,
            cumulative_manifest=cumulative_manifest,
            summary_path=summary_path,
        )

    def _finalize_completed_cycle(
        self,
        *,
        cycle: int,
        state: RunState,
        cycle_dir: Path,
        artifact_cycle_dir: Path,
        summary_path: Path,
    ) -> CycleResult:
        """Recover the narrow window between summary write and state commit."""

        payload = json.loads(summary_path.read_text(encoding="utf-8"))

        if (
            payload.get("run_id") != self.config.run_id
            or int(payload.get("cycle", -1)) != cycle
            or payload.get("strategy") != self.config.strategy
        ):
            raise ValueError("Completed cycle summary has a different experiment identity.")

        selection = _load_selection(cycle_dir / "selection.json")
        labeled = _load_labeled(cycle_dir / "revealed_labels.json")
        cumulative_manifest = Path(payload["cumulative_manifest"])
        request = TrainingRequest(
            train_manifest=cumulative_manifest,
            validation_manifest=self._validation_manifest(),
            output_dir=artifact_cycle_dir / "training",
            run_id=self.config.run_id,
            cycle=cycle,
            strategy=self.config.strategy,
            seed=self.config.seed,
            initial_checkpoint=self._training_checkpoint(cycle),
        )
        training = self._train_or_resume(request)
        evaluation = self._evaluate_or_resume(
            checkpoint_path=training.checkpoint_path,
            output_dir=artifact_cycle_dir / "evaluation",
        )
        state.complete_cycle(cycle)

        return CycleResult(
            cycle=cycle,
            selection=selection,
            labeled=labeled,
            training=training,
            evaluation=evaluation,
            cumulative_manifest=cumulative_manifest,
            summary_path=summary_path,
        )

    def available_ids(self) -> tuple[str, ...]:
        """Return acquisition-pool IDs not acquired in completed/active cycles."""

        acquired = {
            record.object_id
            for record in self.labeled_sets.get_acquired_through(self._last_recorded_cycle())
        }

        return tuple(object_id for object_id in self.acquisition_ids if object_id not in acquired)

    def _infer_or_resume(
        self,
        *,
        object_ids: Sequence[str],
        checkpoint_path: Path,
        path: Path,
    ) -> ClassificationInferenceResult:
        if path.exists():
            result = _load_inference(path)
        else:
            result = self.predictor.predict(
                object_ids=object_ids,
                checkpoint_path=checkpoint_path,
            )
            _save_inference(result, path)

        if len(result.object_ids) != len(set(result.object_ids)):
            raise ValueError("Inference must contain one row per object.")

        if set(result.object_ids) != set(object_ids):
            raise ValueError("Inference objects do not match the current unlabeled set.")

        return result

    def _select_or_resume(
        self,
        *,
        inference: ClassificationInferenceResult,
        budget: int,
        cycle: int,
        path: Path,
    ) -> SelectionBatch:
        if path.exists():
            selection = _load_selection(path)
        else:
            scorer = self._scorer(cycle)
            acquisition_data = self._scoring_input(inference, cycle=cycle)
            score_path = path.with_name("candidate_scores.npz")

            if score_path.exists():
                scores = _load_candidate_scores(
                    score_path,
                    expected_ids=inference.object_ids,
                    expected_strategy=self.config.strategy,
                )
            else:
                scores = scorer.score(acquisition_data)
                _save_candidate_scores(
                    object_ids=inference.object_ids,
                    scores=scores,
                    strategy=self.config.strategy,
                    path=score_path,
                )

            pool = Pool.from_ids(inference.object_ids)
            if self.config.strategy == "entropy_diversity":
                if inference.embeddings is None:
                    raise ValueError("Entropy diversity requires object-aligned embeddings.")
                selector = DiversitySelector(
                    inference.embeddings,
                    shortlist_multiplier=self.config.batch_policy.shortlist_multiplier,
                )
            elif self.config.strategy == "entropy_balanced_explore":
                selector = ExplorationSelector(
                    exploration_fraction=self.config.batch_policy.exploration_fraction,
                    rng=np.random.default_rng(
                        np.random.SeedSequence([self.config.seed, cycle, 173])
                    ),
                )
            else:
                selector = TopKSelector()
            selection = selector.select(
                ids=list(inference.object_ids),
                scores=scores,
                k=budget,
                pool=pool,
            )
            if self.config.strategy == "entropy_balanced_explore":
                selection.metadata.update(
                    {
                        "class_names": list(self.class_names),
                        "labeled_object_counts": acquisition_data.labeled_counts.tolist(),
                        "class_bonus_weights": scorer.class_weights(
                            acquisition_data.labeled_counts
                        ).tolist(),
                        "count_source": "confirmed_training_manifest_previous_cycle",
                    }
                )
            _save_selection(selection, path)

        if len(selection) != budget:
            raise ValueError("Persisted selection has a different acquisition budget.")

        if not set(selection.ids).issubset(inference.object_ids):
            raise ValueError("Persisted selection is not a subset of current inference.")

        return selection

    def _labels_or_resume(
        self,
        *,
        selection: SelectionBatch,
        path: Path,
    ) -> LabeledBatch:
        if path.exists():
            labeled = _load_labeled(path)
        else:
            labeled = ExistingLabelsOracle(self.hidden_labels).query(selection)
            _save_labeled(labeled, path)

        if tuple(labeled.ids) != tuple(selection.ids):
            raise ValueError("Revealed labels do not match the selected object order.")

        return labeled

    def _training_rows(
        self,
        *,
        selection: SelectionBatch,
        labeled: LabeledBatch,
    ) -> tuple[tuple[AcquisitionRecord, ...], tuple[dict[str, object], ...]]:
        fieldnames = _csv_fieldnames(self.labeled_sets.base_manifest)
        id_column = self._required_dataset_column(
            self.dataset.object_id_column,
            "object_id_column",
        )
        file_column = self._required_dataset_column(
            self.dataset.sample_file_column,
            "sample_file_column",
        )
        label_column = self._required_dataset_column(
            self.dataset.label_column,
            "label_column",
        )
        label_by_id = {
            object_id: label
            for object_id, label in zip(
                labeled.ids,
                labeled.labels,
                strict=True,
            )
        }
        records = []
        rows = []

        for object_id in selection.ids:
            physical_records = self.manifest.records_for_object(object_id)

            if not physical_records:
                raise ValueError(f"Selected object {object_id!r} has no physical samples.")

            label = str(label_by_id[object_id])
            references = tuple(
                dict.fromkeys(
                    record.source_reference
                    for record in physical_records
                    if record.source_reference
                )
            )
            records.append(
                AcquisitionRecord(
                    object_id=object_id,
                    survey=physical_records[0].survey,
                    label=label,
                    strategy=self.config.strategy,
                    data_reference=("|".join(references) if references else None),
                )
            )

            for physical in physical_records:
                row: dict[str, object] = {field: "" for field in fieldnames}
                row[id_column] = object_id
                row[file_column] = physical.filepath
                row[label_column] = label

                if "alert" in row:
                    row["alert"] = physical.alert

                rows.append(row)

        return tuple(records), tuple(rows)

    def _train_or_resume(
        self,
        request: TrainingRequest,
    ) -> TrainingResult:
        completed_result = getattr(
            self.trainer,
            "completed_result",
            None,
        )

        if callable(completed_result):
            result = completed_result(request)

            if result is not None:
                return result

        return self.trainer.train(request)

    def _evaluate_or_resume(
        self,
        *,
        checkpoint_path: Path,
        output_dir: Path,
    ) -> ModelEvaluationResult:
        completed = self.evaluator.completed_result(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )

        if completed is not None:
            return completed

        return self.evaluator.evaluate(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )

    def _record_evaluation(
        self,
        *,
        cycle: int,
        n_acquired: int,
        result: ModelEvaluationResult,
    ) -> None:
        history = (
            EvaluationHistory.load(self.history_path)
            if self.history_path.exists()
            else EvaluationHistory()
        )
        n_labeled = len(self.labeled_sets.object_ids_through(cycle))

        for split in result.splits:
            duplicate = any(
                record.cycle == cycle
                and record.split == split.split
                and record.strategy == self.config.strategy
                and record.seed == self.config.seed
                and record.run_id == self.config.run_id
                for record in history.records
            )

            if duplicate:
                continue

            history.add(
                cycle=cycle,
                n_labeled=n_labeled,
                n_acquired=n_acquired,
                split=split.split,
                metrics=split.object_metrics,
                strategy=self.config.strategy,
                seed=self.config.seed,
                run_id=self.config.run_id,
            )

        history.save(self.history_path)

    def _inference_checkpoint(self, cycle: int) -> Path:
        if cycle == 1:
            return self.baseline_checkpoint

        checkpoint = (
            self.artifact_dir / f"cycle_{cycle - 1:03d}" / "training" / "best_vit_tower.pth"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(f"Previous-cycle checkpoint is missing: {checkpoint}")

        return checkpoint

    def _training_checkpoint(self, cycle: int) -> Path | None:
        mode = self.config.training_initialization

        if mode == "restart_from_random":
            return None

        if mode == "restart_from_baseline" or cycle == 1:
            return self.baseline_checkpoint

        return self._inference_checkpoint(cycle)

    def _scorer(self, cycle: int):
        if self.config.strategy in {"entropy", "entropy_diversity"}:
            return EntropyScorer()

        if self.config.strategy == "entropy_balanced_explore":
            return self.config.batch_policy.make_scorer()

        if self.config.strategy == "fisher":
            return FisherScorer(include_bias=True)

        if self.config.strategy == "margin":
            return MarginScorer()

        seed_sequence = np.random.SeedSequence([self.config.seed, cycle])
        return RandomScorer(np.random.default_rng(seed_sequence))

    def _scoring_input(
        self,
        inference: ClassificationInferenceResult,
        *,
        cycle: int = 1,
    ) -> object:
        """Build the explicit input required by the configured scorer."""

        if self.config.strategy == "random":
            return len(inference.object_ids)

        if self.config.strategy == "entropy_balanced_explore":
            return ClassBalancedInput(
                predictions=inference.predictions,
                labeled_counts=self._confirmed_object_counts(cycle - 1),
            )

        if self.config.strategy == "fisher":
            if inference.embeddings is None:
                raise ValueError(
                    "Fisher acquisition requires final-layer embeddings from the inference backend."
                )

            return FisherScoringInput(
                predictions=inference.predictions,
                embeddings=inference.embeddings,
            )

        return inference.predictions

    def _confirmed_object_counts(self, through_cycle: int) -> np.ndarray:
        """Read only confirmed L_t; never consult the hidden candidate labels."""
        path = self.labeled_sets.training_manifest(through_cycle)
        object_column = self.dataset.object_id_column
        label_column = self.dataset.label_column
        by_object: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                object_id = row[object_column]
                raw_label = row[label_column]
                label = self.label_mapping.get(raw_label, raw_label)
                if label not in self.class_names:
                    raise ValueError(f"Confirmed label {raw_label!r} has no probability column.")
                previous = by_object.setdefault(object_id, label)
                if previous != label:
                    raise ValueError(f"Conflicting confirmed labels for {object_id!r}.")
        return np.asarray(
            [sum(label == name for label in by_object.values()) for name in self.class_names],
            dtype=np.int64,
        )

    def _validation_manifest(self) -> Path:
        if self.dataset.validation_manifest is None:
            raise ValueError("Dataset requires a fixed validation manifest for training.")

        return self.dataset.validation_manifest

    def _last_recorded_cycle(self) -> int:
        if not self.labeled_sets.ledger_path.exists():
            return 0

        with self.labeled_sets.ledger_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))

        return max(
            (int(row["cycle"]) for row in rows),
            default=0,
        )

    def _validate_static_inputs(self) -> None:
        if not self.baseline_checkpoint.exists():
            raise FileNotFoundError(self.baseline_checkpoint)

        if len(self.holdout_ids) != len(set(self.holdout_ids)):
            raise ValueError("holdout_ids must not contain duplicates.")

        unknown_holdout = set(self.holdout_ids).difference(self.manifest.object_ids)

        if unknown_holdout:
            raise ValueError(
                f"holdout_ids are not present in the candidate pool: {sorted(unknown_holdout)}."
            )

        missing_labels = set(self.manifest.object_ids).difference(self.hidden_labels)

        if missing_labels:
            raise ValueError(
                f"Hidden labels are missing for candidate objects: {sorted(missing_labels)[:10]}."
            )

        missing_files = self.manifest.missing_files()

        if missing_files:
            raise FileNotFoundError(
                "Candidate pool references missing samples: " + ", ".join(missing_files[:10])
            )

        if not self.acquisition_ids:
            raise ValueError("The acquisition pool is empty.")

        for path in (
            self.dataset.train_manifest,
            self.dataset.validation_manifest,
        ):
            if path is None or not path.exists():
                raise FileNotFoundError(path)

        if self.config.training_initialization == "restart_from_random":
            initialization = getattr(self.trainer, "initialization", None)

            if initialization not in {None, "random_vit"}:
                raise ValueError(
                    "restart_from_random requires a trainer configured with "
                    "initialization='random_vit'."
                )

    def _validate_no_historical_overlap(self) -> None:
        historical_ids = self.labeled_sets.object_ids_through(0)
        overlap = historical_ids.intersection(self.manifest.object_ids)

        if overlap:
            raise ValueError(f"Candidate pool overlaps historical L0: {sorted(overlap)[:10]}.")

    def _identity_payload(self) -> dict[str, object]:
        config_payload = asdict(self.config)
        # Preserve identities of pre-existing ordinary strategy runs.
        if self.config.batch_policy is None:
            config_payload.pop("batch_policy")
        payload = {
            "schema_version": 1,
            "config": config_payload,
            "pool_id": self.manifest.pool_id,
            "candidate_manifest": str(self.manifest_path),
            "candidate_manifest_sha256": _sha256_file(self.manifest_path),
            "candidate_objects": len(self.manifest.object_ids),
            "acquisition_objects": len(self.acquisition_ids),
            "artifact_dir": str(self.artifact_dir),
            "holdout_ids": list(self.holdout_ids),
            "hidden_labels_sha256": _mapping_sha256(self.hidden_labels),
            "historical_train_manifest": str(self.dataset.train_manifest),
            "historical_train_manifest_sha256": _sha256_file(self.dataset.train_manifest),
            "validation_manifest": str(self.dataset.validation_manifest),
            "validation_manifest_sha256": _sha256_file(self.dataset.validation_manifest),
            "baseline_checkpoint": str(self.baseline_checkpoint),
            "baseline_checkpoint_sha256": _sha256_file(self.baseline_checkpoint),
        }
        if self.config.strategy == "entropy_balanced_explore":
            payload["class_names"] = list(self.class_names)
            payload["label_mapping"] = self.label_mapping
        return payload

    @staticmethod
    def _required_dataset_column(
        value: str | None,
        name: str,
    ) -> str:
        if value is None:
            raise ValueError(f"DatasetSpec requires {name}.")

        return value


def load_hidden_labels_csv(
    path: str | Path,
    *,
    object_id_column: str = "object_id",
    label_column: str = "target_label",
) -> dict[str, str]:
    """Load a unique object-to-label mapping from an audited CSV."""

    source = Path(path)

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {object_id_column, label_column}
        missing = required.difference(fieldnames)

        if missing:
            raise ValueError(f"Hidden-label CSV is missing columns: {sorted(missing)}.")

        labels: dict[str, str] = {}

        for row in reader:
            object_id = str(row[object_id_column]).strip()
            label = str(row[label_column]).strip()

            if not object_id or not label:
                raise ValueError("Hidden-label rows require non-empty IDs and labels.")

            previous = labels.setdefault(object_id, label)

            if previous != label:
                raise ValueError(f"Object {object_id!r} has conflicting hidden labels.")

    if not labels:
        raise ValueError("Hidden-label CSV is empty.")

    return labels


def stratified_holdout_ids(
    labels: Mapping[str, object],
    *,
    fraction: float,
    seed: int,
    minimum_acquisition_per_class: int = 1,
) -> tuple[str, ...]:
    """Choose a deterministic label-stratified fixed evaluation holdout.

    Classes too small to leave ``minimum_acquisition_per_class`` objects in
    the acquisition pool are not placed in the holdout. This is important for
    the current BOOM/TNS pool, whose rarest classes contain only one or two
    objects.
    """

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between zero and one.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    if minimum_acquisition_per_class < 1:
        raise ValueError("minimum_acquisition_per_class must be at least one.")

    ids_by_label: dict[str, list[str]] = {}

    for object_id, label in labels.items():
        ids_by_label.setdefault(str(label), []).append(str(object_id))

    selected: set[str] = set()

    for class_index, label in enumerate(sorted(ids_by_label)):
        object_ids = sorted(ids_by_label[label])
        maximum = max(
            0,
            len(object_ids) - minimum_acquisition_per_class,
        )
        count = min(
            int(round(len(object_ids) * fraction)),
            maximum,
        )

        if count == 0:
            continue

        generator = np.random.default_rng(np.random.SeedSequence([seed, class_index]))
        chosen = generator.choice(
            np.asarray(object_ids, dtype=object),
            size=count,
            replace=False,
        )
        selected.update(str(value) for value in chosen)

    return tuple(object_id for object_id in labels if object_id in selected)


def materialize_labeled_candidate_manifest(
    *,
    candidate_manifest: CandidatePoolManifest,
    dataset: DatasetSpec,
    labels: Mapping[str, object],
    object_ids: Sequence[str],
    destination: str | Path,
) -> Path:
    """Create a model-ready labeled manifest for a fixed candidate subset."""

    if dataset.train_manifest is None:
        raise ValueError("DatasetSpec requires train_manifest.")

    columns = {
        "object_id": dataset.object_id_column,
        "filepath": dataset.sample_file_column,
        "label": dataset.label_column,
    }

    if any(value is None for value in columns.values()):
        raise ValueError("DatasetSpec requires object_id, sample-file, and label columns.")

    requested = tuple(str(value) for value in object_ids)

    if len(requested) != len(set(requested)):
        raise ValueError("object_ids must not contain duplicates.")

    unknown = set(requested).difference(candidate_manifest.object_ids)

    if unknown:
        raise ValueError(f"Manifest subset contains unknown objects: {sorted(unknown)}.")

    missing_labels = set(requested).difference(labels)

    if missing_labels:
        raise ValueError(f"Manifest subset is missing labels: {sorted(missing_labels)}.")

    fieldnames = _csv_fieldnames(dataset.train_manifest)
    rows: list[dict[str, object]] = []

    for object_id in requested:
        for physical in candidate_manifest.records_for_object(object_id):
            row: dict[str, object] = {field: "" for field in fieldnames}
            row[str(columns["object_id"])] = object_id
            row[str(columns["filepath"])] = physical.filepath
            row[str(columns["label"])] = str(labels[object_id])

            if "alert" in row:
                row["alert"] = physical.alert

            rows.append(row)

    destination = Path(destination)

    if destination.exists():
        with destination.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            existing = list(csv.DictReader(handle))

        normalized = [{field: str(row[field]) for field in fieldnames} for row in rows]

        if existing != normalized:
            raise ValueError("Existing labeled candidate manifest has different rows.")

        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(destination)
    return destination


def _save_inference(
    result: ClassificationInferenceResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    payload: dict[str, np.ndarray] = {
        "object_ids": np.asarray(result.object_ids, dtype=str),
        "alerts": np.asarray(
            [-1 if alert is None else alert for alert in result.alerts],
            dtype=np.int64,
        ),
        "values": result.predictions.values,
        "output_type": np.asarray(result.predictions.output_type),
    }

    if result.embeddings is not None:
        payload["embeddings"] = result.embeddings

    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    temporary.replace(path)


def _load_inference(path: Path) -> ClassificationInferenceResult:
    with np.load(path, allow_pickle=False) as payload:
        return ClassificationInferenceResult(
            object_ids=tuple(payload["object_ids"].astype(str)),
            alerts=tuple(None if int(value) < 0 else int(value) for value in payload["alerts"]),
            predictions=ClassificationPredictions(
                values=np.asarray(payload["values"]),
                output_type=str(payload["output_type"].item()),
            ),
            embeddings=(
                np.asarray(payload["embeddings"]) if "embeddings" in payload.files else None
            ),
        )


def _save_candidate_scores(
    *,
    object_ids: Sequence[str],
    scores: np.ndarray,
    strategy: str,
    path: Path,
) -> None:
    """Persist every candidate score for audit and strategy comparison."""

    ids = np.asarray(object_ids, dtype=str)
    score_values = np.asarray(scores, dtype=float)

    if score_values.shape != (len(ids),):
        raise ValueError("Candidate scores must contain one value per object.")

    if not np.all(np.isfinite(score_values)):
        raise ValueError("Candidate scores must be finite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            object_ids=ids,
            scores=score_values,
            strategy=np.asarray(strategy),
        )

    temporary.replace(path)


def _load_candidate_scores(
    path: Path,
    *,
    expected_ids: Sequence[str],
    expected_strategy: str,
) -> np.ndarray:
    """Load and validate a resumable full-pool score artifact."""

    with np.load(path, allow_pickle=False) as payload:
        object_ids = tuple(payload["object_ids"].astype(str))
        scores = np.asarray(payload["scores"], dtype=float)
        strategy = str(payload["strategy"].item())

    if object_ids != tuple(expected_ids):
        raise ValueError("Persisted candidate-score objects do not match inference.")

    if strategy != expected_strategy:
        raise ValueError("Persisted candidate scores use a different strategy.")

    if scores.shape != (len(object_ids),):
        raise ValueError("Persisted candidate scores have an invalid shape.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("Persisted candidate scores contain non-finite values.")

    return scores


def _save_selection(selection: SelectionBatch, path: Path) -> None:
    _json_dump_atomic(
        {
            "ids": list(selection.ids),
            "indices": selection.indices.tolist(),
            "scores": selection.scores.tolist(),
            **({"metadata": selection.metadata} if selection.metadata else {}),
        },
        path,
    )


def _load_selection(path: Path) -> SelectionBatch:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SelectionBatch(
        ids=[str(value) for value in payload["ids"]],
        indices=np.asarray(payload["indices"], dtype=np.int64),
        scores=np.asarray(payload["scores"], dtype=float),
        metadata=payload.get("metadata", {}),
    )


def _save_labeled(labeled: LabeledBatch, path: Path) -> None:
    _json_dump_atomic(
        {
            "ids": list(labeled.ids),
            "labels": labeled.labels.tolist(),
        },
        path,
    )


def _load_labeled(path: Path) -> LabeledBatch:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LabeledBatch(
        ids=[str(value) for value in payload["ids"]],
        labels=np.asarray(payload["labels"]),
    )


def _csv_fieldnames(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)

        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc


def _mapping_sha256(mapping: Mapping[str, object]) -> str:
    payload = json.dumps(
        sorted((str(key), str(value)) for key, value in mapping.items()),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path | None) -> str:
    if path is None:
        raise ValueError("Cannot hash an undefined path.")

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _json_dump_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "CycleResult",
    "PoolPredictor",
    "SimulatedALConfig",
    "SimulatedActiveLearningExperiment",
    "StrategyName",
    "TrainingInitialization",
    "load_hidden_labels_csv",
    "materialize_labeled_candidate_manifest",
    "stratified_holdout_ids",
]
