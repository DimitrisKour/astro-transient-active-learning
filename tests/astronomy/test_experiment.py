"""Tests for the resumable simulated-label experiment coordinator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest, CandidatePoolRecord
from al_pipeline.astronomy.data.registry import DatasetSpec
from al_pipeline.core.evaluation.base import ModelEvaluationResult, SplitEvaluation
from al_pipeline.astronomy.experiments.simulated import (
    SimulatedALConfig,
    SimulatedActiveLearningExperiment,
    materialize_labeled_candidate_manifest,
    stratified_holdout_ids,
)
from al_pipeline.core.training.base import TrainingRequest, TrainingResult
from al_pipeline.astronomy.astrominn.types import ClassificationInferenceResult
from al_pipeline.core.types import ClassificationPredictions


class _Predictor:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def predict(self, *, object_ids, checkpoint_path):
        object_ids = tuple(object_ids)
        self.calls.append((object_ids, Path(checkpoint_path)))
        probabilities = []

        for index, _ in enumerate(object_ids):
            probability = 0.51 + 0.08 * index
            probabilities.append([probability, 1.0 - probability])

        return ClassificationInferenceResult(
            object_ids=object_ids,
            alerts=tuple(1 for _ in object_ids),
            predictions=ClassificationPredictions(
                values=np.asarray(probabilities),
                output_type="probabilities",
            ),
            embeddings=np.asarray([[float(index + 1), 1.0] for index in range(len(object_ids))]),
        )


class _Trainer:
    initialization = "pretrained"

    def __init__(self) -> None:
        self.requests: list[TrainingRequest] = []
        self.results: dict[Path, TrainingResult] = {}

    def completed_result(self, request):
        return self.results.get(request.output_dir)

    def train(self, request):
        self.requests.append(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = request.output_dir / "best_vit_tower.pth"
        history = request.output_dir / "training_history.json"
        summary = request.output_dir / "training_summary.json"
        checkpoint.write_bytes(f"cycle-{request.cycle}".encode())
        history.write_text("[]\n", encoding="utf-8")
        summary.write_text("{}\n", encoding="utf-8")
        result = TrainingResult(
            checkpoint_path=checkpoint,
            history_path=history,
            summary_path=summary,
            best_epoch=1,
            epochs_completed=1,
            best_validation_loss=1.0,
            elapsed_seconds=0.1,
        )
        self.results[request.output_dir] = result
        return result


class _Evaluator:
    def completed_result(self, *, checkpoint_path, output_dir):
        summary = Path(output_dir) / "evaluation_summary.json"

        if not summary.exists():
            return None

        return self._result(Path(checkpoint_path), summary)

    def evaluate(self, *, checkpoint_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = output_dir / "evaluation_summary.json"
        summary.write_text(
            json.dumps({"checkpoint": str(checkpoint_path)}) + "\n",
            encoding="utf-8",
        )
        return self._result(Path(checkpoint_path), summary)

    @staticmethod
    def _result(checkpoint, summary):
        return ModelEvaluationResult(
            checkpoint_path=checkpoint,
            summary_path=summary,
            splits=(
                SplitEvaluation(
                    split="test",
                    sample_count=4,
                    object_count=4,
                    sample_metrics={"accuracy": 0.5},
                    object_metrics={"accuracy": 0.5},
                ),
            ),
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["object_id", "filepath", "label", "alert"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path: Path):
    historical_sample = tmp_path / "historical.npy"
    historical_sample.touch()
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    historical_rows = [
        {
            "object_id": "old",
            "filepath": historical_sample,
            "label": "A",
            "alert": 1,
        }
    ]
    _write_csv(train, historical_rows)
    _write_csv(validation, historical_rows)
    dataset = DatasetSpec(
        name="toy",
        survey="ZTF",
        modality="image_metadata",
        selection_unit="object",
        train_manifest=train,
        validation_manifest=validation,
        test_manifest=validation,
        object_id_column="object_id",
        sample_file_column="filepath",
        label_column="label",
    )
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    records = []

    for index in range(5):
        object_id = f"new_{index}"
        sample = pool_dir / f"{object_id}.npy"
        sample.touch()
        records.append(
            CandidatePoolRecord(
                survey="ZTF",
                object_id=object_id,
                alert=1,
                filepath=str(sample),
                source_reference=f"test:{index}",
            )
        )

    CandidatePoolManifest(
        pool_id="toy-pool",
        records=records,
    ).to_csv(pool_dir / "candidate_pool.csv")
    checkpoint = tmp_path / "baseline.pth"
    checkpoint.write_bytes(b"baseline")
    labels = {record.object_id: "A" for record in records}
    return dataset, pool_dir, checkpoint, labels


def test_two_cycles_are_fully_automated_and_cumulative(
    tmp_path: Path,
) -> None:
    dataset, pool_dir, checkpoint, labels = _inputs(tmp_path)
    predictor = _Predictor()
    trainer = _Trainer()
    experiment = SimulatedActiveLearningExperiment(
        config=SimulatedALConfig(
            run_id="toy-entropy",
            strategy="entropy",
            acquisition_size=2,
            max_cycles=2,
        ),
        dataset=dataset,
        pool_path=pool_dir,
        hidden_labels=labels,
        run_dir=tmp_path / "state",
        artifact_dir=tmp_path / "heavy",
        baseline_checkpoint=checkpoint,
        predictor=predictor,
        trainer=trainer,
        evaluator=_Evaluator(),
    )

    results = experiment.run()

    assert [result.cycle for result in results] == [1, 2]
    assert len(experiment.available_ids()) == 1
    assert len(experiment.labeled_sets.object_ids_through(2)) == 5
    assert trainer.requests[0].initial_checkpoint == checkpoint
    assert trainer.requests[1].initial_checkpoint == results[0].training.checkpoint_path
    assert len(predictor.calls[0][0]) == 5
    assert len(predictor.calls[1][0]) == 3

    history = json.loads(experiment.history_path.read_text(encoding="utf-8"))
    assert [row["cycle"] for row in history] == [0, 1, 2]


def test_strategy_runs_do_not_share_mutable_state(tmp_path: Path) -> None:
    dataset, pool_dir, checkpoint, labels = _inputs(tmp_path)
    experiments = []

    for strategy in ("entropy", "random"):
        experiment = SimulatedActiveLearningExperiment(
            config=SimulatedALConfig(
                run_id=f"toy-{strategy}",
                strategy=strategy,
                acquisition_size=2,
                max_cycles=1,
            ),
            dataset=dataset,
            pool_path=pool_dir,
            hidden_labels=labels,
            run_dir=tmp_path / strategy / "state",
            artifact_dir=tmp_path / strategy / "heavy",
            baseline_checkpoint=checkpoint,
            predictor=_Predictor(),
            trainer=_Trainer(),
            evaluator=_Evaluator(),
        )
        experiment.run()
        experiments.append(experiment)

    assert len(experiments[0].available_ids()) == 3
    assert len(experiments[1].available_ids()) == 3
    assert experiments[0].labeled_sets.ledger_path != experiments[1].labeled_sets.ledger_path


def test_fisher_strategy_persists_full_candidate_scores(
    tmp_path: Path,
) -> None:
    dataset, pool_dir, checkpoint, labels = _inputs(tmp_path)
    experiment = SimulatedActiveLearningExperiment(
        config=SimulatedALConfig(
            run_id="toy-fisher",
            strategy="fisher",
            acquisition_size=2,
            max_cycles=1,
        ),
        dataset=dataset,
        pool_path=pool_dir,
        hidden_labels=labels,
        run_dir=tmp_path / "state",
        artifact_dir=tmp_path / "heavy",
        baseline_checkpoint=checkpoint,
        predictor=_Predictor(),
        trainer=_Trainer(),
        evaluator=_Evaluator(),
    )

    results = experiment.run()
    score_path = experiment.run_dir / "cycle_001" / "candidate_scores.npz"

    assert len(results) == 1
    assert score_path.exists()

    with np.load(score_path, allow_pickle=False) as payload:
        assert tuple(payload["object_ids"].astype(str)) == tuple(experiment.acquisition_ids)
        assert payload["scores"].shape == (5,)
        assert np.all(np.isfinite(payload["scores"]))
        assert str(payload["strategy"].item()) == "fisher"


def test_stratified_holdout_preserves_rare_acquisition_examples() -> None:
    labels = {
        **{f"common_{index}": "common" for index in range(10)},
        "rare_only": "rare",
    }

    holdout = stratified_holdout_ids(
        labels,
        fraction=0.2,
        seed=135,
    )

    assert len(holdout) == 2
    assert "rare_only" not in holdout


def test_materialize_labeled_holdout_manifest(tmp_path: Path) -> None:
    dataset, pool_dir, _, labels = _inputs(tmp_path)
    manifest = CandidatePoolManifest.from_csv(pool_dir / "candidate_pool.csv")
    destination = materialize_labeled_candidate_manifest(
        candidate_manifest=manifest,
        dataset=dataset,
        labels=labels,
        object_ids=("new_0", "new_1"),
        destination=tmp_path / "holdout.csv",
    )

    with destination.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["object_id"] for row in rows] == ["new_0", "new_1"]
    assert {row["label"] for row in rows} == {"A"}
