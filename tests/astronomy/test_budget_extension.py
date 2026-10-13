"""Extend a real coordinator run without repeating prior acquisition/training."""
import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest, CandidatePoolRecord
from al_pipeline.astronomy.data.registry import DatasetSpec
from al_pipeline.astronomy.experiments.simulated import SimulatedActiveLearningExperiment
from al_pipeline.astronomy.astrominn.types import ClassificationInferenceResult
from al_pipeline.core.experiments.config import SimulatedALConfig
from al_pipeline.core.types import ClassificationPredictions
from al_pipeline.core.training.base import TrainingResult
from al_pipeline.core.evaluation.base import ModelEvaluationResult, SplitEvaluation


class Predictor:
    def predict(self, *, object_ids, checkpoint_path):
        return ClassificationInferenceResult(
            object_ids=tuple(object_ids), alerts=(1,) * len(object_ids),
            predictions=ClassificationPredictions(
                values=np.tile([0.6, 0.4], (len(object_ids), 1)),
                output_type="probabilities",
            ),
        )


class Trainer:
    initialization = "random_vit"

    def __init__(self):
        self.requests = []
        self.results = {}

    def completed_result(self, request):
        return self.results.get(request.output_dir)

    def train(self, request):
        self.requests.append(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        paths = [request.output_dir / name for name in
                 ("best_vit_tower.pth", "training_history.json", "training_summary.json")]
        for p in paths:
            p.write_text(str(request.cycle))
        result = TrainingResult(*paths, best_epoch=1, epochs_completed=1,
                                best_validation_loss=1.0, elapsed_seconds=0.1)
        self.results[request.output_dir] = result
        return result


class Evaluator:
    def __init__(self):
        self.results = {}

    def completed_result(self, *, checkpoint_path, output_dir):
        return self.results.get(output_dir)

    def evaluate(self, *, checkpoint_path, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = output_dir / "evaluation_summary.json"
        summary.write_text("{}")
        result = ModelEvaluationResult(
            checkpoint_path=checkpoint_path, summary_path=summary,
            splits=(SplitEvaluation(split="test", sample_count=1, object_count=1,
                                    sample_metrics={"accuracy": 0.5},
                                    object_metrics={"accuracy": 0.5}),),
        )
        self.results[output_dir] = result
        return result


@pytest.fixture
def completed(tmp_path):
    sample = tmp_path / "old.npy"
    sample.touch()
    manifest = tmp_path / "train.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object_id", "filepath", "label", "alert"])
        writer.writeheader()
        writer.writerow(dict(object_id="old", filepath=str(sample), label="A", alert=1))
    dataset = DatasetSpec(name="toy", survey="ZTF", modality="image_metadata",
                          selection_unit="object", train_manifest=manifest,
                          validation_manifest=manifest, test_manifest=manifest,
                          object_id_column="object_id", sample_file_column="filepath",
                          label_column="label")
    pool = tmp_path / "pool"
    pool.mkdir()
    records = []
    for i in range(5):
        p = pool / f"{i}.npy"
        p.touch()
        records.append(CandidatePoolRecord(survey="ZTF", object_id=str(i), alert=1,
                                          filepath=str(p), source_reference="toy"))
    CandidatePoolManifest(pool_id="toy", records=records).to_csv(pool / "candidate_pool.csv")
    checkpoint = tmp_path / "baseline.pth"
    checkpoint.write_bytes(b"baseline")
    experiment = SimulatedActiveLearningExperiment(
        config=SimulatedALConfig(run_id="toy", strategy="entropy", seed=135,
                                 acquisition_size=2, max_cycles=2,
                                 training_initialization="restart_from_random"),
        dataset=dataset, pool_path=pool, hidden_labels={str(i): "A" for i in range(5)},
        run_dir=tmp_path / "state", artifact_dir=tmp_path / "heavy",
        baseline_checkpoint=checkpoint, predictor=Predictor(), trainer=Trainer(),
        evaluator=Evaluator(),
    )
    experiment.run()
    return experiment


def test_extension_runs_only_remainder_and_keeps_prior_outputs(completed):
    old_identity = json.loads(completed.identity_path.read_text())
    old_history = completed.history_path.read_bytes()
    earlier = {p: p.read_bytes() for root in (completed.run_dir, completed.artifact_dir)
               for p in root.rglob("*") if p.is_file() and "cycle_00" in str(p)}
    completed.config = replace(completed.config, max_cycles=3, allow_partial_final_cycle=True)
    with pytest.raises(ValueError, match="different experiment"):
        completed.initialize()
    completed.extend_completed_budget()
    assert completed.history_path.read_bytes() == old_history
    assert json.loads((completed.run_dir / "experiment_before_budget_extension_002.json").read_text()) == old_identity
    results = completed.run()
    assert [r.cycle for r in results] == [3]
    assert len(results[0].selection) == 1
    assert [r.cycle for r in completed.trainer.requests] == [1, 2, 3]
    assert all(p.read_bytes() == content for p, content in earlier.items())
    assert len(completed.labeled_sets.object_ids_through(3)) == 6
    completed.extend_completed_budget()
    assert completed.run() == ()


@pytest.mark.parametrize("change", [dict(seed=246), dict(acquisition_size=1)])
def test_extension_rejects_scientific_changes(completed, change):
    original = completed.identity_path.read_bytes()
    completed.config = replace(completed.config, max_cycles=3,
                               allow_partial_final_cycle=True, **change)
    with pytest.raises(ValueError, match="inputs or policy"):
        completed.extend_completed_budget()
    assert completed.identity_path.read_bytes() == original


def test_extension_rejects_active_run(completed):
    state = completed.initialize()
    state.start_next_cycle()
    completed.config = replace(completed.config, max_cycles=3, allow_partial_final_cycle=True)
    with pytest.raises(ValueError, match="no active cycle"):
        completed.extend_completed_budget()


def test_extension_rejects_missing_prior_training(completed):
    completed.trainer.results.clear()
    completed.config = replace(completed.config, max_cycles=3, allow_partial_final_cycle=True)
    with pytest.raises(ValueError, match="prior training"):
        completed.extend_completed_budget()
