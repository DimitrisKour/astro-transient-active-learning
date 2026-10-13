"""Scientific controls, immutable inputs, resumability and paired accounting."""

from dataclasses import asdict
from pathlib import Path
import csv
import json

import pytest

from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainingConfig
from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.training_diagnostics import (
    ORDER_SEEDS, ORDER_STRATEGIES, canonical_rows, load_plan, prepare_diagnostics,
    read_manifest, run_task, sha256, verify_plan, write_json,
)
from al_pipeline.astronomy.training_diagnostic_reporting import report_diagnostics
from al_pipeline.core.training.base import TrainingResult
from al_pipeline.core.evaluation.base import ModelEvaluationResult, SplitEvaluation

REPO = Path(__file__).resolve().parents[2]


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["obj_id", "filepath", "target", "alert", "survey"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fixture(tmp_path):
    base = tmp_path / "input"
    legacy = base / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "astroMiNN_V2.py").write_text("# immutable legacy model\n")
    artifacts = base / "model"
    artifacts.mkdir()
    for name in ["astroMiNN_config.json", "vit_tower_135.pth", "best_lc_tower_135.pth",
                 "best_spatial_tower_135.pth", "best_coord_tower_135.pth"]:
        (artifacts / name).write_text("{}")
    labels = ["AGN", "CV", "SN I", "SN II", "TDE", "SN I", "SN II"]
    rows = []
    for i, label in enumerate(labels):
        sample = base / f"sample_{i}.npy"
        sample.write_bytes(b"fixture only")
        rows.append({"obj_id": f"obj_{i}", "filepath": str(sample), "target": label,
                     "alert": "1", "survey": "ZTF"})
    rows.append(dict(rows[0]))  # Preserve multiplicity, rather than deduplicating.
    full = base / "train.csv"
    validation = base / "validation.csv"
    test = base / "test.csv"
    write_manifest(full, rows)
    write_manifest(validation, [{**rows[0], "obj_id": "validation_only"}])
    write_manifest(test, [{**rows[0], "obj_id": "test_only"}])
    persistent, heavy = tmp_path / "persistent", tmp_path / "heavy"
    registry = base / "registry.toml"
    registry.write_text(f'''
[storage]
persistent_root = "{persistent}"
data_root = "{heavy}"
[datasets.astrominn_ztf]
survey = "ZTF"
modality = "image_metadata"
selection_unit = "object"
train_manifest = "{full}"
validation_manifest = "{validation}"
test_manifest = "{test}"
data_root = "{base}"
object_id_column = "obj_id"
sample_file_column = "filepath"
label_column = "target"
[profiles.astrominn_ztf]
model = "AstroMiNN"
dataset = "astrominn_ztf"
checkpoint = "{artifacts / 'vit_tower_135.pth'}"
''')
    suite = persistent / "runs/astrominn_historical_replay_v1"
    source_artifacts = heavy / "runs/astrominn_historical_replay_v1"
    for seed in ORDER_SEEDS:
        for n, strategy in enumerate(ORDER_STRATEGIES):
            state_dir = suite / strategy / f"seed_{seed}"
            state_dir.mkdir(parents=True)
            write_json(state_dir / "run_state.json", {"last_completed_cycle": 9, "active_cycle": None})
            write_json(state_dir / "experiment.json", {
                "config": {"training_initialization": "restart_from_random"},
                "artifact_dir": str(source_artifacts / strategy / f"seed_{seed}"),
                "validation_manifest_sha256": sha256(validation),
            })
            cycles = (6, 7, 9) if strategy == "entropy_balanced_explore" and seed == 135 else (9,)
            for cycle in cycles:
                chosen = rows if cycle == 9 else [r for r in rows if int(r["obj_id"][-1]) < cycle - 1]
                chosen = chosen[n:] + chosen[:n]
                manifest = state_dir / "manifests" / f"train_cycle_{cycle:03d}.csv"
                write_manifest(manifest, chosen)
                policy = asdict(AstroMiNNViTTrainingConfig(
                    loader_seed=seed, max_epochs=2, patience=1, train_num_workers=0,
                    validation_num_workers=0, class_weight_source="training_manifest",
                ))
                summary = {
                    "cycle": cycle, "seed": seed, "strategy": strategy,
                    "initialization": "random_vit", "initial_checkpoint": None,
                    "train_manifest_sha256": sha256(manifest), "training_samples": len(chosen),
                    "validation_manifest_sha256": sha256(validation), "policy": policy,
                    "run_id": f"source_{strategy}_{seed}", "elapsed_seconds": 200,
                }
                write_json(source_artifacts / strategy / f"seed_{seed}" /
                           f"cycle_{cycle:03d}/training/training_summary.json", summary)
            write_json(state_dir / "evaluation_history.json", [
                {"cycle": cycle, "split": split, "metrics": {
                    "f1_macro": 0.6, "balanced_accuracy": 0.65, "accuracy": 0.7,
                }} for cycle in (6, 7) for split in ("validation", "test")
            ])
    return {
        "repo": REPO, "registry_path": registry, "diagnostic_id": "test_diag",
        "legacy_code_dir": legacy, "repeat_count": 3, "first_repeat_seed": 1001,
    }


def test_default_plan_has_twenty_fresh_pairs_and_fifteen_order_controls(fixture):
    fixture["repeat_count"] = 20
    path = prepare_diagnostics(**fixture)
    plan = load_plan(path)
    tasks = plan["tasks"]
    assert len(tasks) == 57
    assert [t["training_seed"] for t in tasks[:2]] == [135, 135]
    repeats = [t for t in tasks if t["kind"] == "transition" and not t["anchor"]]
    assert len(repeats) == 40
    assert {t["training_seed"] for t in repeats} == set(range(1001, 1021))
    assert {t["loader_seed"] for t in repeats} == {135}
    for seed in ORDER_SEEDS:
        group = [t for t in tasks if t["kind"] == "order" and t["training_seed"] == seed]
        assert len(group) == 5
        assert len({t["content_sha256"] for t in group}) == 1
        assert len({json.dumps(t["policy"], sort_keys=True) for t in group}) == 1
        assert {t["loader_seed"] for t in group} == {seed}
    assert prepare_diagnostics(**fixture) == path
    assert all(not (Path(plan["artifact_root"]) / t["task_id"]).exists() for t in tasks)


def test_canonical_sort_keeps_duplicates_and_original_files(fixture):
    registry = DatasetRegistry.from_toml(fixture["registry_path"]).resolve_profile("astrominn_ztf")
    source = registry.dataset.train_manifest
    original = sha256(source)
    path = prepare_diagnostics(**fixture)
    plan = load_plan(path)
    canonical = read_manifest(Path(plan["tasks"][-5]["manifest"]), registry.dataset)
    before = read_manifest(source, registry.dataset)
    assert len(canonical.rows) == len(before.rows) == 8
    assert canonical.content_sha256 == before.content_sha256
    assert len([r for r in canonical.rows if r["obj_id"] == "obj_0"]) == 2
    assert list(canonical.rows) == canonical_rows(before, registry.dataset)
    assert sha256(source) == original


def test_modified_source_or_frozen_copy_rejects_execution(fixture):
    path = prepare_diagnostics(**fixture)
    plan = load_plan(path)
    manifest = Path(plan["tasks"][0]["manifest"])
    manifest.write_text(manifest.read_text().replace("AGN", "TDE"))
    with pytest.raises(ValueError, match="changed"):
        verify_plan(path, 0)


def test_order_control_rejects_loader_seed_mismatch(fixture):
    resolved = DatasetRegistry.from_toml(fixture["registry_path"]).resolve_profile("astrominn_ztf")
    path = resolved.storage.data_root / "runs/astrominn_historical_replay_v1/entropy/seed_135/cycle_009/training/training_summary.json"
    summary = json.loads(path.read_text())
    summary["policy"]["loader_seed"] = 999
    write_json(path, summary)
    with pytest.raises(ValueError, match="same loader seed"):
        prepare_diagnostics(**fixture)


def test_source_hash_mismatch_fails_before_writes(fixture):
    resolved = DatasetRegistry.from_toml(fixture["registry_path"]).resolve_profile("astrominn_ztf")
    path = resolved.storage.persistent_root / "runs/astrominn_historical_replay_v1/random/seed_135/manifests/train_cycle_009.csv"
    path.write_text(path.read_text().replace("AGN", "CV"))
    with pytest.raises(ValueError, match="Source training mismatch"):
        prepare_diagnostics(**fixture)
    assert not (resolved.storage.persistent_root / "training_diagnostics").exists()


def test_holdout_overlap_rejected(fixture):
    resolved = DatasetRegistry.from_toml(fixture["registry_path"]).resolve_profile("astrominn_ztf")
    path = resolved.dataset.test_manifest
    path.write_text(path.read_text().replace("test_only", "obj_0"))
    with pytest.raises(ValueError, match="overlap"):
        prepare_diagnostics(**fixture)


def test_legacy_wandb_archives_are_not_runtime_inputs(fixture):
    legacy = fixture["legacy_code_dir"]
    archive = legacy / "wandb/run-old/files/things_for_felipe"
    archive.mkdir(parents=True)
    (archive / "Towers.py").symlink_to(archive / "missing-original.py")
    (archive / "old_model.py").write_text("# unused archived source\n")
    path = prepare_diagnostics(**fixture)
    inputs = load_plan(path)["immutable_inputs"]
    assert str(legacy / "astroMiNN_V2.py") in inputs
    assert not any(Path(name).is_relative_to(archive) for name in inputs)
    verify_plan(path)


class FakeTrainer:
    calls = []
    fail_once = False

    def __init__(self, **kwargs):
        self.policy = kwargs["policy"]

    def completed_result(self, request):
        path = request.output_dir / "training_summary.json"
        if not path.exists():
            return None
        summary = json.loads(path.read_text())
        return TrainingResult(
            checkpoint_path=Path(summary["checkpoint_path"]),
            history_path=Path(summary["history_path"]), summary_path=path,
            best_epoch=1, epochs_completed=1, best_validation_loss=0.1, elapsed_seconds=1,
        )

    def train(self, request):
        assert request.initial_checkpoint is None
        self.calls.append(request)
        request.output_dir.mkdir(parents=True)
        checkpoint = request.output_dir / "model.pth"
        checkpoint.write_text(json.dumps({"seed": request.seed, "cycle": request.cycle}))
        if self.fail_once:
            type(self).fail_once = False
            raise RuntimeError("simulated interrupted fit")
        history = request.output_dir / "training_history.json"
        history.write_text("[]")
        summary = request.output_dir / "training_summary.json"
        write_json(summary, {
            "checkpoint_path": str(checkpoint), "history_path": str(history),
            "seed": request.seed, "cycle": request.cycle, "policy": asdict(self.policy),
            "best_epoch": 1, "epochs_completed": 1, "best_validation_loss": 0.1,
            "elapsed_seconds": 1,
        })
        return self.completed_result(request)


class FakeEvaluator:
    fail_once = False

    def __init__(self, **kwargs):
        pass

    def evaluate(self, *, checkpoint_path, output_dir):
        if self.fail_once:
            type(self).fail_once = False
            raise RuntimeError("simulated evaluation failure")
        metadata = json.loads(checkpoint_path.read_text())
        value = 0.6 + (metadata["seed"] % 3) * 0.01
        if metadata["cycle"] == 7:
            value -= 0.03
        metrics = {"f1_macro": value, "accuracy": value, "balanced_accuracy": value}
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in ("validation", "test"):
            write_json(output_dir / f"{split}_per_class_metrics.json", {
                "object": {"TDE": {"precision": 0.2, "recall": 0.5, "f1": 0.3, "support": 11}},
            })
        return ModelEvaluationResult(
            checkpoint_path=checkpoint_path, summary_path=output_dir / "summary.json",
            splits=tuple(SplitEvaluation(split, 10, 5, metrics, metrics)
                         for split in ("validation", "test")),
        )


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeTrainer.calls = []
    FakeTrainer.fail_once = False
    FakeEvaluator.fail_once = False


def execute(plan, index):
    return run_task(plan, index, device="cpu", trainer_class=FakeTrainer,
                    evaluator_class=FakeEvaluator)


def test_resume_evaluation_without_retraining_and_skip_completed(fixture):
    path = prepare_diagnostics(**fixture)
    FakeEvaluator.fail_once = True
    with pytest.raises(RuntimeError, match="evaluation failure"):
        execute(path, 0)
    result = execute(path, 0)
    assert len(FakeTrainer.calls) == 1
    assert execute(path, 0) == result
    assert len(FakeTrainer.calls) == 1


def test_interrupted_training_kept_and_retried_in_new_attempt(fixture):
    path = prepare_diagnostics(**fixture)
    FakeTrainer.fail_once = True
    with pytest.raises(RuntimeError, match="interrupted fit"):
        execute(path, 0)
    first = FakeTrainer.calls[0].output_dir / "model.pth"
    before = first.read_bytes()
    execute(path, 0)
    assert len(FakeTrainer.calls) == 2
    assert first.read_bytes() == before
    assert "attempt_002" in str(FakeTrainer.calls[-1].output_dir)


def test_paired_report_excludes_anchor_and_counts_incomplete_pairs(fixture, tmp_path):
    path = prepare_diagnostics(**fixture)
    # Anchor pair, one fresh pair and one unpaired fresh fit.
    for index in (0, 1, 2, 3, 4):
        execute(path, index)
    summary = report_diagnostics(path, tmp_path / "report", plots=False)
    assert summary["completed_tasks"] == 5
    assert not summary["all_tasks_complete"]
    for row in summary["transition"]:
        assert row["planned_pairs"] == 3
        assert row["completed_pairs"] == 1
        assert row["delta_mean_pp"] == pytest.approx(-3)
        assert row["negative_pairs"] == 1
        assert not row["all_pairs_complete"]
    assert 5 in summary["incomplete_task_indices"]
    assert (tmp_path / "report/anchor_reproduction.csv").exists()


def test_order_report_uses_same_seed_canonical_reference(fixture, tmp_path):
    path = prepare_diagnostics(**fixture)
    plan = load_plan(path)
    indices = [t["index"] for t in plan["tasks"] if t["kind"] == "order"]
    for index in indices:
        execute(path, index)
    report_diagnostics(path, tmp_path / "report", plots=False)
    with (tmp_path / "report/order_differences.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 4 * 2 * 3
    assert all(float(row["original_minus_canonical_pp"]) == 0 for row in rows)


def test_report_is_portable_with_only_small_metadata(fixture, tmp_path):
    import shutil
    path = prepare_diagnostics(**fixture)
    execute(path, 0)
    downloaded = tmp_path / "downloaded"
    shutil.copytree(path.parent, downloaded)
    summary = report_diagnostics(downloaded / "plan.json", tmp_path / "report", plots=False)
    assert summary["completed_tasks"] == 1


def test_initial_weight_audit_preserves_rng_and_matches_order_variants(fixture, monkeypatch):
    import torch
    import al_pipeline.astronomy.astrominn.inference.astrominn_vit as inference
    states = []
    def toy_model(**kwargs):
        model = torch.nn.Module()
        model.vit_tower = torch.nn.Linear(2, 5)
        model.vit_tower.register_buffer("scalar_counter", torch.tensor(0))
        states.append(torch.get_rng_state().clone())
        return model
    monkeypatch.setattr(inference, "load_astrominn_vit_model", toy_model)
    class AuditedTrainer(FakeTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.factory = kwargs["model_factory"]
        def train(self, request):
            torch.manual_seed(request.seed)
            self.factory(False, None)
            assert torch.equal(torch.get_rng_state(), states[-1])
            return super().train(request)
    path = prepare_diagnostics(**fixture)
    plan = load_plan(path)
    hashes = []
    for task in plan["tasks"]:
        if task["kind"] == "order" and task["training_seed"] == 135:
            result = run_task(path, task["index"], device="cpu", trainer_class=AuditedTrainer,
                              evaluator_class=FakeEvaluator)
            hashes.append(json.loads(result.read_text())["initialization"]["vit_state_sha256"])
    assert len(hashes) == 5
    assert len(set(hashes)) == 1


def test_report_rejects_unmatched_initial_weights(fixture, tmp_path):
    path = prepare_diagnostics(**fixture)
    plan = load_plan(path)
    indices = [t["index"] for t in plan["tasks"]
               if t["kind"] == "order" and t["training_seed"] == 135][:2]
    for i, index in enumerate(indices):
        result = execute(path, index)
        payload = json.loads(result.read_text())
        payload["initialization"] = {"vit_state_sha256": str(i)}
        write_json(result, payload)
    with pytest.raises(ValueError, match="different initial ViT"):
        report_diagnostics(path, tmp_path / "report", plots=False)



def test_registry_aliases_and_missing_replay_survey_preserve_order_controls(fixture):
    registry = DatasetRegistry.from_toml(fixture["registry_path"]).resolve_profile("astrominn_ztf")
    manifest = read_manifest(registry.dataset.train_manifest, registry.dataset)
    aliases = {"SN I": "SN Ia", "TDE": "Tidal Disruption Event", "CV": "Cataclysmic"}
    rows = [{**row, "target": aliases.get(row["target"], row["target"]), "survey": ""}
            for row in manifest.rows]
    write_manifest(registry.dataset.train_manifest, rows)
    path = prepare_diagnostics(**fixture)
    tasks = [t for t in load_plan(path)["tasks"] if t["kind"] == "order"]
    assert len({t["content_sha256"] for t in tasks}) == 1
    canonical = read_manifest(Path(tasks[0]["manifest"]), registry.dataset)
    assert canonical.content_sha256 == manifest.content_sha256


def test_registry_changed_class_still_rejected(fixture):
    registry = DatasetRegistry.from_toml(fixture["registry_path"]).resolve_profile("astrominn_ztf")
    manifest = read_manifest(registry.dataset.train_manifest, registry.dataset)
    rows = [{**row, "target": "CV" if row["target"] == "AGN" else row["target"]}
            for row in manifest.rows]
    write_manifest(registry.dataset.train_manifest, rows)
    with pytest.raises(ValueError, match="differs from the historical dataset"):
        prepare_diagnostics(**fixture)
