"""Synthetic, model-free tests for independent epoch-checkpoint selection."""

import json
from pathlib import Path

import pytest

from al_pipeline.core.training.checkpoint_selection import (
    CheckpointTrainingIdentity,
    MetricCriterion,
    WithinCycleCheckpointKeeper,
)


def identity(**changes):
    args = dict(
        run_id="r1", cycle=3, seed=135, labeled_set_version="labels-L3",
        validation_identity="sha256:fixed_validation", evaluation_level="mixed",
        initial_checkpoint="/runs/c2/model.pt",
    )
    args.update(changes)
    return CheckpointTrainingIdentity(**args)


def saver(count):
    def write(path: Path):
        count.append(path)
        path.write_bytes(b"synthetic-model-weights")
    return write


def sample(tmp_path, *, resume=False, **changes):
    return WithinCycleCheckpointKeeper(tmp_path / "fit", identity=identity(**changes), resume=resume)


def test_first_epoch_saves_once_for_two_selections(tmp_path):
    keeper = sample(tmp_path)
    saved = []
    update = keeper.consider(epoch=1, validation_metrics={
        "validation_loss": 0.8, "validation_macro_f1": 0.25,
    }, save_checkpoint=saver(saved))
    assert len(saved) == 1
    assert update.improved == ("validation_loss", "validation_macro_f1")
    assert keeper.selected("validation_loss").path == keeper.selected("validation_macro_f1").path
    assert Path(keeper.selected("validation_loss").path).is_file()


def test_independent_metric_best_epochs(tmp_path):
    keeper = sample(tmp_path)
    calls = []
    for epoch, loss, f1 in [(1, 0.8, 0.25), (2, 0.7, 0.24), (3, 0.72, 0.5)]:
        keeper.consider(epoch=epoch, validation_metrics={
            "validation_loss": loss, "validation_macro_f1": f1,
        }, save_checkpoint=saver(calls))
    assert len(calls) == 3
    assert keeper.selected("validation_loss").epoch == 2
    assert keeper.selected("validation_macro_f1").epoch == 3
    assert {p.name for p in (tmp_path / "fit/artifacts").iterdir()} == {
        "epoch_000002.pt", "epoch_000003.pt"
    }


def test_unimproved_epoch_never_invokes_save(tmp_path):
    keeper = sample(tmp_path)
    calls = []
    keeper.consider(epoch=1, validation_metrics={
        "validation_loss": 0.5, "validation_macro_f1": 0.6,
    }, save_checkpoint=saver(calls))
    update = keeper.consider(epoch=2, validation_metrics={
        "validation_loss": 0.8, "validation_macro_f1": 0.2,
    }, save_checkpoint=saver(calls))
    assert len(calls) == 1
    assert update.checkpoint_path is None and not update.improved
    assert keeper.last_epoch == 2


def test_ties_keep_previous(tmp_path):
    keeper = sample(tmp_path)
    for epoch in (1, 2):
        keeper.consider(epoch=epoch, validation_metrics={
            "validation_loss": 0.5, "validation_macro_f1": 0.6,
        }, save_checkpoint=saver([]))
    assert all(item.epoch == 1 for item in keeper.best.values())


def test_resume_preserves_selections(tmp_path):
    old = sample(tmp_path)
    old.consider(epoch=1, validation_metrics={
        "validation_loss": 0.8, "validation_macro_f1": 0.3,
    }, save_checkpoint=saver([]))
    new = sample(tmp_path, resume=True)
    assert new.best == old.best
    assert new.last_epoch == 1
    new.consider(epoch=2, validation_metrics={
        "validation_loss": 0.7, "validation_macro_f1": 0.31,
    }, save_checkpoint=saver([]))
    assert new.last_epoch == 2
    assert not Path(old.selected("validation_loss").path).exists()


def test_resume_rejects_mismatched_validation_and_labels(tmp_path):
    keeper = sample(tmp_path)
    keeper.consider(epoch=1, validation_metrics={
        "validation_loss": 0.8, "validation_macro_f1": 0.3,
    }, save_checkpoint=saver([]))
    for changes in [dict(validation_identity="wrong"), dict(labeled_set_version="L2"), dict(seed=246)]:
        with pytest.raises(ValueError, match="provenance"):
            sample(tmp_path, resume=True, **changes)


def test_existing_state_cannot_be_overwritten_implicitly(tmp_path):
    keeper = sample(tmp_path)
    keeper.consider(epoch=1, validation_metrics={
        "validation_loss": 0.8, "validation_macro_f1": 0.3,
    }, save_checkpoint=saver([]))
    with pytest.raises(FileExistsError):
        sample(tmp_path)


def test_checkpoint_failure_does_not_advance_or_clobber(tmp_path):
    keeper = sample(tmp_path)
    keeper.consider(epoch=1, validation_metrics={
        "validation_loss": 0.8, "validation_macro_f1": 0.3,
    }, save_checkpoint=saver([]))
    before = keeper.best
    def broken(path):
        path.write_bytes(b"partial")
        raise RuntimeError("save failure")
    with pytest.raises(RuntimeError):
        keeper.consider(epoch=2, validation_metrics={
            "validation_loss": 0.5, "validation_macro_f1": 0.7,
        }, save_checkpoint=broken)
    assert keeper.last_epoch == 1
    assert keeper.best == before
    assert len(list((tmp_path / "fit/artifacts").iterdir())) == 1


def test_empty_checkpoint_is_rejected(tmp_path):
    keeper = sample(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        keeper.consider(epoch=1, validation_metrics={
            "validation_loss": 0.8, "validation_macro_f1": 0.3,
        }, save_checkpoint=lambda path: None)
    assert keeper.last_epoch == 0


def test_missing_metrics_and_nan_fail_before_saving(tmp_path):
    keeper = sample(tmp_path)
    with pytest.raises(KeyError):
        keeper.consider(epoch=1, validation_metrics={"validation_loss": 0.4},
                        save_checkpoint=lambda path: pytest.fail("called"))
    with pytest.raises(ValueError, match="finite"):
        keeper.consider(epoch=1, validation_metrics={
            "validation_loss": float("nan"), "validation_macro_f1": 0.3,
        }, save_checkpoint=lambda path: pytest.fail("called"))
    assert keeper.last_epoch == 0


def test_monotonic_epoch_and_missing_artifact_detection(tmp_path):
    keeper = sample(tmp_path)
    keeper.consider(epoch=2, validation_metrics={
        "validation_loss": 0.8, "validation_macro_f1": 0.3,
    }, save_checkpoint=saver([]))
    with pytest.raises(ValueError, match="strictly"):
        keeper.consider(epoch=2, validation_metrics={
            "validation_loss": 0.7, "validation_macro_f1": 0.4,
        }, save_checkpoint=saver([]))
    Path(keeper.selected("validation_loss").path).unlink()
    with pytest.raises(FileNotFoundError):
        sample(tmp_path, resume=True)


def test_reject_duplicate_criteria_bad_identity_and_extension(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        WithinCycleCheckpointKeeper(tmp_path, identity=identity(), criteria=(
            MetricCriterion("loss", "min"), MetricCriterion("loss", "min"),
        ))
    with pytest.raises(ValueError, match="extension"):
        WithinCycleCheckpointKeeper(tmp_path, identity=identity(), suffix="../bad")
    with pytest.raises(ValueError, match="seed"):
        identity(seed=-1)


def test_single_metric_selection(tmp_path):
    keeper = WithinCycleCheckpointKeeper(
        tmp_path / "single", identity=identity(),
        criteria=(MetricCriterion("macro_f1", "max"),), suffix=".weights",
    )
    keeper.consider(epoch=1, validation_metrics={"macro_f1": 0.32}, save_checkpoint=saver([]))
    assert keeper.selected("macro_f1").path.endswith(".weights")


def test_manifest_records_fit_and_independent_choices(tmp_path):
    keeper = sample(tmp_path)
    keeper.consider(epoch=1, validation_metrics={
        "validation_loss": 0.8, "validation_macro_f1": 0.3,
    }, save_checkpoint=saver([]))
    d = json.loads(keeper.manifest_path.read_text())
    assert d["identity"]["labeled_set_version"] == "labels-L3"
    assert len(d["best"]) == 2
    assert d["schema_version"] == 1


def test_validation_metric_granularity_must_match_fit(tmp_path):
    with pytest.raises(ValueError, match="evaluation level conflicts"):
        sample(tmp_path, evaluation_level="sample")


def test_explicit_object_level_only_f1_can_be_selected(tmp_path):
    keeper = WithinCycleCheckpointKeeper(tmp_path / "object-only", identity=identity(evaluation_level="object"),
        criteria=(MetricCriterion("object_macro_f1", "max", "object"),))
    keeper.consider(epoch=1, validation_metrics={"object_macro_f1": 0.4}, save_checkpoint=saver([]))
    assert keeper.selected("object_macro_f1").epoch == 1
