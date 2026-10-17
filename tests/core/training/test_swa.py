"""Synthetic-only tests: optional PyTorch SWA point-estimate checkpointing."""

import pytest

torch = pytest.importorskip("torch")  # PyTorch is optional for core users.

from al_pipeline.core.training.checkpoint_selection import CheckpointTrainingIdentity
from al_pipeline.integrations.pytorch.swa import (
    SWAConfig,
    SWATracker,
    load_swa_checkpoint,
)


def identity(**overrides):
    values = dict(
        run_id="toy-al", cycle=3, seed=135, labeled_set_version="L3-sha256",
        validation_identity="fixed-val-sha256", evaluation_level="mixed",
        initial_checkpoint="/immutable/prev-cycle.pt",
    )
    values.update(overrides)
    return CheckpointTrainingIdentity(**values)


def simple(weight=0.0):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight)
    return model


def set_weight(model, value):
    with torch.no_grad():
        model.weight.fill_(value)


@pytest.mark.parametrize("args,exc", [
    ({"start_epoch": 0}, ValueError),
    ({"start_epoch": True}, ValueError),
    ({"update_every": 0}, ValueError),
    ({"update_every": 1.2}, ValueError),
    ({"anneal_epochs": -1}, ValueError),
    ({"anneal_strategy": "parabolic"}, ValueError),
    ({"swa_lr": 0}, ValueError),
    ({"swa_lr": float("nan")}, ValueError),
    ({"swa_lr": float("inf")}, ValueError),
    ({"swa_lr": True}, TypeError),
    ({"enabled": "yes"}, TypeError),
])
def test_bad_config(args, exc):
    with pytest.raises(exc):
        SWAConfig(**args)


def test_swa_disabled_is_noop_and_normal_scheduler_used():
    model = simple()
    tracker = SWATracker(model)  # default off
    calls = []

    class Counter:
        def step(self):
            calls.append("base")

    assert tracker.averaged_model is None
    for epoch in range(1, 4):
        status = tracker.end_epoch(epoch, base_scheduler=Counter())
        assert not status.swa_active
        assert not status.averaged
    assert tracker.n_averaged == 0
    assert calls == ["base"] * 3
    assert not tracker.ready


def test_swa_requires_opt_in_and_optimizer_for_swalr():
    with pytest.raises(ValueError, match="optimizer"):
        SWATracker(simple(), config=SWAConfig(enabled=True, swa_lr=0.1))


def test_exact_parameter_average_with_schedule_and_no_bn(tmp_path):
    model = simple(0)
    tracker = SWATracker(model, config=SWAConfig(enabled=True, start_epoch=2, update_every=2))
    expected = [(1, 0, False), (2, 2, True), (3, 2, False), (4, 4, True), (5, 4, False), (6, 6, True)]
    for epoch, value, averaged in expected:
        set_weight(model, epoch)
        result = tracker.end_epoch(epoch)
        assert result.averaged is averaged
        assert tracker.n_averaged == len(tracker.averaged_epochs)
        if averaged:
            assert tracker.averaged_epochs[-1] == value
    assert tracker.averaged_epochs == [2, 4, 6]
    assert tracker.ready
    assert tracker.averaged_model.module.weight.item() == pytest.approx(4.0)
    assert model.weight.item() == pytest.approx(6.0)  # source untouched
    path = tracker.save_checkpoint(tmp_path / "swa.pt", identity=identity())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["checkpoint_type"] == "swa"
    assert payload["swa"]["averaged_epochs"] == [2, 4, 6]
    assert payload["swa"]["n_averaged"] == 3
    assert payload["swa"]["batchnorm_refreshed"] is False
    assert payload["identity"]["labeled_set_version"] == "L3-sha256"
    assert payload["model_state"]["weight"].item() == pytest.approx(4)
    assert not (tmp_path / "swa.pt.pending").exists()

    loaded = simple(-10)
    provenance = load_swa_checkpoint(path, model=loaded, expected_identity=identity())
    assert loaded.weight.item() == pytest.approx(4)
    assert provenance["n_averaged"] == 3


def test_swa_ends_before_start_has_no_checkpoint(tmp_path):
    tracker = SWATracker(simple(), config=SWAConfig(enabled=True, start_epoch=8))
    for epoch in range(1, 4):
        tracker.end_epoch(epoch)
    assert tracker.ready is False
    with pytest.raises(RuntimeError, match="No SWA weights"):
        tracker.save_checkpoint(tmp_path / "none.pt", identity=identity())
    assert not (tmp_path / "none.pt").exists()


def test_swa_does_not_change_source_model_and_validates_epochs():
    model = simple()
    tracker = SWATracker(model, config=SWAConfig(enabled=True, start_epoch=1))
    tracker.end_epoch(1)
    with pytest.raises(ValueError, match="strictly"):
        tracker.end_epoch(1)
    with pytest.raises(ValueError, match="consecutive"):
        tracker.end_epoch(3)
    set_weight(model, 3)
    tracker.end_epoch(2)
    assert tracker.averaged_model.module.weight.item() == pytest.approx(1.5)
    assert model.weight.item() == 3


def test_swalr_switches_without_stepping_base_scheduler_twice():
    model = simple()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    tracker = SWATracker(model, config=SWAConfig(
        enabled=True, start_epoch=3, swa_lr=0.05, anneal_epochs=2,
    ), optimizer=optimizer)
    calls = []

    class BaseSchedule:
        def step(self):
            calls.append("base")

    base = BaseSchedule()
    statuses = []
    for epoch in range(1, 6):
        optimizer.zero_grad()
        model(torch.ones(4, 1)).mean().backward()
        optimizer.step()
        statuses.append(tracker.end_epoch(epoch, base_scheduler=base))
    assert calls == ["base", "base"]
    assert [status.scheduler_used for status in statuses] == [
        "base", "base", "swa", "swa", "swa",
    ]
    assert tracker.n_averaged == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_without_swa_lr_base_scheduler_keeps_stepping():
    t = SWATracker(simple(), config=SWAConfig(enabled=True, start_epoch=2))
    calls = []

    class Counter:
        def step(self):
            calls.append(1)

    for i in (1, 2, 3):
        assert t.end_epoch(i, base_scheduler=Counter()).scheduler_used == "base"
    assert sum(calls) == 3


def test_batchnorm_requires_recalibration(tmp_path):
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    tracker = SWATracker(m, config=SWAConfig(enabled=True, start_epoch=1))
    tracker.end_epoch(1)
    with pytest.raises(ValueError, match="BatchNorm"):
        tracker.save_checkpoint(tmp_path / "swa.pt", identity=identity())
    assert not (tmp_path / "swa.pt").exists()
    loader = [torch.randn(8, 2) * 2 + 5 for _ in range(3)]
    tracker.save_checkpoint(tmp_path / "swa.pt", identity=identity(), bn_loader=loader)
    payload = torch.load(tmp_path / "swa.pt", weights_only=True)
    assert payload["swa"]["batchnorm_refreshed"]
    assert payload["model_state"]["1.num_batches_tracked"].item() == 3
    assert torch.all(torch.isfinite(payload["model_state"]["1.running_mean"]))


def test_batchnorm_multimodal_callback_supported(tmp_path):
    class TwoInputs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2)
        def forward(self, metadata, image):
            return self.bn(metadata + image)

    t = SWATracker(TwoInputs(), config=SWAConfig(enabled=True, start_epoch=1))
    t.end_epoch(1)
    calls = []

    def calibrate(model):
        assert isinstance(model, TwoInputs)
        model.train()
        model.bn.reset_running_stats()
        with torch.no_grad():
            for _ in range(3):
                model(torch.randn(8, 2), torch.randn(8, 2))
        calls.append(True)

    with pytest.raises(ValueError, match="either bn_loader or bn_refresh"):
        t.save_checkpoint(tmp_path / "a.pt", identity=identity(),
                          bn_loader=[torch.ones(5, 2)], bn_refresh=calibrate)
    t.save_checkpoint(tmp_path / "a.pt", identity=identity(), bn_refresh=calibrate)
    assert calls == [True]
    p = torch.load(tmp_path / "a.pt", weights_only=True)
    assert p["model_state"]["bn.num_batches_tracked"].item() == 3


def test_batchnorm_refresh_invalidated_by_later_averaging(tmp_path):
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    t = SWATracker(m, config=SWAConfig(enabled=True, start_epoch=1))
    t.end_epoch(1)
    t.save_checkpoint(tmp_path / "first.pt", identity=identity(), bn_loader=[torch.randn(4, 2)])
    t.end_epoch(2)
    with pytest.raises(ValueError, match="BatchNorm"):
        t.save_checkpoint(tmp_path / "second.pt", identity=identity())


def test_restore_rejects_other_training_identity_and_architecture(tmp_path):
    t = SWATracker(simple(2), config=SWAConfig(enabled=True, start_epoch=1))
    t.end_epoch(1)
    path = t.save_checkpoint(tmp_path / "swa.pt", identity=identity())
    with pytest.raises(ValueError, match="identity"):
        load_swa_checkpoint(path, model=simple(), expected_identity=identity(seed=246))
    with pytest.raises(RuntimeError):
        load_swa_checkpoint(path, model=torch.nn.Linear(1, 2), expected_identity=identity())
    with pytest.raises(FileExistsError, match="overwrite"):
        t.save_checkpoint(path, identity=identity())


def test_state_dict_roundtrip_resumes_averaging():
    model = simple()
    conf = SWAConfig(enabled=True, start_epoch=2, update_every=1)
    first = SWATracker(model, config=conf)
    set_weight(model, 1)
    first.end_epoch(1)
    set_weight(model, 2)
    first.end_epoch(2)
    saved = first.state_dict()

    copied_model = simple(2)  # source model must also be restored by training backend
    restored = SWATracker(copied_model, config=conf)
    restored.load_state_dict(saved)
    assert restored.last_epoch == 2
    assert restored.averaged_epochs == [2]
    assert restored.n_averaged == 1
    set_weight(copied_model, 4)
    restored.end_epoch(3)
    assert restored.n_averaged == 2
    assert restored.averaged_model.module.weight.item() == pytest.approx(3)


def test_state_dict_detects_invalid_resume_state():
    config = SWAConfig(enabled=True, start_epoch=1)
    tracker = SWATracker(simple(), config=config)
    tracker.end_epoch(1)
    original = tracker.state_dict()
    tampered = dict(original, averaged_epochs=[2])
    other = SWATracker(simple(), config=config)
    with pytest.raises(ValueError, match="epochs"):
        other.load_state_dict(tampered)
    assert other.last_epoch == 0
    with pytest.raises(ValueError, match="config"):
        other.load_state_dict(dict(original, config=dict(original["config"], start_epoch=5)))
    with pytest.raises(ValueError, match="n_averaged"):
        other.load_state_dict(dict(original, averaged_model=dict(original["averaged_model"], n_averaged=torch.tensor(4))))


def test_swalr_state_roundtrip_with_optimizer_state():
    c = SWAConfig(enabled=True, start_epoch=1, swa_lr=0.04, anneal_epochs=3)
    model = simple(1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    original = SWATracker(model, config=c, optimizer=opt)
    for epoch in (1, 2):
        opt.zero_grad()
        model(torch.ones(2, 1)).mean().backward()
        opt.step()
        original.end_epoch(epoch)
    state = original.state_dict()
    other_model = simple()
    other_model.load_state_dict(model.state_dict())
    other_opt = torch.optim.SGD(other_model.parameters(), lr=0.1)
    other_opt.load_state_dict(opt.state_dict())
    other = SWATracker(other_model, config=c, optimizer=other_opt)
    other.load_state_dict(state)
    assert other.n_averaged == 2
    assert other.last_epoch == 2
    assert other.swa_scheduler.state_dict() == original.swa_scheduler.state_dict()


def test_atomic_export_cleanup_on_failure(tmp_path, monkeypatch):
    t = SWATracker(simple(), config=SWAConfig(enabled=True, start_epoch=1))
    t.end_epoch(1)
    original_save = torch.save

    def broken(*args, **kwargs):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError, match="filesystem"):
        t.save_checkpoint(tmp_path / "swa.pt", identity=identity())
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(torch, "save", original_save)
    t.save_checkpoint(tmp_path / "swa.pt", identity=identity())
    assert (tmp_path / "swa.pt").is_file()


def test_pure_core_training_import_remains_separate():
    from al_pipeline.core.training import TrainingRequest
    assert TrainingRequest.__name__ == "TrainingRequest"


def test_state_dict_is_an_immutable_snapshot_for_later_saving():
    model = simple(2)
    tracker = SWATracker(model, config=SWAConfig(enabled=True, start_epoch=1))
    tracker.end_epoch(1)
    snapshot = tracker.state_dict()
    set_weight(model, 6)
    tracker.end_epoch(2)
    assert snapshot["averaged_model"]["module.weight"].item() == pytest.approx(2)
    assert tracker.averaged_model.module.weight.item() == pytest.approx(4)
