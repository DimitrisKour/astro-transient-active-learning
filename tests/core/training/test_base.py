"""Tests for model-independent training contracts."""

from pathlib import Path

import pytest

from al_pipeline.core.training.base import TrainingRequest, TrainingResult


def test_training_request_normalizes_paths() -> None:
    request = TrainingRequest(
        train_manifest="train.csv",
        validation_manifest="validation.csv",
        output_dir="run",
        run_id="example",
        cycle=1,
        strategy="entropy",
        seed=135,
        initial_checkpoint="initial.pth",
    )

    assert request.train_manifest == Path("train.csv")
    assert request.validation_manifest == Path("validation.csv")
    assert request.output_dir == Path("run")
    assert request.initial_checkpoint == Path("initial.pth")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", ""),
        ("cycle", -1),
        ("strategy", ""),
        ("seed", -1),
    ],
)
def test_training_request_rejects_invalid_identity(
    field: str,
    value: object,
) -> None:
    values = {
        "train_manifest": Path("train.csv"),
        "validation_manifest": Path("validation.csv"),
        "output_dir": Path("run"),
        "run_id": "example",
        "cycle": 1,
        "strategy": "random",
        "seed": 135,
    }
    values[field] = value

    with pytest.raises(ValueError):
        TrainingRequest(**values)


def test_training_result_validates_epoch_order() -> None:
    with pytest.raises(
        ValueError,
        match="epochs_completed",
    ):
        TrainingResult(
            checkpoint_path=Path("best.pth"),
            history_path=Path("history.json"),
            summary_path=Path("summary.json"),
            best_epoch=2,
            epochs_completed=1,
            best_validation_loss=1.0,
            elapsed_seconds=1.0,
        )
