"""Model-independent training contracts for adaptive-learning cycles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """Inputs required to train one model for one AL cycle.

    The request contains paths and experiment identity only. Model-specific
    hyperparameters belong to the concrete training backend so the generic
    active-learning coordinator does not need to understand them.
    """

    train_manifest: Path
    validation_manifest: Path
    output_dir: Path
    run_id: str
    cycle: int
    strategy: str
    seed: int
    initial_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "train_manifest",
            Path(self.train_manifest),
        )
        object.__setattr__(
            self,
            "validation_manifest",
            Path(self.validation_manifest),
        )
        object.__setattr__(
            self,
            "output_dir",
            Path(self.output_dir),
        )

        if self.initial_checkpoint is not None:
            object.__setattr__(
                self,
                "initial_checkpoint",
                Path(self.initial_checkpoint),
            )

        if not self.run_id:
            raise ValueError("run_id must be a non-empty string.")

        if self.cycle < 0:
            raise ValueError("cycle must be non-negative.")

        if not self.strategy:
            raise ValueError("strategy must be a non-empty string.")

        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Durable outputs produced by one training request."""

    checkpoint_path: Path
    history_path: Path
    summary_path: Path
    best_epoch: int
    epochs_completed: int
    best_validation_loss: float
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "checkpoint_path",
            "history_path",
            "summary_path",
        ):
            object.__setattr__(
                self,
                field_name,
                Path(getattr(self, field_name)),
            )

        if self.best_epoch <= 0:
            raise ValueError("best_epoch must be positive.")

        if self.epochs_completed < self.best_epoch:
            raise ValueError("epochs_completed cannot be smaller than best_epoch.")

        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative.")


class TrainingBackend(Protocol):
    """Interface implemented by model-specific training adapters."""

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Train one cycle and return its durable artifacts."""

        ...


class ResumableTrainingBackend(TrainingBackend, Protocol):
    """Training backend that can recover an already completed request."""

    def completed_result(
        self,
        request: TrainingRequest,
    ) -> TrainingResult | None:
        """Return a validated completed result, or ``None`` if absent."""

        ...


__all__ = [
    "TrainingBackend",
    "ResumableTrainingBackend",
    "TrainingRequest",
    "TrainingResult",
]
