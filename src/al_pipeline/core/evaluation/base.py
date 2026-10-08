"""Model-evaluation contracts for adaptive-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """Metrics and artifact counts for one fixed evaluation split."""

    split: str
    sample_count: int
    object_count: int
    sample_metrics: dict[str, float]
    object_metrics: dict[str, float]

    def __post_init__(self) -> None:
        if not self.split:
            raise ValueError("split must be a non-empty string.")

        if self.sample_count <= 0 or self.object_count <= 0:
            raise ValueError("sample_count and object_count must be positive.")


@dataclass(frozen=True, slots=True)
class ModelEvaluationResult:
    """Durable evaluation output produced for one model checkpoint."""

    checkpoint_path: Path
    summary_path: Path
    splits: tuple[SplitEvaluation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_path",
            Path(self.checkpoint_path),
        )
        object.__setattr__(
            self,
            "summary_path",
            Path(self.summary_path),
        )

        if not self.splits:
            raise ValueError("Evaluation must contain at least one split.")

        names = [split.split for split in self.splits]

        if len(names) != len(set(names)):
            raise ValueError("Evaluation split names must be unique.")

    def split(self, name: str) -> SplitEvaluation:
        """Return a named split or raise a descriptive error."""

        for result in self.splits:
            if result.split == name:
                return result

        raise KeyError(f"Evaluation does not contain split {name!r}.")


class ModelEvaluator(Protocol):
    """Interface implemented by fixed-split evaluation backends."""

    def evaluate(
        self,
        *,
        checkpoint_path: Path,
        output_dir: Path,
    ) -> ModelEvaluationResult:
        """Evaluate one checkpoint and persist reproducible artifacts."""

        ...

    def completed_result(
        self,
        *,
        checkpoint_path: Path,
        output_dir: Path,
    ) -> ModelEvaluationResult | None:
        """Recover a validated completed evaluation when available."""

        ...


__all__ = [
    "ModelEvaluationResult",
    "ModelEvaluator",
    "SplitEvaluation",
]
