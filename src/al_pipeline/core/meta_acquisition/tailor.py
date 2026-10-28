"""Multiclass TAILOR (Zhang et al., NeurIPS 2023) and a random-meta control.

The controller never sees features, predictions, or unrevealed labels. It only
chooses an acquisition *algorithm* for each position in the next batch. After
the entire batch has been annotated, the true labels of the selected objects
update the chosen algorithms' Dirichlet posteriors.

This separates the published decision rule from the application-specific
implementation of the constituent acquisition algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_ARMS = (
    "entropy_diversity",
    "entropy_balanced_explore",
    "fisher_batch",
)


@dataclass(frozen=True, slots=True)
class TailorSettings:
    arms: tuple[str, ...] = DEFAULT_ARMS
    discount: float = 0.9
    prior_concentration: float = 1.0
    reward: str = "inverse_confirmed_class_count"

    def __post_init__(self) -> None:
        if len(self.arms) < 2 or len(set(self.arms)) != len(self.arms):
            raise ValueError("TAILOR requires at least two distinct acquisition arms")
        if any(not isinstance(name, str) or not name for name in self.arms):
            raise ValueError("Acquisition arm names must be nonempty strings")
        if not np.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError("TAILOR discount must lie in (0, 1]")
        if not np.isfinite(self.prior_concentration) or self.prior_concentration <= 0:
            raise ValueError("Dirichlet prior concentration must be positive")
        if self.reward != "inverse_confirmed_class_count":
            raise ValueError("Unsupported TAILOR reward function")


class TailorBandit:
    """Batched Thompson selection over multiclass label-frequency posteriors.

    The paper uses alpha_i <- gamma * alpha_i + acquired_class_counts_i
    (gamma=0.9 in its experiments). Posterior updates occur only *after*
    the corresponding acquired labels become available.
    """

    def __init__(self, *, class_names: Sequence[str], settings: TailorSettings | None = None):
        self.settings = TailorSettings() if settings is None else settings
        self.class_names = tuple(class_names)
        if len(self.class_names) < 2 or len(self.class_names) != len(set(self.class_names)):
            raise ValueError("TAILOR requires at least two distinct, ordered classes")
        self.class_index = {name: i for i, name in enumerate(self.class_names)}
        self.alpha = np.full(
            (len(self.settings.arms), len(self.class_names)),
            self.settings.prior_concentration,
            dtype=np.float64,
        )
        self.rounds_seen = 0

    def reward_weights(self, confirmed_counts: Sequence[int]) -> np.ndarray:
        counts = np.asarray(confirmed_counts, dtype=np.float64)
        if counts.shape != (len(self.class_names),):
            raise ValueError("Confirmed object counts must follow the class order")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("Confirmed class counts must be finite and nonnegative")
        if not np.all(counts == np.floor(counts)):
            raise ValueError("Confirmed class counts must be integers")
        return 1.0 / (len(self.class_names) * np.maximum(counts, 1.0))

    def choose(
        self,
        *,
        batch_size: int,
        confirmed_counts: Sequence[int],
        rng: np.random.Generator,
        method: str,
    ) -> tuple[str, ...]:
        """Choose all arm assignments without observing new batch labels."""
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Acquisition batch_size must be a positive integer")
        if method not in {"tailor_diversity", "uniform_mixture"}:
            raise ValueError("Unknown meta-acquisition controller")
        weights = self.reward_weights(confirmed_counts)
        if method == "uniform_mixture":
            indices = rng.integers(len(self.settings.arms), size=batch_size)
        else:
            indices = np.empty(batch_size, dtype=np.int64)
            for step in range(batch_size):
                # Separate posterior sample for each arm at *each* batch position.
                utilities = np.asarray(
                    [float(rng.dirichlet(row) @ weights) for row in self.alpha]
                )
                indices[step] = int(np.argmax(utilities))
        return tuple(self.settings.arms[int(i)] for i in indices)

    def update(self, assignments: Sequence[str], revealed_labels: Sequence[str]) -> None:
        """Discount and update only from the genuinely acquired, revealed labels."""
        assignments = tuple(assignments)
        labels = tuple(revealed_labels)
        if not assignments or len(assignments) != len(labels):
            raise ValueError("Arm assignments and acquired labels must align and be nonempty")
        arm_index = {name: i for i, name in enumerate(self.settings.arms)}
        counts = np.zeros_like(self.alpha)
        for arm, label in zip(assignments, labels, strict=True):
            if arm not in arm_index:
                raise ValueError(f"Unknown acquired arm {arm!r}")
            if label not in self.class_index:
                raise ValueError(f"Unknown revealed class {label!r}")
            counts[arm_index[arm], self.class_index[label]] += 1.0
        self.alpha = self.settings.discount * self.alpha + counts
        if not np.all(np.isfinite(self.alpha)) or np.any(self.alpha <= 0):
            raise ValueError("Invalid updated Dirichlet posterior")
        self.rounds_seen += 1

    def history_summary(self) -> dict[str, object]:
        return {
            "arms": list(self.settings.arms),
            "class_names": list(self.class_names),
            "alpha": self.alpha.tolist(),
            "rounds_seen": self.rounds_seen,
            "discount": self.settings.discount,
            "prior_concentration": self.settings.prior_concentration,
            "reward": self.settings.reward,
        }
