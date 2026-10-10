"""Soft class-aware entropy using only already confirmed label counts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from al_pipeline.core.scoring.base import Scorer
from al_pipeline.core.scoring.entropy import EntropyScorer
from al_pipeline.core.types import ClassificationPredictions


@dataclass(frozen=True, slots=True)
class ClassBalancedInput:
    """Counts must follow the classifier's probability-column order.

    One count represents one labeled selection unit, not one observation.
    Candidate true labels must never be used to construct these counts.
    """

    predictions: ClassificationPredictions
    labeled_counts: np.ndarray

    def __post_init__(self) -> None:
        counts = np.asarray(self.labeled_counts, dtype=float)
        if counts.shape != (self.predictions.values.shape[1],):
            raise ValueError("labeled_counts must match the probability columns.")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("labeled_counts must be finite and non-negative.")
        if np.any(counts != np.floor(counts)):
            raise ValueError("labeled_counts must contain integer object counts.")
        object.__setattr__(self, "labeled_counts", counts.copy())


class ClassBalancedEntropyScorer(Scorer[ClassBalancedInput]):
    """Entropy times a bounded, probability-weighted rarity bonus.

    w_c = clip((mean(n + alpha)/(n_c + alpha))**beta, 1, cap)
    score(x) = H(p(x)) * (1 + strength * sum_c p_c(x)*(w_c - 1))

    This is a transparent heuristic, not a calibrated expected utility or
    a guarantee of class quotas. Every nonzero probability participates.
    """

    def __init__(self, *, alpha=1.0, beta=0.5, strength=1.0, cap=3.0):
        values = np.asarray([alpha, beta, strength, cap], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Class-balance parameters must be finite.")
        if alpha <= 0 or beta < 0 or strength < 0 or cap < 1:
            raise ValueError("Require alpha > 0, beta/strength >= 0, cap >= 1.")
        self.alpha, self.beta, self.strength, self.cap = values.tolist()

    def class_weights(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=float)
        if counts.ndim != 1 or counts.size < 2:
            raise ValueError("At least two class counts are required.")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("Invalid labeled class counts.")
        smoothed = counts + self.alpha
        # Evaluate in log space so an extreme exponent cannot overflow.
        exponent = np.clip(
            self.beta * (np.log(smoothed.mean()) - np.log(smoothed)),
            0,
            np.log(self.cap),
        )
        return np.exp(exponent)

    def score(self, data: ClassBalancedInput) -> np.ndarray:
        probabilities = data.predictions.as_probabilities()
        bonus = probabilities @ (self.class_weights(data.labeled_counts) - 1)
        scores = EntropyScorer().score(data.predictions) * (1 + self.strength * bonus)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Class-balanced scoring produced non-finite values.")
        return scores


__all__ = ["ClassBalancedInput", "ClassBalancedEntropyScorer"]
