"""Expected last-layer Fisher-trace scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from al_pipeline.core.scoring.base import Scorer
from al_pipeline.core.types import ClassificationPredictions


@dataclass(frozen=True, slots=True)
class FisherScoringInput:
    """Classifier predictions and final-layer input representations.

    ``embeddings`` must contain the representation supplied directly to the
    model's final linear classification layer. Rows must correspond exactly
    to the rows in ``predictions``.
    """

    predictions: ClassificationPredictions
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        embeddings = np.asarray(self.embeddings)

        if embeddings.ndim != 2:
            raise ValueError(
                "Fisher embeddings must be two-dimensional with shape (n_objects, embedding_dim)."
            )

        if embeddings.shape[0] != self.predictions.values.shape[0]:
            raise ValueError(
                "Fisher embeddings and predictions must contain the same number of objects."
            )

        if embeddings.shape[1] < 1:
            raise ValueError("Fisher embeddings must contain at least one feature.")

        if not np.issubdtype(embeddings.dtype, np.number):
            raise TypeError("Fisher embeddings must be numeric.")

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Fisher embeddings must contain only finite values.")

        object.__setattr__(self, "embeddings", embeddings)


class FisherScorer(Scorer[FisherScoringInput]):
    """Score expected Fisher information for the final classifier layer.

    For softmax probabilities ``p`` and the representation ``h`` supplied to
    the final linear layer, the expected Fisher trace is

    ``(||h||^2 + 1) * (1 - sum_c p_c^2)``.

    The ``+ 1`` includes the classifier bias parameters. Set
    ``include_bias=False`` only for a final linear layer without a bias.
    The expectation over the unknown label uses the model's predictive class
    distribution. Larger values imply a larger expected final-layer gradient
    norm and therefore higher acquisition priority.

    This is deliberately a last-layer approximation. It does not construct a
    full-network Fisher matrix or claim to approximate a batch log-determinant
    information-gain objective.
    """

    def __init__(self, *, include_bias: bool = True) -> None:
        self.include_bias = bool(include_bias)

    def score(self, data: FisherScoringInput) -> np.ndarray:
        """Return one expected Fisher-trace score per candidate object."""

        probabilities = np.asarray(
            data.predictions.as_probabilities(),
            dtype=np.float64,
        )
        embeddings = np.asarray(
            data.embeddings,
            dtype=np.float64,
        )

        representation_norm_squared = np.einsum(
            "ij,ij->i",
            embeddings,
            embeddings,
        )

        if self.include_bias:
            representation_norm_squared = representation_norm_squared + 1.0

        class_covariance_trace = 1.0 - np.einsum(
            "ij,ij->i",
            probabilities,
            probabilities,
        )

        # Floating-point softmax arithmetic can produce a tiny negative value
        # for a numerically one-hot row. The theoretical quantity is >= 0.
        class_covariance_trace = np.maximum(
            class_covariance_trace,
            0.0,
        )
        scores = representation_norm_squared * class_covariance_trace

        if not np.all(np.isfinite(scores)):
            raise ValueError("Fisher scoring produced non-finite values.")

        return scores


__all__ = ["FisherScorer", "FisherScoringInput"]
