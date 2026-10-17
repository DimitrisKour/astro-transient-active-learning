"""Model-independent BADGE acquisition with factored last-layer gradients.

The caller supplies one representative feature vector and prediction per
acquisition unit. For a linear softmax classifier with cross-entropy, the
pseudo-label gradient is outer(p - one_hot(argmax(p)), [features, 1]).

The exact gradient distances are computed from the two factors without
materializing the (n_candidates, n_classes * (feature_dim + 1)) matrix.
This assumes the supplied representation enters the final linear classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from al_pipeline.core.advanced_acquisition._common import (
    eligible_candidates,
    feature_matrix,
    validate_scores,
    validated_ids,
)
from al_pipeline.core.pool import Pool
from al_pipeline.core.scoring.base import Scorer
from al_pipeline.core.types import ClassificationPredictions, SelectionBatch


@dataclass(frozen=True, slots=True)
class BadgeInput:
    """Aligned predictions and inputs to the model's final linear classifier."""

    predictions: ClassificationPredictions
    embeddings: np.ndarray
    candidate_ids: tuple[str, ...]
    include_bias: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.predictions, ClassificationPredictions):
            raise TypeError("predictions must be ClassificationPredictions")
        features = feature_matrix(self.embeddings, name="BADGE embeddings")
        if features.shape[0] != self.predictions.values.shape[0]:
            raise ValueError("BADGE embeddings and predictions must align row-for-row")
        if type(self.include_bias) is not bool:
            raise TypeError("include_bias must be bool")
        self.predictions.as_probabilities()  # validate user-supplied probabilities
        object.__setattr__(self, "candidate_ids", validated_ids(
            self.candidate_ids, expected_rows=len(features), name="BADGE candidate IDs"
        ))
        object.__setattr__(self, "embeddings", features)

    def factors(self) -> tuple[np.ndarray, np.ndarray]:
        """Class residuals and linear-layer inputs, in candidate row order."""
        probabilities = np.asarray(self.predictions.as_probabilities(), dtype=np.float64)
        residuals = probabilities.copy()
        residuals[np.arange(len(residuals)), np.argmax(residuals, axis=1)] -= 1.0
        features = self.embeddings
        if self.include_bias:
            features = np.concatenate((features, np.ones((len(features), 1))), axis=1)
        return residuals, features

    def dense_gradient_embeddings(self) -> np.ndarray:
        """Reference representation; potentially memory-heavy for large pools."""
        residuals, features = self.factors()
        return (residuals[:, :, None] * features[:, None, :]).reshape(len(features), -1)


class BadgeGradientNormScorer(Scorer[BadgeInput]):
    """Squared pseudo-label gradient norm; used to initialize BADGE sampling."""

    def score(self, data: BadgeInput) -> np.ndarray:
        residuals, features = data.factors()
        scores = np.einsum("ij,ij->i", residuals, residuals) * np.einsum(
            "ij,ij->i", features, features
        )
        if not np.all(np.isfinite(scores)):
            raise ValueError("BADGE gradient norms overflowed: scale embeddings first")
        return np.maximum(scores, 0.0)


class BadgeSelector:
    """BADGE batch selection by seeded k-means++ over pseudo-label gradients.

    The first center is the eligible point with maximal gradient norm; further
    centers are sampled proportional to their squared distance from the nearest
    selected center. Distance ties and degenerate all-zero distributions fall
    back to original candidate order. The seed is recreated for every select()
    call, making retries deterministic for unchanged ordered inputs.

    This selector must be paired with BadgeGradientNormScorer on the SAME
    BadgeInput; scores are validated to prevent accidental entropy selection.
    No current or hidden labels are read. Selection is read-only on the pool.
    """

    def __init__(self, data: BadgeInput, *, seed: int) -> None:
        if not isinstance(data, BadgeInput):
            raise TypeError("data must be BadgeInput")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.data = data
        self.seed = seed

    def select(self, *, ids, scores, k: int, pool: Pool) -> SelectionBatch:
        norm_scores = BadgeGradientNormScorer().score(self.data)
        ids_arr, score_arr, eligible = eligible_candidates(
            ids, scores, k=k, pool=pool, expected_rows=len(norm_scores)
        )
        if tuple(ids_arr) != self.data.candidate_ids:
            raise ValueError("BADGE candidate IDs must match the representation row order")
        validate_scores(score_arr, norm_scores, method="BADGE")
        if k == 0:
            return SelectionBatch(
                ids=[], indices=np.empty(0, dtype=np.int64), scores=np.empty(0),
                metadata={"selector": "badge_kmeans_plusplus", "seed": self.seed,
                          "include_bias": self.data.include_bias, "selection_d2": []},
            )

        residuals, features = self.data.factors()
        # Work on eligible rows only; no ineligible example can become a center.
        r = residuals[eligible]
        h = features[eligible]
        rnorm = np.einsum("ij,ij->i", r, r)
        hnorm = np.einsum("ij,ij->i", h, h)
        norms = rnorm * hnorm
        if not np.all(np.isfinite(norms)):
            raise ValueError("BADGE gradient norms overflowed: scale embeddings first")

        first = int(np.argmax(norms))
        chosen: list[int] = []
        step_d2: list[float] = []
        minimum_d2 = np.full(len(eligible), np.inf)
        rng = np.random.default_rng(self.seed)

        for step in range(k):
            if step == 0:
                row = first
                priority = float(norms[row])
            else:
                # Do not select an existing center, including zero-distance duplicates.
                available = np.ones(len(eligible), dtype=bool)
                available[chosen] = False
                weights = np.where(available, np.maximum(minimum_d2, 0.0), 0.0)
                max_weight = float(np.max(weights))
                if max_weight == 0.0:
                    row = int(np.flatnonzero(available)[0])
                else:
                    scaled = weights / max_weight  # avoids summation overflow
                    scaled_sum = float(np.sum(scaled))
                    probs = scaled / scaled_sum
                    row = int(rng.choice(len(eligible), p=probs))
                priority = float(minimum_d2[row])
            chosen.append(row)
            step_d2.append(priority)
            # ||r_i outer h_i - r_j outer h_j||^2
            distances = (
                norms + norms[row]
                - 2.0 * (r @ r[row]) * (h @ h[row])
            )
            if not np.all(np.isfinite(distances)):
                raise ValueError("BADGE gradient distances overflowed: scale embeddings first")
            minimum_d2 = np.minimum(minimum_d2, np.maximum(distances, 0.0))
            minimum_d2[chosen] = 0.0
        indices = eligible[np.asarray(chosen, dtype=np.int64)]
        return SelectionBatch(
            ids=ids_arr[indices].tolist(), indices=indices, scores=score_arr[indices],
            metadata={
                "selector": "badge_kmeans_plusplus", "seed": self.seed,
                "include_bias": self.data.include_bias,
                "selection_d2": step_d2,
                "score_semantics": "squared_pseudo_label_gradient_norm",
                "distance_semantics": "squared_euclidean_gradient_distance",
            },
        )


__all__ = ["BadgeInput", "BadgeGradientNormScorer", "BadgeSelector"]
