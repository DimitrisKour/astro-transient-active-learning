"""Greedy k-center core-set acquisition conditioned on existing labeled points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from al_pipeline.core.advanced_acquisition._common import (
    eligible_candidates,
    feature_matrix,
    squared_distances,
    validate_scores,
    validated_ids,
)
from al_pipeline.core.pool import Pool
from al_pipeline.core.scoring.base import Scorer
from al_pipeline.core.types import SelectionBatch


@dataclass(frozen=True, slots=True)
class CoreSetInput:
    """Feature vectors aligned to candidates, plus already-labeled anchors.

    ``labeled_embeddings`` must come from the labeled set under the SAME
    feature extractor/normalization as candidate embeddings. Candidate labels
    are not used. Caller handles object-level aggregation where necessary.
    """

    candidate_embeddings: np.ndarray
    labeled_embeddings: np.ndarray
    candidate_ids: tuple[str, ...]
    labeled_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        candidates = feature_matrix(self.candidate_embeddings, name="candidate embeddings")
        anchors = feature_matrix(self.labeled_embeddings, name="labeled embeddings")
        if candidates.shape[1] != anchors.shape[1]:
            raise ValueError("Candidate and labeled embeddings must share feature dimension")
        object.__setattr__(self, "candidate_ids", validated_ids(
            self.candidate_ids, expected_rows=len(candidates), name="core-set candidate IDs"
        ))
        object.__setattr__(self, "labeled_ids", validated_ids(
            self.labeled_ids, expected_rows=len(anchors), name="core-set labeled IDs"
        ))
        object.__setattr__(self, "candidate_embeddings", candidates)
        object.__setattr__(self, "labeled_embeddings", anchors)

    def initial_distances(self) -> np.ndarray:
        """Squared distance from each candidate to its nearest labeled anchor."""
        distances = np.full(self.candidate_embeddings.shape[0], np.inf)
        for anchor in self.labeled_embeddings:
            distances = np.minimum(distances, squared_distances(self.candidate_embeddings, anchor))
        return distances


class CoreSetScorer(Scorer[CoreSetInput]):
    """Initial squared distance to the nearest labeled-set representation."""

    def score(self, data: CoreSetInput) -> np.ndarray:
        return data.initial_distances()


class CoreSetSelector:
    """Deterministic greedy k-center acquisition with labeled-set anchors.

    Select the candidate maximally distant from its nearest labeled or newly
    selected point, then update all remaining nearest-center distances.
    The input scores MUST be the initial scores from CoreSetScorer; the
    returned scores are the dynamic distances AT THE TIME OF selection.
    This implementation minimizes the maximum Euclidean covering radius
    approximately via farthest-first traversal, not an uncertainty ranking.
    """

    def __init__(self, data: CoreSetInput) -> None:
        if not isinstance(data, CoreSetInput):
            raise TypeError("data must be CoreSetInput")
        self.data = data

    def select(self, *, ids, scores, k: int, pool: Pool) -> SelectionBatch:
        initial = self.data.initial_distances()
        ids_arr, scores_arr, eligible = eligible_candidates(
            ids, scores, k=k, pool=pool, expected_rows=len(initial)
        )
        if tuple(ids_arr) != self.data.candidate_ids:
            raise ValueError("core-set candidate IDs must match the representation row order")
        validate_scores(scores_arr, initial, method="core-set")
        positions: list[int] = []
        step_distances: list[float] = []
        min_dist = initial[eligible].copy()
        feats = self.data.candidate_embeddings[eligible]
        for _ in range(k):
            row = int(np.argmax(min_dist))  # tie: original candidate position
            positions.append(row)
            step_distances.append(float(min_dist[row]))
            min_dist = np.minimum(min_dist, squared_distances(feats, feats[row]))
            min_dist[positions] = -np.inf
        indices = eligible[np.asarray(positions, dtype=np.int64)]
        return SelectionBatch(
            ids=ids_arr[indices].tolist(), indices=indices,
            scores=np.asarray(step_distances, dtype=float),
            metadata={
                "selector": "coreset_kcenter_greedy",
                "distance": "squared_euclidean",
                "labeled_anchor_count": len(self.data.labeled_embeddings),
                "greedy_squared_covering_distances": step_distances,
                "score_semantics": "distance_to_nearest_labeled_or_selected_center_at_selection",
            },
        )


__all__ = ["CoreSetInput", "CoreSetScorer", "CoreSetSelector"]
