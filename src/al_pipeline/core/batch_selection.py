"""Domain-independent diversity and exploration batch selectors."""

from __future__ import annotations

import math

import numpy as np

from al_pipeline.core.pool import Pool
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.types import SelectionBatch


def _eligible(ids, scores, k, pool):
    ids = np.asarray(ids, dtype=object)
    scores = np.asarray(scores, dtype=float)
    TopKSelector._validate_inputs(ids=ids, scores=scores, k=k)
    eligible = np.asarray(
        [i for i, object_id in enumerate(ids) if object_id in pool.unlabeled_ids],
        dtype=np.int64,
    )
    if k > len(eligible):
        raise ValueError("Not enough eligible objects for the requested budget.")
    order = TopKSelector._top_k_indices(eligible_indices=eligible, scores=scores, k=len(eligible))
    return ids, scores, order


class DiversitySelector:
    """Farthest-first batch within a high-score shortlist.

    Start with the highest-score row. Then maximize the minimum squared
    Euclidean distance to the selected rows, after L2 normalization.
    Score order breaks distance ties. Distances concern the current batch,
    not coverage of the entire labeled set. The returned scores retain the
    original uncertainty meaning; greedy distances are recorded separately.
    """

    def __init__(self, embeddings: np.ndarray, *, shortlist_multiplier=3.0):
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[1] < 1:
            raise ValueError("Embeddings must be a two-dimensional feature matrix.")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings must be finite.")
        if not np.isfinite(shortlist_multiplier) or shortlist_multiplier < 1:
            raise ValueError("shortlist_multiplier must be finite and >= 1.")
        self.embeddings = embeddings
        self.shortlist_multiplier = float(shortlist_multiplier)

    def select(self, *, ids, scores, k: int, pool: Pool) -> SelectionBatch:
        ids, scores, order = _eligible(ids, scores, k, pool)
        if len(self.embeddings) != len(ids):
            raise ValueError("Embeddings must align row-for-row with candidate IDs.")
        size = min(len(order), math.ceil(self.shortlist_multiplier * k))
        shortlist = order[:size]
        features = self.embeddings[shortlist]
        # Stable normalization also permits zero embeddings (treated as origin).
        scale = np.max(np.abs(features), axis=1, keepdims=True) if size else features[:, :1]
        features = features / np.where(scale > 0, scale, 1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.where(norms > 0, norms, 1)
        chosen = []
        distances = []
        minimum = np.full(size, np.inf)
        for step in range(k):
            row = 0 if step == 0 else int(np.argmax(minimum))
            chosen.append(row)
            distances.append(None if step == 0 else float(minimum[row]))
            delta = features - features[row]
            minimum = np.minimum(minimum, np.einsum("ij,ij->i", delta, delta))
            minimum[chosen] = -np.inf
        indices = shortlist[np.asarray(chosen, dtype=np.int64)]
        return SelectionBatch(
            ids=ids[indices].tolist(),
            indices=indices,
            scores=scores[indices],
            metadata={
                "selector": "shortlist_farthest_first",
                "shortlist_multiplier": self.shortlist_multiplier,
                "shortlist_size": size,
                "normalization": "l2_per_object",
                "greedy_min_squared_distances": distances,
            },
        )


class ExplorationSelector:
    """Uniform random exploration, then highest scores among remaining rows.

    floor(fraction*k + 0.5) exploration slots are drawn without replacement
    from all eligible candidates. The remaining slots are filled by score.
    A caller should construct a fresh cycle-seeded generator on each query.
    """

    def __init__(self, *, exploration_fraction=0.2, rng: np.random.Generator):
        if not np.isfinite(exploration_fraction) or not 0 <= exploration_fraction <= 1:
            raise ValueError("exploration_fraction must lie in [0, 1].")
        self.exploration_fraction = float(exploration_fraction)
        self.rng = rng

    def select(self, *, ids, scores, k: int, pool: Pool) -> SelectionBatch:
        ids, scores, order = _eligible(ids, scores, k, pool)
        n_explore = min(k, math.floor(self.exploration_fraction * k + 0.5))
        # Sample in original row order, independently of the ranking.
        exploration = self.rng.choice(np.sort(order), size=n_explore, replace=False)
        exploration_set = set(exploration.tolist())
        targeted = np.asarray(
            [i for i in order if i not in exploration_set][: k - n_explore], dtype=np.int64
        )
        indices = np.concatenate([targeted, exploration])
        return SelectionBatch(
            ids=ids[indices].tolist(),
            indices=indices,
            scores=scores[indices],
            metadata={
                "selector": "top_k_with_uniform_exploration",
                "exploration_fraction": self.exploration_fraction,
                "exploration_ids": ids[exploration].tolist(),
                "targeted_ids": ids[targeted].tolist(),
            },
        )


# TODO: Add provider costs/eligibility as explicit inputs when a real oracle
# supplies them. Keep survey-specific rules outside these selectors.

__all__ = ["DiversitySelector", "ExplorationSelector"]
