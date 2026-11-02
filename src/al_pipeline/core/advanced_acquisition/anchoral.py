"""AnchorAL-style class-anchored subpool selection (Lesci & Vlachos, NAACL 2024).

Uses confirmed-label anchors only, cosine-nearest-neighbor retrieval and entropy
selection on the retrieved subpool. No hidden candidate labels or test labels.
This is an acquisition-quality adaptation: existing AstroMiNN inference still
scores all U_t before the subpool is filtered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from al_pipeline.core.pool import Pool
from al_pipeline.core.types import SelectionBatch


def _unit(values: np.ndarray, *, name: str) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 1 or not np.isfinite(data).all():
        raise ValueError(f"{name} must be a finite 2-dimensional feature matrix")
    if (np.linalg.norm(data, axis=1) < 1e-12).any():
        raise ValueError(f"{name} includes a zero-length representation")
    return data / np.linalg.norm(data, axis=1, keepdims=True)


@dataclass(frozen=True)
class AnchorALInput:
    candidate_ids: tuple[str, ...]
    candidate_embeddings: np.ndarray
    labeled_ids: tuple[str, ...]
    labeled_embeddings: np.ndarray
    labeled_labels: tuple[str, ...]
    class_names: tuple[str, ...]
    seed: int
    cycle: int
    anchors_per_class: int = 5
    neighbors_per_anchor: int = 200
    max_subpool: int = 1419

    def __post_init__(self) -> None:
        u = _unit(self.candidate_embeddings, name="candidate_embeddings")
        l = _unit(self.labeled_embeddings, name="labeled_embeddings")
        if u.shape[1] != l.shape[1]:
            raise ValueError("Candidate and anchor embedding dimensions differ")
        if (len(self.candidate_ids) != len(u) or len(self.labeled_ids) != len(l)
                or len(self.labeled_labels) != len(l)):
            raise ValueError("Object IDs/labels and embedding rows do not align")
        if (len(set(self.candidate_ids)) != len(self.candidate_ids)
                or len(set(self.labeled_ids)) != len(self.labeled_ids)
                or set(self.candidate_ids) & set(self.labeled_ids)):
            raise ValueError("Candidate and labeled sets must be disjoint and unique")
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Expected unique ordered class names")
        if set(self.labeled_labels) - set(self.class_names):
            raise ValueError("Unknown confirmed training labels")
        if self.seed < 0 or self.cycle < 1:
            raise ValueError("Invalid seed/cycle")
        if min(self.anchors_per_class, self.neighbors_per_anchor, self.max_subpool) < 1:
            raise ValueError("AnchorAL parameters must be positive")
        object.__setattr__(self, "candidate_embeddings", u)
        object.__setattr__(self, "labeled_embeddings", l)


def _sample_anchors(data: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic seeded cosine k-means++-style anchor initialization."""
    count = min(count, len(data))
    first = int(rng.integers(len(data)))
    selected = [first]
    closest = 1.0 - (data @ data[first])
    for _ in range(1, count):
        available = np.ones(len(data), dtype=bool)
        available[selected] = False
        weights = np.where(available, np.maximum(closest, 0.0) ** 2, 0.0)
        if weights.sum() <= 1e-15:
            nxt = int(np.flatnonzero(available)[0])
        else:
            nxt = int(rng.choice(len(data), p=weights / weights.sum()))
        selected.append(nxt)
        closest = np.minimum(closest, 1.0 - data @ data[nxt])
    return np.asarray(selected, dtype=np.int64)


class AnchorALSelector:
    """Select high-entropy examples within a class-anchored fixed-size subpool."""

    def __init__(self, data: AnchorALInput) -> None:
        self.data = data

    def select(self, *, ids, scores, k: int, pool: Pool) -> SelectionBatch:
        d = self.data
        ordered_ids = tuple(str(x) for x in ids)
        s = np.asarray(scores, dtype=np.float64)
        if ordered_ids != d.candidate_ids:
            raise ValueError("Candidate IDs must match inference/embedding order")
        if s.shape != (len(ordered_ids),) or not np.isfinite(s).all():
            raise ValueError("Entropy scores must be finite and object-aligned")
        if not 0 <= k <= len(ordered_ids):
            raise ValueError("Invalid acquisition budget")
        eligible = np.asarray([i for i, oid in enumerate(ordered_ids)
                               if oid in pool.unlabeled_ids], dtype=np.int64)
        if len(eligible) < k:
            raise ValueError("Not enough eligible candidates")
        if not k:
            return SelectionBatch(ids=[], indices=np.empty(0, dtype=np.int64),
                                  scores=np.empty(0), metadata={"selector": "anchoral_entropy"})

        chosen_anchors: list[int] = []
        by_class: dict[str, list[str]] = {}
        classes = np.asarray(d.labeled_labels, dtype=object)
        for class_idx, class_name in enumerate(d.class_names):
            rows = np.flatnonzero(classes == class_name)
            # Sorting by object ID keeps anchor sampling independent of manifest row order.
            rows = rows[np.argsort(np.asarray(d.labeled_ids, dtype=str)[rows], kind="stable")]
            if not len(rows):
                by_class[class_name] = []
                continue
            rng = np.random.default_rng(np.random.SeedSequence(
                [d.seed, d.cycle, class_idx, 1818]))
            local = _sample_anchors(d.labeled_embeddings[rows],
                                    d.anchors_per_class, rng)
            selected = rows[local].tolist()
            chosen_anchors.extend(selected)
            by_class[class_name] = [d.labeled_ids[i] for i in selected]
        if not chosen_anchors:
            raise ValueError("No confirmed labeled anchors available")

        # Exact cosine index for our small historical U_t, replacing ANN/HNSW.
        # Only returned kNN memberships enter the mean-similarity ranking.
        similarities = d.candidate_embeddings[eligible] @ d.labeled_embeddings[chosen_anchors].T
        nearest = min(len(eligible), d.neighbors_per_anchor)
        subset = np.empty(0, dtype=np.int64)
        mean_sim = np.full(len(eligible), -np.inf, dtype=np.float64)
        while True:
            # Sort ensures deterministic ties. For each anchor, retrieve kNN.
            order = np.argsort(-similarities, axis=0, kind="stable")[:nearest]
            counts = np.zeros(len(eligible), dtype=np.int32)
            total = np.zeros(len(eligible), dtype=np.float64)
            for j in range(similarities.shape[1]):
                rows = order[:, j]
                counts[rows] += 1
                total[rows] += similarities[rows, j]
            subset = np.flatnonzero(counts)
            mean_sim[subset] = total[subset] / counts[subset]
            if len(subset) >= k or nearest == len(eligible):
                break
            nearest = min(len(eligible), nearest * 2)
        cap = min(max(k, d.max_subpool), len(subset))
        # Rank candidate IDs by average similarity among retrieved anchor neighbors.
        ranked = np.lexsort((subset, -mean_sim[subset]))[:cap]
        subrows = subset[ranked]
        # Entropy inside class-aware retrieved subpool; ties use original pool order.
        scored = np.lexsort((eligible[subrows], -s[eligible[subrows]]))[:k]
        indices = eligible[subrows[scored]]
        return SelectionBatch(
            ids=[ordered_ids[int(i)] for i in indices],
            indices=indices,
            scores=s[indices],
            metadata={
                "selector": "anchoral_entropy",
                "paper": "Lesci and Vlachos (NAACL 2024), adapted to AstroMiNN",
                "anchor_selection": "seeded_classwise_cosine_kmeans++",
                "similarity": "cosine",
                "candidate_subpool_ranking": "mean_similarity_over_retrieved_neighbors",
                "scoring_within_subpool": "entropy",
                "anchors_per_class": d.anchors_per_class,
                "neighbors_per_anchor_requested": d.neighbors_per_anchor,
                "neighbors_per_anchor_actual": nearest,
                "max_subpool": d.max_subpool,
                "subpool_size": len(subrows),
                "anchor_ids_by_class": by_class,
                "labeled_counts_by_class": {
                    name: int(np.count_nonzero(classes == name))
                    for name in d.class_names
                },
                "no_candidate_labels_used": True,
                "inference_full_pool": True,
            },
        )


__all__ = ["AnchorALInput", "AnchorALSelector"]
