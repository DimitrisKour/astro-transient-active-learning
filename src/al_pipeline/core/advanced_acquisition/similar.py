"""SIMILAR-style facility-location acquisition in hypothesized-gradient space.

Kothawade et al., NeurIPS 2021, SIMILAR. Three explicit instantiations:
  similar_fl:         facility-location coverage f(A) of current U
  similar_flcg:       facility-location conditional gain f(A|L)
  similar_flqmi_rare: FL2MI variant I_FL2(A;Q) = sum_q max_a s_qa + sum_a max_q s_aq

Uses nonnegative cosine similarities of factored final-linear-softmax
hypothesized (argmax-pseudolabel) gradients. This is a specified bounded-kernel
implementation of SIMILAR objectives, not every objective in the original paper.
No candidate ground truth, validation labels or test labels are inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from al_pipeline.core.types import SelectionBatch

POLICIES = ("similar_fl", "similar_flcg", "similar_flqmi_rare")


def _factors(probabilities: np.ndarray, embeddings: np.ndarray):
    p = np.asarray(probabilities, dtype=np.float32)
    h = np.asarray(embeddings, dtype=np.float32)
    if p.ndim != 2 or h.ndim != 2 or len(p) != len(h) or len(h) == 0:
        raise ValueError("Expect aligned nonempty 2-D probabilities and classifier-input features")
    if p.shape[1] < 2 or h.shape[1] < 1 or not np.isfinite(p).all() or not np.isfinite(h).all():
        raise ValueError("Nonfinite or invalid SIMILAR model outputs")
    if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("SIMILAR requires normalized probabilities")
    r = p.copy()
    r[np.arange(len(p)), np.argmax(p, axis=1)] -= 1
    h = np.c_[h, np.ones(len(h), dtype=np.float32)]
    # Normalize each factor: their tensor product is a normalized gradient.
    # A fully confident prediction can have zero hypothetical gradient.
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-10)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-10)
    return r, h


def _kernel(a, b):
    r1, h1 = a
    r2, h2 = b
    # float32 limits 4253^2 dense kernel to ~72 MB. Nonnegative similarities
    # guarantee monotone facility-location marginal gains for lazy greedy.
    sim = (r1 @ r2.T) * (h1 @ h2.T)
    np.clip(sim, 0., 1., out=sim)
    return sim


@dataclass
class SimilarInput:
    candidate_ids: tuple[str, ...]
    candidate_probabilities: np.ndarray
    candidate_embeddings: np.ndarray
    reference_ids: tuple[str, ...] = ()
    reference_probabilities: np.ndarray | None = None
    reference_embeddings: np.ndarray | None = None
    reference_labels: tuple[str, ...] = ()
    rare_classes: tuple[str, ...] = ()
    policy: str = "similar_fl"

    def __post_init__(self):
        self.candidate_ids = tuple(self.candidate_ids)
        self.reference_ids = tuple(self.reference_ids)
        self.reference_labels = tuple(self.reference_labels)
        self.rare_classes = tuple(self.rare_classes)
        if self.policy not in POLICIES:
            raise ValueError(f"Unknown SIMILAR variant: {self.policy}")
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Candidate IDs must be unique and nonempty")
        if len(self.candidate_ids) != len(self.candidate_probabilities):
            raise ValueError("Candidate IDs and prediction rows mismatch")
        if self.policy == 'similar_fl':
            if self.reference_ids or self.rare_classes:
                raise ValueError("Unconditional FL must not receive reference data")
        else:
            if not self.reference_ids or len(set(self.reference_ids)) != len(self.reference_ids):
                raise ValueError("Conditioned SIMILAR needs unique labeled reference IDs")
            if set(self.reference_ids) & set(self.candidate_ids):
                raise ValueError("Labeled references must not intersect unlabeled candidates")
            if self.reference_probabilities is None or self.reference_embeddings is None:
                raise ValueError("Missing labeled reference model outputs")
            if len(self.reference_ids) != len(self.reference_probabilities):
                raise ValueError("Labeled reference row count mismatch")
            if self.policy == "similar_flqmi_rare":
                if len(self.reference_labels) != len(self.reference_ids) or not self.rare_classes:
                    raise ValueError("Rare query requires confirmed reference labels and rare target classes")
                if not any(label in self.rare_classes for label in self.reference_labels):
                    raise ValueError("Rare query set is empty")

    def build(self):
        u = _factors(self.candidate_probabilities, self.candidate_embeddings)
        if self.policy == 'similar_fl':
            return _kernel(u, u), None
        l = _factors(self.reference_probabilities, self.reference_embeddings)
        if self.policy == 'similar_flcg':
            return _kernel(u, u), _kernel(u, l).max(axis=1)
        indices = np.array([i for i, label in enumerate(self.reference_labels)
                            if label in self.rare_classes], dtype=np.int64)
        # Query-by-candidate similarities. FL2MI needs only Q-by-U kernel.
        # FL2MI = sum_{q in Q} max_{a in A} s(q,a)
        #       + sum_{a in A} max_{q in Q} s(a,q).
        return _kernel((l[0][indices], l[1][indices]), u), None


class SimilarSelector:
    """Exact greedy by lazy marginal recalculation, deterministic ties by row ID.

    The heap records an upper bound on every unselected marginal. Diminishing
    returns of nonnegative FL / FLCG / FLQMI allows lazy updates. This returns
    the same greedy sequence as full per-step marginal recomputation (ties
    within float32 kernel arithmetic), not an approximation in the optimizer.
    """
    def __init__(self, data: SimilarInput):
        self.data = data
        self._matrix = None
        self._base = None
        self._coverage = None
        self._relevance = None
        self._initial = None

    def _ensure(self):
        if self._initial is not None:
            return
        matrix, base = self.data.build()
        self._matrix = matrix
        self._base = base
        self._coverage = np.zeros(matrix.shape[0], dtype=np.float32)
        if self.data.policy == 'similar_flqmi_rare':
            self._relevance = matrix.max(axis=0)
            self._initial = matrix.sum(axis=0, dtype=np.float64) + self._relevance
        else:
            reference = base if base is not None else self._coverage
            self._initial = np.maximum(matrix - reference[:, None], 0).sum(axis=0, dtype=np.float64)

    def initial_gains(self):
        self._ensure()
        return self._initial.copy()

    def _gain(self, idx):
        if self.data.policy == 'similar_flqmi_rare':
            return float(np.maximum(self._matrix[:, idx]-self._coverage, 0).sum(dtype=np.float64)
                         + self._relevance[idx])
        effective = self._coverage
        if self._base is not None:
            effective = np.maximum(effective, self._base)
        return float(np.maximum(self._matrix[:, idx]-effective, 0).sum(dtype=np.float64))

    def select(self, *, ids, scores, k, pool):
        self._ensure()
        ids = tuple(ids)
        scores = np.asarray(scores)
        if ids != self.data.candidate_ids or len(scores) != len(ids):
            raise ValueError("SIMILAR score row alignment mismatch")
        if not np.allclose(scores, self._initial, atol=1e-4, rtol=1e-5):
            raise ValueError("SIMILAR must select from own initial-gain scores")
        if not 0 <= k <= len(ids):
            raise ValueError("Invalid acquisition budget")
        eligible = set(pool.unlabeled_ids)
        if k > sum(x in eligible for x in ids):
            raise ValueError("Acquisition budget exceeds eligible candidates")
        # All heap scores are valid upper bounds on future gains.
        heap = [(-float(self._initial[i]), i) for i, name in enumerate(ids) if name in eligible]
        heapq.heapify(heap)
        used = set()
        chosen = []
        gains = []
        lazy_recomputations = 0
        for _ in range(k):
            while True:
                negative_bound, idx = heapq.heappop(heap)
                if idx in used:
                    continue
                actual = self._gain(idx)
                lazy_recomputations += 1
                # Another candidate wins a tie only when its index is smaller.
                next_bound = (-heap[0][0] if heap else -np.inf)
                next_idx = heap[0][1] if heap else len(ids)
                if actual > next_bound or (actual == next_bound and idx <= next_idx):
                    break
                heapq.heappush(heap, (-actual, idx))
            used.add(idx)
            chosen.append(idx)
            gains.append(actual)
            np.maximum(self._coverage, self._matrix[:, idx], out=self._coverage)
        chosen_arr = np.asarray(chosen, dtype=np.int64)
        return SelectionBatch(
            ids=[ids[i] for i in chosen], indices=chosen_arr,
            scores=np.asarray(gains, dtype=np.float64),
            metadata={
                'selector': 'similar_facility_location_lazy_greedy',
                'policy': self.data.policy,
                'kernel': 'nonnegative_cosine_factored_last_layer_pseudolabel_gradients',
                'reference_count': len(self.data.reference_ids),
                'rare_classes': list(self.data.rare_classes),
                'query_size': int(self._matrix.shape[0]) if self.data.policy == 'similar_flqmi_rare' else 0,
                'lazy_recomputations': lazy_recomputations,
            },
        )


class SimilarScorer:
    def score(self, data: SimilarInput) -> np.ndarray:
        return SimilarSelector(data).initial_gains()
