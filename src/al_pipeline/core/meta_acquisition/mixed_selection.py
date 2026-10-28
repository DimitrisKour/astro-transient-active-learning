"""Sequential, cross-policy batch acquisition for the Task13 TAILOR pilot.

All acquisition arms share one *unlabeled* candidate pool and a single set of
already selected positions. No candidate ground-truth labels are accepted.

For a batch containing only one arm, the implementation reproduces that arm's
existing standalone selector, including its seeded randomness and tie rules.
With multiple arms, diversity and Fisher information account for selections
already made by other arms; this is a principled interleaving rather than a
merge of separately precomputed batch rankings.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Sequence

import numpy as np

from al_pipeline.core.advanced_acquisition.fisher_batch import FisherBatchInput
from al_pipeline.core.batch_selection import DiversitySelector
from al_pipeline.core.experiments.batch_policy import BatchAcquisitionConfig
from al_pipeline.core.scoring.class_balanced import ClassBalancedInput
from al_pipeline.core.scoring.entropy import EntropyScorer
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.types import ClassificationPredictions, SelectionBatch


SUPPORTED_ARMS = frozenset(
    {"entropy_diversity", "entropy_balanced_explore", "fisher_batch"}
)


def _unit_features(embeddings: np.ndarray) -> np.ndarray:
    """Match DiversitySelector's safe per-object scaling and L2 norm."""
    feats = np.asarray(embeddings, dtype=np.float64)
    scale = np.max(np.abs(feats), axis=1, keepdims=True)
    feats = feats / np.where(scale > 0, scale, 1.0)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.where(norms > 0, norms, 1.0)


def select_mixed_batch(
    *,
    object_ids: Sequence[str],
    predictions: ClassificationPredictions,
    embeddings: np.ndarray,
    confirmed_counts: Sequence[int],
    labeled_probabilities: np.ndarray,
    labeled_embeddings: np.ndarray,
    arm_assignments: Sequence[str],
    exploration_rng: np.random.Generator,
    batch_policy: BatchAcquisitionConfig | None = None,
) -> SelectionBatch:
    """Execute an externally chosen arm sequence without labels or retraining.

    Each arm's total number of positions is known before selection (as in
    TAILOR Algorithm 2). Hence the original entropy-diversity shortlist and
    the balanced-exploration quota can be determined before interleaving.

    Fisher A-optimal information is updated after *every* selected object,
    including ones obtained using another arm, because they all enter L_t.
    """
    ids = tuple(object_ids)
    arms = tuple(arm_assignments)
    n = len(ids)
    if n == 0 or len(set(ids)) != n or any(not isinstance(x, str) for x in ids):
        raise ValueError("Candidate IDs must be unique, nonempty strings")
    if not arms or len(arms) > n or set(arms) - SUPPORTED_ARMS:
        raise ValueError("Invalid mixed-policy arm assignments or acquisition budget")
    if not isinstance(predictions, ClassificationPredictions):
        raise TypeError("predictions must be ClassificationPredictions")
    probs = np.asarray(predictions.as_probabilities(), dtype=np.float64)
    feats = np.asarray(embeddings, dtype=np.float64)
    if probs.shape[0] != n or feats.ndim != 2 or feats.shape[0] != n:
        raise ValueError("Candidate predictions, embeddings, and IDs do not align")
    if feats.shape[1] == 0 or not np.isfinite(feats).all():
        raise ValueError("Candidate embeddings must be finite and nonempty")
    policy = batch_policy or BatchAcquisitionConfig()
    counts = np.asarray(confirmed_counts)
    if counts.shape != (probs.shape[1],):
        raise ValueError("Confirmed counts must match the classifier class order")
    from al_pipeline.core.scoring.class_balanced import ClassBalancedEntropyScorer

    entropy = EntropyScorer().score(predictions)
    balanced = policy.make_scorer().score(
        ClassBalancedInput(predictions=predictions, labeled_counts=counts)
    )
    if not np.all(np.isfinite(entropy)) or not np.all(np.isfinite(balanced)):
        raise ValueError("Acquisition scores contain non-finite values")
    all_indices = np.arange(n, dtype=np.int64)
    entropy_order = TopKSelector._top_k_indices(
        eligible_indices=all_indices, scores=entropy, k=n
    )
    balance_order = TopKSelector._top_k_indices(
        eligible_indices=all_indices, scores=balanced, k=n
    )
    allocations = Counter(arms)
    used = np.zeros(n, dtype=bool)
    chosen: list[int] = []
    chosen_scores: list[float] = []
    chosen_arm: list[str] = []
    normalized = _unit_features(feats)

    # Preserve the full-batch entropy-diversity shortlist when only this
    # acquisition method is selected. Other arms may deplete that shortlist;
    # there is an explicitly recorded fallback to the score-ordered pool.
    k_div = allocations["entropy_diversity"]
    shortlist_size = min(n, math.ceil(policy.shortlist_multiplier * k_div))
    shortlist = entropy_order[:shortlist_size]
    div_min_d2 = np.full(shortlist_size, np.inf)
    div_fallbacks = 0

    # The original ExplorationSelector samples exploration IDs before
    # choosing targeted points, then emits targeted points first. Reproduce
    # that behavior for the standalone-arm special case.
    k_bal = allocations["entropy_balanced_explore"]
    n_explore = min(k_bal, math.floor(policy.exploration_fraction * k_bal + 0.5))
    n_target = k_bal - n_explore
    reserved = (
        np.asarray(exploration_rng.choice(all_indices, size=n_explore, replace=False), dtype=np.int64)
        if n_explore
        else np.empty(0, dtype=np.int64)
    )
    reserved_set = set(int(x) for x in reserved)
    balance_pulls = 0
    exploration_fallbacks = 0

    fisher_engine = None
    inverse = None
    if allocations["fisher_batch"]:
        fisher_input = FisherBatchInput(
            candidate_ids=ids,
            probabilities=probs,
            embeddings=feats,
            labeled_probabilities=np.asarray(labeled_probabilities),
            labeled_embeddings=np.asarray(labeled_embeddings),
            rank=32,
            ridge=1.0,
            projection_seed=271828,
        )
        fisher_engine = fisher_input.engine
        inverse = fisher_engine.inverse.copy()

    for arm in arms:
        if arm == "entropy_diversity":
            eligible = np.asarray([i for i, candidate in enumerate(shortlist) if not used[candidate]])
            if not len(eligible):
                div_fallbacks += 1
                row = next(int(candidate) for candidate in entropy_order if not used[candidate])
                score = float(entropy[row])
            elif not chosen:
                row = int(shortlist[eligible[0]])
                score = float(entropy[row])
            else:
                shortlist_pos = int(eligible[int(np.argmax(div_min_d2[eligible]))])
                row = int(shortlist[shortlist_pos])
                score = float(div_min_d2[shortlist_pos])
        elif arm == "entropy_balanced_explore":
            if balance_pulls < n_target:
                row = next(
                    (int(i) for i in balance_order if not used[i] and i not in reserved_set),
                    None,
                )
                if row is None:
                    row = next(int(i) for i in balance_order if not used[i])
                score = float(balanced[row])
            else:
                reserved_available = [int(i) for i in reserved if not used[i]]
                if reserved_available:
                    row = reserved_available.pop(0)
                else:
                    exploration_fallbacks += 1
                    row = int(exploration_rng.choice(np.flatnonzero(~used)))
                score = float(balanced[row])
            balance_pulls += 1
        else:
            assert fisher_engine is not None and inverse is not None
            gains = (
                fisher_engine.initial_gains()
                if not chosen
                else fisher_engine._gain(inverse)
            )
            gains[used] = -np.inf
            row = int(np.argmax(gains))
            score = float(gains[row])

        if used[row]:
            raise RuntimeError("Mixed acquisition attempted a duplicate selection")
        used[row] = True
        chosen.append(row)
        chosen_scores.append(score)
        chosen_arm.append(arm)

        # Each new object can improve within-batch geometric diversity,
        # regardless of which algorithm selected it.
        if shortlist_size:
            delta = normalized[shortlist] - normalized[row]
            d2 = np.einsum("ij,ij->i", delta, delta)
            div_min_d2 = np.minimum(div_min_d2, d2)
            div_min_d2[used[shortlist]] = -np.inf

        if fisher_engine is not None:
            assert inverse is not None
            z = fisher_engine.z[row]
            mid = np.eye(z.shape[0]) + z @ inverse @ z.T
            inverse -= inverse @ z.T @ np.linalg.solve(mid, z @ inverse)
            inverse = (inverse + inverse.T) * 0.5

    selected_indices = np.asarray(chosen, dtype=np.int64)
    return SelectionBatch(
        ids=[ids[row] for row in chosen],
        indices=selected_indices,
        scores=np.asarray(chosen_scores, dtype=np.float64),
        metadata={
            "selector": "tailor_sequential_policy_interleaving",
            "arm_assignments": chosen_arm,
            "arm_counts": dict(Counter(chosen_arm)),
            "subselector_score_semantics": {
                "entropy_diversity": "entropy_for_first_then_greedy_squared_distance",
                "entropy_balanced_explore": "class_balanced_entropy",
                "fisher_batch": "projected_Fisher_A_optimal_marginal_gain",
            },
            "diversity_shortlist_size": shortlist_size,
            "diversity_shortlist_fallbacks": div_fallbacks,
            "balanced_exploration_quota": n_explore,
            "balanced_exploration_fallbacks": exploration_fallbacks,
            "fisher_updates_include_all_selected_objects": True,
        },
    )
