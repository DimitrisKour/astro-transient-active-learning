"""Policy behavior, edge cases, and domain-independent loop integration."""

import numpy as np
import pytest

from al_pipeline.core.batch_selection import DiversitySelector, ExplorationSelector
from al_pipeline.core.experiments.batch_policy import BatchAcquisitionConfig
from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.pool import Pool
from al_pipeline.core.scoring.class_balanced import ClassBalancedEntropyScorer, ClassBalancedInput
from al_pipeline.core.scoring.entropy import EntropyScorer
from al_pipeline.core.types import ClassificationPredictions


def predictions(values):
    return ClassificationPredictions(np.asarray(values), "probabilities")


def test_soft_bonus_uses_non_winning_class_probability():
    p = predictions([[0.51, 0.49], [0.9, 0.1]])
    data = ClassBalancedInput(p, np.array([100, 1]))
    scorer = ClassBalancedEntropyScorer()
    entropy = EntropyScorer().score(p)
    bonus = scorer.score(data) / entropy
    assert bonus[0] > bonus[1] > 1
    assert np.all(bonus <= 3)
    assert np.all(np.argmax(p.values, axis=1) == 0)


def test_balanced_counts_and_disabled_bonus_reduce_to_entropy():
    p = predictions([[0.5, 0.5], [0.8, 0.2], [1.0, 0.0]])
    expected = EntropyScorer().score(p)
    for counts, scorer in [
        ([10, 10], ClassBalancedEntropyScorer()),
        ([100, 0], ClassBalancedEntropyScorer(strength=0)),
        ([100, 0], ClassBalancedEntropyScorer(cap=1)),
    ]:
        np.testing.assert_allclose(scorer.score(ClassBalancedInput(p, counts)), expected)


@pytest.mark.parametrize("counts", [[1], [-1, 2], [np.nan, 1], [1.5, 1]])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):
        ClassBalancedInput(predictions([[0.5, 0.5]]), counts)


def test_diversity_avoids_duplicate_features_and_retains_original_indices():
    ids = ["ineligible", "a", "duplicate", "different", "outside"]
    pool = Pool.from_ids(ids[1:])
    features = np.array([[1, 0], [1, 0], [2, 0], [-1, 0], [0, 1]])
    selector = DiversitySelector(features, shortlist_multiplier=1.5)
    result = selector.select(ids=ids, scores=np.array([99, 5, 4, 3, 2]), k=2, pool=pool)
    assert result.ids == ["a", "different"]
    assert result.indices.tolist() == [1, 3]
    assert result.scores.tolist() == [5, 3]
    assert result.metadata["shortlist_size"] == 3
    assert result.metadata["greedy_min_squared_distances"] == [None, 4.0]


@pytest.mark.parametrize("k", [0, 1, 4])
def test_diversity_zero_identical_and_full_budget(k):
    ids = ["a", "b", "c", "d"]
    selector = DiversitySelector(np.zeros((4, 2)))
    result = selector.select(ids=ids, scores=np.ones(4), k=k, pool=Pool.from_ids(ids))
    assert result.ids == ids[:k]
    assert len(result.ids) == len(set(result.ids))


def test_diversity_rejects_misaligned_features():
    with pytest.raises(ValueError, match="align"):
        DiversitySelector(np.ones((1, 2))).select(
            ids=["a", "b"], scores=np.ones(2), k=1, pool=Pool.from_ids(["a", "b"])
        )


def test_exploration_can_reach_low_scoring_rows_and_is_reproducible():
    ids = [str(i) for i in range(100)]
    results = [
        ExplorationSelector(exploration_fraction=0.5, rng=np.random.default_rng(42)).select(
            ids=ids, scores=np.arange(100.0, 0, -1), k=10, pool=Pool.from_ids(ids)
        )
        for _ in range(2)
    ]
    assert results[0].ids == results[1].ids
    assert len(set(results[0].ids)) == 10
    assert len(results[0].metadata["exploration_ids"]) == 5
    assert any(int(x) >= 10 for x in results[0].metadata["exploration_ids"])


def test_core_loop_accepts_both_selectors_without_astronomy():
    ids = ["a", "b", "c"]
    p = predictions([[0.5, 0.5], [0.6, 0.4], [0.9, 0.1]])
    for selector in [
        DiversitySelector(np.eye(3)),
        ExplorationSelector(rng=np.random.default_rng(1)),
    ]:
        pool = Pool.from_ids(ids)
        loop = ActiveLearningLoop(scorer=EntropyScorer(), selector=selector, pool=pool)
        selected = loop.acquire(ids=ids, acquisition_data=p, k=2)
        assert len(selected) == 2
        assert set(selected.ids) == pool.pending_ids


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shortlist_multiplier": 0.5},
        {"balance_alpha": 0},
        {"balance_cap": 0.5},
        {"exploration_fraction": 1.1},
        {"balance_strength": np.inf},
    ],
)
def test_invalid_policy_parameters(kwargs):
    with pytest.raises(ValueError):
        BatchAcquisitionConfig(**kwargs)
