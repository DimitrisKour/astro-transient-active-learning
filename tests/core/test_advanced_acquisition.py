"""Package C: model-independent BADGE, core-set and ensemble disagreement tests."""

from __future__ import annotations

import numpy as np
import pytest

from al_pipeline.core.advanced_acquisition import (
    BadgeGradientNormScorer,
    BadgeInput,
    BadgeSelector,
    CoreSetInput,
    CoreSetScorer,
    CoreSetSelector,
    EnsembleDisagreementScorer,
)
from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.pool import Pool
from al_pipeline.core.prediction.ensemble import EnsembleMember, PredictionEnsemble
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.types import ClassificationPredictions



def make_badge(predictions, embeddings, *, candidate_ids=None, include_bias=True):
    if candidate_ids is None:
        candidate_ids = tuple(str(i) for i in range(len(embeddings)))
    return BadgeInput(predictions, embeddings, tuple(candidate_ids), include_bias=include_bias)


def make_core(candidate_embeddings, labeled_embeddings, *, candidate_ids=None, labeled_ids=None):
    if candidate_ids is None:
        candidate_ids = tuple(str(i) for i in range(len(candidate_embeddings)))
    if labeled_ids is None:
        labeled_ids = tuple(f"L{i}" for i in range(len(labeled_embeddings)))
    return CoreSetInput(candidate_embeddings, labeled_embeddings, tuple(candidate_ids), tuple(labeled_ids))


def p(values, *, output_type="probabilities"):
    return ClassificationPredictions(np.asarray(values, dtype=float), output_type)


def badge_input(ids=None):
    return make_badge(
        p([[0.5, 0.5], [0.9, 0.1], [0.5, 0.5], [0.7, 0.3]]),
        np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
        candidate_ids=ids,
    )


def badge_select(data, ids=None, *, k=2, seed=135, pool=None):
    ids = ids if ids is not None else [str(i) for i in range(len(data.embeddings))]
    pool = pool if pool is not None else Pool.from_ids(ids)
    return BadgeSelector(data, seed=seed).select(
        ids=ids, scores=BadgeGradientNormScorer().score(data), k=k, pool=pool,
    )


def core_input(ids=None):
    return make_core(
        candidate_embeddings=np.array([[0., 0.], [10., 0.], [0., 8.], [4., 0.]]),
        labeled_embeddings=np.array([[0., 0.]]),
        candidate_ids=ids,
    )


def core_select(data, ids=None, *, k=2, pool=None):
    ids = ids if ids is not None else [str(i) for i in range(len(data.candidate_embeddings))]
    return CoreSetSelector(data).select(
        ids=ids, scores=CoreSetScorer().score(data), k=k,
        pool=pool if pool is not None else Pool.from_ids(ids),
    )


def ensemble():
    return PredictionEnsemble([
        EnsembleMember(
            "a", ("a", "b", "c"), ("X", "Y"),
            p([[.99, .01], [.99, .01], [.5, .5]]),
        ),
        EnsembleMember(
            "b", ("c", "b", "a"), ("Y", "X"),
            p([[.5, .5], [.01, .99], [.99, .01]]),
        ),
    ])


def test_badge_dense_gradient_matches_analytic():
    data = make_badge(p([[.75, .25]]), [[2., -3.]])
    np.testing.assert_allclose(
        data.dense_gradient_embeddings(), [[-.5, .75, -.25, .5, -.75, .25]],
    )


def test_badge_no_bias_has_expected_width():
    data = make_badge(p([[.75, .25]]), [[2., -3.]], include_bias=False)
    np.testing.assert_allclose(data.dense_gradient_embeddings(), [[-.5, .75, .5, -.75]])


def test_badge_factors_match_dense_norms():
    data = badge_input()
    dense = data.dense_gradient_embeddings()
    np.testing.assert_allclose(BadgeGradientNormScorer().score(data), (dense**2).sum(axis=1))


def test_badge_zero_confidence_is_small_gradient():
    data = make_badge(p([[1., 0.], [.5, .5]]), [[1.], [1.]])
    values = BadgeGradientNormScorer().score(data)
    assert values[0] == 0 and values[1] > 0


def test_badge_initial_center_has_max_norm():
    data = badge_input()
    scores = BadgeGradientNormScorer().score(data)
    result = badge_select(data, k=1)
    assert result.indices[0] == np.argmax(scores)
    assert result.scores[0] == scores.max()


def test_badge_euclidean_factorization_matches_dense():
    data = badge_input()
    residuals, features = data.factors()
    gradients = data.dense_gradient_embeddings()
    i = 0
    implicit = (
        np.einsum("ij,ij->i", residuals, residuals) * np.einsum("ij,ij->i", features, features)
        + np.dot(residuals[i], residuals[i]) * np.dot(features[i], features[i])
        - 2 * (residuals @ residuals[i]) * (features @ features[i])
    )
    np.testing.assert_allclose(implicit, np.sum((gradients - gradients[i])**2, axis=1), atol=1e-12)


def test_badge_reproducible_seeded_selection():
    data = badge_input()
    assert badge_select(data, seed=246).indices.tolist() == badge_select(data, seed=246).indices.tolist()


def test_badge_batch_unique_and_full_budget():
    data = badge_input()
    result = badge_select(data, k=4)
    assert len(result.ids) == len(set(result.ids)) == 4
    assert sorted(result.indices.tolist()) == list(range(4))


def test_badge_handles_zero_gradients_without_nan():
    data = make_badge(p([[1., 0.]] * 4), np.zeros((4, 3)))
    result = badge_select(data, k=4)
    assert result.indices.tolist() == [0, 1, 2, 3]
    assert result.metadata["selection_d2"] == [0.0] * 4


def test_badge_handles_identical_nonzero_gradients():
    data = make_badge(p([[.5, .5]] * 3), np.ones((3, 2)))
    assert badge_select(data, k=3).indices.tolist() == [0, 1, 2]


def test_badge_does_not_select_labeled_or_pending():
    ids = ["a", "b", "c", "d"]
    data = badge_input(ids)
    pool = Pool.from_ids(ids, labeled_ids=["a"])
    pool.mark_pending(["c"])
    result = badge_select(data, ids, pool=pool, k=2)
    assert set(result.ids) == {"b", "d"}
    assert pool.pending_ids == {"c"}


def test_badge_zero_budget():
    result = badge_select(badge_input(), k=0)
    assert result.ids == [] and result.indices.size == 0


def test_badge_unavailable_budget_rejected():
    with pytest.raises(ValueError, match="eligible"):
        badge_select(badge_input(), k=5)


def test_badge_rejects_wrong_scorer_values():
    data = badge_input(["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="matching scorer"):
        BadgeSelector(data, seed=3).select(
            ids=["a", "b", "c", "d"], scores=np.ones(4), k=2,
            pool=Pool.from_ids(["a", "b", "c", "d"]),
        )


def test_badge_refuses_misaligned_candidate_ids():
    with pytest.raises(ValueError, match="align"):
        badge_select(badge_input(), ids=["a", "b"], k=1)


@pytest.mark.parametrize("bad", [
    np.array([[float("nan")]]), np.array([[float("inf")]]),
    np.array([[1 + 2j]]), np.ones((1, 1, 1)), np.zeros((1, 0)),
])
def test_badge_rejects_bad_embeddings(bad):
    with pytest.raises((ValueError, TypeError)):
        make_badge(p([[.5, .5]]), bad)


def test_badge_rejects_mismatched_embeddings_and_labels():
    with pytest.raises(ValueError, match="align"):
        make_badge(p([[.5, .5]]), np.ones((2, 1)))


def test_badge_rejects_invalid_seed():
    with pytest.raises(ValueError, match="seed"):
        BadgeSelector(badge_input(), seed=-1)


def test_badge_loop_integration_and_pool_commit():
    ids = ["a", "b", "c", "d"]
    data = badge_input(ids)
    pool = Pool.from_ids(ids)
    loop = ActiveLearningLoop(
        scorer=BadgeGradientNormScorer(), selector=BadgeSelector(data, seed=1), pool=pool,
    )
    result = loop.acquire(ids=ids, acquisition_data=data, k=2)
    assert len(result.ids) == 2
    assert set(result.ids) == pool.pending_ids


def test_core_set_initial_distances():
    np.testing.assert_allclose(CoreSetScorer().score(core_input()), [0., 100., 64., 16.])


def test_core_set_greedy_sequence_uses_labeled_anchors():
    result = core_select(core_input(), k=2)
    assert result.indices.tolist() == [1, 2]
    np.testing.assert_allclose(result.scores, [100., 64.])


def test_core_set_selection_scores_are_dynamic():
    data = make_core(np.array([[10., 0.], [11., 0.], [-9., 0.]]), np.array([[0., 0.]]))
    result = core_select(data, k=2)
    assert result.indices.tolist() == [1, 2]
    np.testing.assert_allclose(result.scores, [121., 81.])
    assert result.metadata["labeled_anchor_count"] == 1


def test_core_set_ties_use_input_order():
    data = make_core(np.array([[1., 0.], [-1., 0.], [0., 1.]]), [[0., 0.]])
    assert core_select(data, k=3).indices.tolist() == [0, 1, 2]


def test_core_set_zero_budget():
    result = core_select(core_input(), k=0)
    assert result.ids == [] and result.scores.size == 0


def test_core_set_excludes_pending_and_labeled():
    ids = ["a", "b", "c", "d"]
    pool = Pool.from_ids(ids, labeled_ids=["b"])
    pool.mark_pending(["c"])
    result = core_select(core_input(ids), ids, pool=pool, k=2)
    assert set(result.ids) == {"a", "d"}
    assert set(pool.pending_ids) == {"c"}


def test_core_set_unavailable_budget():
    with pytest.raises(ValueError, match="eligible"):
        core_select(core_input(), k=5)


def test_core_set_rejects_misaligned_scores():
    data = core_input(["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="matching scorer"):
        CoreSetSelector(data).select(
            ids=["a", "b", "c", "d"], scores=np.ones(4), k=1,
            pool=Pool.from_ids(["a", "b", "c", "d"]),
        )


def test_core_set_rejects_wrong_feature_width():
    with pytest.raises(ValueError, match="feature dimension"):
        make_core(np.ones((2, 3)), np.ones((1, 2)))


def test_core_set_rejects_missing_labeled_anchors():
    with pytest.raises(ValueError, match="shape"):
        make_core(np.ones((2, 3)), np.zeros((0, 3)))


def test_core_set_rejects_bad_features():
    with pytest.raises(ValueError, match="finite"):
        make_core(np.array([[np.inf, 1.]]), np.zeros((1, 2)))


def test_core_set_duplicate_candidates_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        core_select(core_input(), ids=["a", "a", "b", "c"], k=2)


def test_core_set_loop_integration():
    ids = ["a", "b", "c", "d"]
    data = core_input(ids)
    pool = Pool.from_ids(ids)
    loop = ActiveLearningLoop(scorer=CoreSetScorer(), selector=CoreSetSelector(data), pool=pool)
    selection = loop.acquire(ids=ids, acquisition_data=data, k=2)
    assert selection.ids == ["b", "c"]
    assert pool.pending_ids == {"b", "c"}


def test_ensemble_disagreement_agreement_vs_disagreement():
    e = ensemble()
    scores = EnsembleDisagreementScorer().score(e)
    assert scores.shape == (3,)
    assert scores[0] > scores[1] and scores[0] > scores[2]
    assert scores[1] == pytest.approx(0., abs=1e-10)
    assert scores[2] == pytest.approx(0., abs=1e-10)


def test_ensemble_scoring_preserves_package_a_values():
    e = ensemble()
    np.testing.assert_allclose(EnsembleDisagreementScorer().score(e), e.disagreement())


def test_ensemble_integration_with_top_k():
    e = ensemble()
    ids = list(e.object_ids)
    pool = Pool.from_ids(ids)
    loop = ActiveLearningLoop(
        scorer=EnsembleDisagreementScorer(), selector=TopKSelector(), pool=pool,
    )
    result = loop.acquire(ids=ids, acquisition_data=e, k=1)
    assert result.ids == ["a"]
    assert pool.pending_ids == {"a"}


def test_ensemble_type_guard():
    with pytest.raises(TypeError, match="PredictionEnsemble"):
        EnsembleDisagreementScorer().score(object())


def test_badge_rejects_reordered_ids_even_if_scores_match():
    data = badge_input(["a", "b", "c", "d"])
    scores = BadgeGradientNormScorer().score(data)
    with pytest.raises(ValueError, match="row order"):
        BadgeSelector(data, seed=5).select(
            ids=["b", "a", "c", "d"], scores=scores, k=1,
            pool=Pool.from_ids(["a", "b", "c", "d"]),
        )


def test_core_set_rejects_reordered_ids_even_if_scores_match():
    data = core_input(["a", "b", "c", "d"])
    scores = CoreSetScorer().score(data)
    with pytest.raises(ValueError, match="row order"):
        CoreSetSelector(data).select(
            ids=["b", "a", "c", "d"], scores=scores, k=1,
            pool=Pool.from_ids(["a", "b", "c", "d"]),
        )


def test_badge_duplicate_candidate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        make_badge(p([[.5,.5], [.5,.5]]), [[0.], [1.]], candidate_ids=["a", "a"])


def test_core_set_duplicate_labeled_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        make_core([[0.]], [[1.], [2.]], labeled_ids=["L", "L"])
