"""Tests for active-learning loop orchestration."""

import numpy as np

from al_pipeline.core.acquisition.entropy import EntropyAcquisition
from al_pipeline.core.acquisition.random import RandomAcquisition
from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.pool import Pool
from al_pipeline.core.types import ClassificationPredictions


def test_query_returns_expected_entropy_selection():
    pool = Pool.from_ids(["a", "b", "c"])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.99, 0.01],
                [0.60, 0.40],
                [0.50, 0.50],
            ]
        ),
        output_type="probabilities",
    )

    loop = ActiveLearningLoop(
        acquisition=EntropyAcquisition(),
        selector=TopKSelector(),
        pool=pool,
    )

    selection = loop.query(
        ids=["a", "b", "c"],
        acquisition_data=predictions,
        k=2,
    )

    assert selection.ids == ["c", "b"]
    assert np.array_equal(
        selection.indices,
        np.array([2, 1]),
    )


def test_query_does_not_modify_pool():
    pool = Pool.from_ids(["a", "b", "c"])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.99, 0.01],
                [0.60, 0.40],
                [0.50, 0.50],
            ]
        ),
        output_type="probabilities",
    )

    loop = ActiveLearningLoop(
        acquisition=EntropyAcquisition(),
        selector=TopKSelector(),
        pool=pool,
    )

    loop.query(
        ids=["a", "b", "c"],
        acquisition_data=predictions,
        k=2,
    )

    assert pool.unlabeled_ids == frozenset({"a", "b", "c"})
    assert pool.pending_ids == frozenset()
    assert pool.labeled_ids == frozenset()


def test_acquire_moves_selected_objects_to_pending():
    pool = Pool.from_ids(["a", "b", "c"])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.99, 0.01],
                [0.60, 0.40],
                [0.50, 0.50],
            ]
        ),
        output_type="probabilities",
    )

    loop = ActiveLearningLoop(
        acquisition=EntropyAcquisition(),
        selector=TopKSelector(),
        pool=pool,
    )

    selection = loop.acquire(
        ids=["a", "b", "c"],
        acquisition_data=predictions,
        k=2,
    )

    assert selection.ids == ["c", "b"]

    assert pool.unlabeled_ids == frozenset({"a"})
    assert pool.pending_ids == frozenset({"b", "c"})
    assert pool.labeled_ids == frozenset()


def test_loop_respects_existing_pool_state():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["c"],
    )

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.99, 0.01],
                [0.60, 0.40],
                [0.50, 0.50],
            ]
        ),
        output_type="probabilities",
    )

    loop = ActiveLearningLoop(
        acquisition=EntropyAcquisition(),
        selector=TopKSelector(),
        pool=pool,
    )

    selection = loop.query(
        ids=["a", "b", "c"],
        acquisition_data=predictions,
        k=1,
    )

    # "c" has the highest entropy, but it is already labeled.
    assert selection.ids == ["b"]


def test_loop_supports_random_acquisition_without_predictions():
    pool = Pool.from_ids(["a", "b", "c", "d"])

    loop = ActiveLearningLoop(
        acquisition=RandomAcquisition(np.random.default_rng(42)),
        selector=TopKSelector(),
        pool=pool,
    )

    selection = loop.query(
        ids=["a", "b", "c", "d"],
        acquisition_data=4,
        k=2,
    )

    assert len(selection) == 2
    assert set(selection.ids).issubset({"a", "b", "c", "d"})
    assert selection.scores.shape == (2,)


def test_loop_does_not_select_excluded_objects():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        excluded_ids=["c"],
    )

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.99, 0.01],
                [0.60, 0.40],
                [0.50, 0.50],
            ]
        ),
        output_type="probabilities",
    )

    loop = ActiveLearningLoop(
        acquisition=EntropyAcquisition(),
        selector=TopKSelector(),
        pool=pool,
    )

    selection = loop.query(
        ids=["a", "b", "c"],
        acquisition_data=predictions,
        k=1,
    )

    # "c" has the highest entropy but is permanently excluded.
    assert selection.ids == ["b"]


def test_loop_accepts_canonical_scorer_keyword():
    from al_pipeline.core.scoring.entropy import EntropyScorer

    pool = Pool.from_ids(["a", "b", "c"])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.99, 0.01],
                [0.60, 0.40],
                [0.50, 0.50],
            ]
        ),
        output_type="probabilities",
    )

    loop = ActiveLearningLoop(
        scorer=EntropyScorer(),
        selector=TopKSelector(),
        pool=pool,
    )

    selection = loop.query(
        ids=["a", "b", "c"],
        acquisition_data=predictions,
        k=2,
    )

    assert selection.ids == ["c", "b"]


def test_loop_rejects_scorer_and_legacy_acquisition_together():
    from al_pipeline.core.scoring.entropy import EntropyScorer

    pool = Pool.from_ids(["a", "b"])

    with np.testing.assert_raises_regex(
        TypeError,
        "only one of scorer or acquisition",
    ):
        ActiveLearningLoop(
            scorer=EntropyScorer(),
            acquisition=EntropyAcquisition(),
            selector=TopKSelector(),
            pool=pool,
        )
