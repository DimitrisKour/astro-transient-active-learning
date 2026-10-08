"""Tests for active-learning label providers."""

import numpy as np
import pytest

from al_pipeline.core.acquisition.entropy import EntropyAcquisition
from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.oracle.existing_labels import ExistingLabelsOracle
from al_pipeline.core.pool import Pool
from al_pipeline.core.types import ClassificationPredictions, SelectionBatch


def test_existing_labels_oracle_returns_selected_labels():
    oracle = ExistingLabelsOracle(
        {
            "a": 2,
            "b": 0,
            "c": 4,
        }
    )

    selection = SelectionBatch(
        ids=["a", "c"],
        indices=np.array([0, 2]),
        scores=np.array([0.9, 0.8]),
    )

    labeled = oracle.query(selection)

    assert labeled.ids == ["a", "c"]
    assert np.array_equal(
        labeled.labels,
        np.array([2, 4]),
    )


def test_existing_labels_oracle_preserves_selection_order():
    oracle = ExistingLabelsOracle(
        {
            "a": 1,
            "b": 2,
            "c": 3,
        }
    )

    selection = SelectionBatch(
        ids=["c", "a", "b"],
        indices=np.array([2, 0, 1]),
        scores=np.array([0.9, 0.8, 0.7]),
    )

    labeled = oracle.query(selection)

    assert labeled.ids == ["c", "a", "b"]
    assert np.array_equal(
        labeled.labels,
        np.array([3, 1, 2]),
    )


def test_existing_labels_oracle_rejects_missing_labels():
    oracle = ExistingLabelsOracle(
        {
            "a": 1,
            "b": 2,
        }
    )

    selection = SelectionBatch(
        ids=["a", "c"],
        indices=np.array([0, 2]),
        scores=np.array([0.9, 0.8]),
    )

    with pytest.raises(
        KeyError,
        match="No existing label is available",
    ):
        oracle.query(selection)


def test_existing_labels_oracle_rejects_non_string_ids():
    with pytest.raises(
        TypeError,
        match="keys must contain only strings",
    ):
        ExistingLabelsOracle(
            {
                "a": 1,
                2: 0,
            }
        )


def test_existing_labels_oracle_supports_structured_labels():
    oracle = ExistingLabelsOracle(
        {
            "a": [1, 0, 0],
            "b": [0, 1, 0],
        }
    )

    selection = SelectionBatch(
        ids=["b", "a"],
        indices=np.array([1, 0]),
        scores=np.array([0.9, 0.8]),
    )

    labeled = oracle.query(selection)

    assert labeled.labels.shape == (2, 3)

    assert np.array_equal(
        labeled.labels,
        np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
            ]
        ),
    )


def test_complete_simulated_active_learning_round():
    ids = ["a", "b", "c", "d"]

    pool = Pool.from_ids(
        ids,
        labeled_ids=["a"],
    )

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.99, 0.01],
                [0.50, 0.50],
                [0.60, 0.40],
                [0.80, 0.20],
            ]
        ),
        output_type="probabilities",
    )

    loop = ActiveLearningLoop(
        acquisition=EntropyAcquisition(),
        selector=TopKSelector(),
        pool=pool,
    )

    oracle = ExistingLabelsOracle(
        {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 0,
        }
    )

    selection = loop.acquire(
        ids=ids,
        acquisition_data=predictions,
        k=2,
    )

    assert selection.ids == ["b", "c"]

    assert pool.labeled_ids == frozenset({"a"})
    assert pool.pending_ids == frozenset({"b", "c"})
    assert pool.unlabeled_ids == frozenset({"d"})

    labeled = oracle.query(selection)

    assert labeled.ids == ["b", "c"]
    assert np.array_equal(
        labeled.labels,
        np.array([1, 2]),
    )

    pool.resolve(labeled.ids)

    assert pool.labeled_ids == frozenset({"a", "b", "c"})
    assert pool.pending_ids == frozenset()
    assert pool.unlabeled_ids == frozenset({"d"})


def test_existing_labels_oracle_accepts_empty_selection():
    oracle = ExistingLabelsOracle(
        {
            "a": 1,
            "b": 2,
        }
    )

    selection = SelectionBatch(
        ids=[],
        indices=np.array([], dtype=int),
        scores=np.array([], dtype=float),
    )

    labeled = oracle.query(selection)

    assert labeled.ids == []
    assert labeled.labels.size == 0
    assert len(labeled) == 0
