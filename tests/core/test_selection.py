"""Tests for active-learning candidate selection."""

import numpy as np
import pytest

from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.pool import Pool


def test_top_k_selector_returns_highest_scoring_objects():
    pool = Pool.from_ids(["a", "b", "c", "d"])

    selector = TopKSelector()

    selection = selector.select(
        ids=["a", "b", "c", "d"],
        scores=np.array([0.1, 0.9, 0.4, 0.8]),
        k=2,
        pool=pool,
    )

    assert selection.ids == ["b", "d"]
    assert np.array_equal(
        selection.indices,
        np.array([1, 3]),
    )
    assert np.allclose(
        selection.scores,
        np.array([0.9, 0.8]),
    )


def test_top_k_selector_excludes_labeled_objects():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["b"],
    )

    selector = TopKSelector()

    selection = selector.select(
        ids=["a", "b", "c"],
        scores=np.array([0.5, 0.99, 0.8]),
        k=2,
        pool=pool,
    )

    assert selection.ids == ["c", "a"]


def test_top_k_selector_excludes_pending_objects():
    pool = Pool.from_ids(["a", "b", "c"])

    pool.mark_pending(["b"])

    selector = TopKSelector()

    selection = selector.select(
        ids=["a", "b", "c"],
        scores=np.array([0.5, 0.99, 0.8]),
        k=2,
        pool=pool,
    )

    assert selection.ids == ["c", "a"]


def test_top_k_selector_rejects_k_larger_than_eligible_pool():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["a"],
    )

    selector = TopKSelector()

    with pytest.raises(
        ValueError,
        match="only 2 eligible candidates",
    ):
        selector.select(
            ids=["a", "b", "c"],
            scores=np.array([0.1, 0.2, 0.3]),
            k=3,
            pool=pool,
        )


def test_top_k_selector_supports_zero_selection():
    pool = Pool.from_ids(["a", "b"])

    selector = TopKSelector()

    selection = selector.select(
        ids=["a", "b"],
        scores=np.array([0.9, 0.8]),
        k=0,
        pool=pool,
    )

    assert len(selection) == 0
    assert selection.ids == []
    assert selection.indices.size == 0
    assert selection.scores.size == 0


def test_top_k_selector_rejects_mismatched_lengths():
    pool = Pool.from_ids(["a", "b"])

    selector = TopKSelector()

    with pytest.raises(
        ValueError,
        match="same number of candidates",
    ):
        selector.select(
            ids=["a", "b"],
            scores=np.array([0.9]),
            k=1,
            pool=pool,
        )


def test_top_k_selector_rejects_duplicate_ids():
    pool = Pool.from_ids(["a", "b"])

    selector = TopKSelector()

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        selector.select(
            ids=["a", "a"],
            scores=np.array([0.9, 0.8]),
            k=1,
            pool=pool,
        )


def test_top_k_selector_rejects_nonfinite_scores():
    pool = Pool.from_ids(["a", "b"])

    selector = TopKSelector()

    with pytest.raises(
        ValueError,
        match="only finite values",
    ):
        selector.select(
            ids=["a", "b"],
            scores=np.array([0.9, np.nan]),
            k=1,
            pool=pool,
        )


def test_top_k_selector_breaks_score_ties_by_original_position():
    pool = Pool.from_ids(["a", "b", "c"])

    selector = TopKSelector()

    selection = selector.select(
        ids=["a", "b", "c"],
        scores=np.array([0.8, 0.8, 0.8]),
        k=2,
        pool=pool,
    )

    assert selection.ids == ["a", "b"]
    assert np.array_equal(
        selection.indices,
        np.array([0, 1]),
    )
