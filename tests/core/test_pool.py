"""Tests for active-learning pool state management."""

import pytest

from al_pipeline.core.pool import Pool


def test_pool_from_ids_creates_expected_initial_state():
    pool = Pool.from_ids(
        ["a", "b", "c", "d"],
        labeled_ids=["a", "b"],
    )

    assert pool.labeled_ids == frozenset({"a", "b"})
    assert pool.unlabeled_ids == frozenset({"c", "d"})
    assert pool.pending_ids == frozenset()
    assert len(pool) == 4


def test_pool_mark_pending_moves_unlabeled_to_pending():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["a"],
    )

    pool.mark_pending(["b"])

    assert pool.labeled_ids == frozenset({"a"})
    assert pool.unlabeled_ids == frozenset({"c"})
    assert pool.pending_ids == frozenset({"b"})


def test_pool_resolve_moves_pending_to_labeled():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["a"],
    )

    pool.mark_pending(["b"])
    pool.resolve(["b"])

    assert pool.labeled_ids == frozenset({"a", "b"})
    assert pool.unlabeled_ids == frozenset({"c"})
    assert pool.pending_ids == frozenset()


def test_pool_release_moves_pending_back_to_unlabeled():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["a"],
    )

    pool.mark_pending(["b"])
    pool.release(["b"])

    assert pool.labeled_ids == frozenset({"a"})
    assert pool.unlabeled_ids == frozenset({"b", "c"})
    assert pool.pending_ids == frozenset()


def test_pool_rejects_overlapping_initial_states():
    with pytest.raises(
        ValueError,
        match="labeled_ids and unlabeled_ids must be disjoint",
    ):
        Pool(
            labeled_ids=["a"],
            unlabeled_ids=["a", "b"],
        )


def test_pool_rejects_labeled_ids_outside_all_ids():
    with pytest.raises(
        ValueError,
        match="labeled_ids must be a subset of all_ids",
    ):
        Pool.from_ids(
            ["a", "b"],
            labeled_ids=["a", "c"],
        )


def test_pool_rejects_marking_non_unlabeled_objects_as_pending():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["a"],
    )

    with pytest.raises(
        ValueError,
        match="Only unlabeled objects can be marked as pending",
    ):
        pool.mark_pending(["a"])


def test_pool_rejects_resolving_non_pending_objects():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["a"],
    )

    with pytest.raises(
        ValueError,
        match="Only pending objects can be resolved",
    ):
        pool.resolve(["b"])


def test_pool_rejects_releasing_non_pending_objects():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["a"],
    )

    with pytest.raises(
        ValueError,
        match="Only pending objects can be released",
    ):
        pool.release(["b"])


def test_pool_rejects_duplicate_ids():
    with pytest.raises(
        ValueError,
        match="must not contain duplicate IDs",
    ):
        Pool(
            unlabeled_ids=["a", "a"],
        )


def test_pool_rejects_non_string_ids():
    with pytest.raises(
        TypeError,
        match="must contain only strings",
    ):
        Pool(
            unlabeled_ids=["a", 2],
        )


def test_pool_from_ids_supports_excluded_objects():
    pool = Pool.from_ids(
        ["a", "b", "c", "d"],
        labeled_ids=["a"],
        excluded_ids=["d"],
    )

    assert pool.labeled_ids == frozenset({"a"})
    assert pool.unlabeled_ids == frozenset({"b", "c"})
    assert pool.pending_ids == frozenset()
    assert pool.excluded_ids == frozenset({"d"})
    assert len(pool) == 4


def test_pool_exclude_moves_pending_to_excluded():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["a"],
    )

    pool.mark_pending(["b"])
    pool.exclude(["b"])

    assert pool.labeled_ids == frozenset({"a"})
    assert pool.unlabeled_ids == frozenset({"c"})
    assert pool.pending_ids == frozenset()
    assert pool.excluded_ids == frozenset({"b"})


def test_pool_rejects_excluding_non_pending_objects():
    pool = Pool.from_ids(
        ["a", "b", "c"],
        labeled_ids=["a"],
    )

    with pytest.raises(
        ValueError,
        match="Only pending objects can be excluded",
    ):
        pool.exclude(["b"])


def test_pool_rejects_overlapping_excluded_state():
    with pytest.raises(
        ValueError,
        match="unlabeled_ids and excluded_ids must be disjoint",
    ):
        Pool(
            unlabeled_ids=["a", "b"],
            excluded_ids=["b"],
        )


def test_pool_rejects_excluded_ids_outside_all_ids():
    with pytest.raises(
        ValueError,
        match="excluded_ids must be a subset of all_ids",
    ):
        Pool.from_ids(
            ["a", "b"],
            excluded_ids=["c"],
        )


def test_pool_rejects_labeled_and_excluded_overlap():
    with pytest.raises(
        ValueError,
        match="labeled_ids and excluded_ids must be disjoint",
    ):
        Pool.from_ids(
            ["a", "b"],
            labeled_ids=["a"],
            excluded_ids=["a"],
        )
