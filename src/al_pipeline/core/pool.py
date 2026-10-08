"""Active-learning pool state management."""

from __future__ import annotations

from collections.abc import Iterable


class Pool:
    """Track labeled, unlabeled, pending, and excluded object IDs.

    The pool stores only logical active-learning state. It does not store
    sample data, model outputs, or labels.

    Parameters
    ----------
    labeled_ids
        Object IDs whose labels are currently available for training.

    unlabeled_ids
        Object IDs eligible for acquisition.

    pending_ids
        Object IDs already selected for acquisition but whose labels have
        not yet been incorporated into the labeled set.

    excluded_ids
        Object IDs that remain part of the experiment population but are no
        longer eligible for acquisition or training under the current policy.
        Scientific status and pool eligibility are intentionally distinct: an
        object outside a classifier's present taxonomy is not automatically
        excluded, because it may still be valuable for anomaly review or
        future taxonomy expansion.
    """

    __slots__ = (
        "_labeled_ids",
        "_unlabeled_ids",
        "_pending_ids",
        "_excluded_ids",
    )

    def __init__(
        self,
        *,
        labeled_ids: Iterable[str] = (),
        unlabeled_ids: Iterable[str] = (),
        pending_ids: Iterable[str] = (),
        excluded_ids: Iterable[str] = (),
    ) -> None:
        self._labeled_ids = self._validate_ids(
            labeled_ids,
            "labeled_ids",
        )

        self._unlabeled_ids = self._validate_ids(
            unlabeled_ids,
            "unlabeled_ids",
        )

        self._pending_ids = self._validate_ids(
            pending_ids,
            "pending_ids",
        )

        self._excluded_ids = self._validate_ids(
            excluded_ids,
            "excluded_ids",
        )

        self._validate_disjoint()

    @classmethod
    def from_ids(
        cls,
        all_ids: Iterable[str],
        *,
        labeled_ids: Iterable[str] = (),
        excluded_ids: Iterable[str] = (),
    ) -> Pool:
        """Create a pool from a complete population and known subsets."""

        all_ids_set = cls._validate_ids(
            all_ids,
            "all_ids",
        )

        labeled_ids_set = cls._validate_ids(
            labeled_ids,
            "labeled_ids",
        )

        excluded_ids_set = cls._validate_ids(
            excluded_ids,
            "excluded_ids",
        )

        unknown_labeled = labeled_ids_set - all_ids_set

        if unknown_labeled:
            raise ValueError(
                f"labeled_ids must be a subset of all_ids. Unknown IDs: {sorted(unknown_labeled)}"
            )

        unknown_excluded = excluded_ids_set - all_ids_set

        if unknown_excluded:
            raise ValueError(
                f"excluded_ids must be a subset of all_ids. Unknown IDs: {sorted(unknown_excluded)}"
            )

        overlap = labeled_ids_set & excluded_ids_set

        if overlap:
            raise ValueError(
                f"labeled_ids and excluded_ids must be disjoint. Overlapping IDs: {sorted(overlap)}"
            )

        return cls(
            labeled_ids=labeled_ids_set,
            unlabeled_ids=(all_ids_set - labeled_ids_set - excluded_ids_set),
            excluded_ids=excluded_ids_set,
        )

    @property
    def labeled_ids(self) -> frozenset[str]:
        """Return the currently labeled object IDs."""

        return frozenset(self._labeled_ids)

    @property
    def unlabeled_ids(self) -> frozenset[str]:
        """Return object IDs currently eligible for acquisition."""

        return frozenset(self._unlabeled_ids)

    @property
    def pending_ids(self) -> frozenset[str]:
        """Return selected object IDs awaiting label resolution."""

        return frozenset(self._pending_ids)

    @property
    def excluded_ids(self) -> frozenset[str]:
        """Return IDs made ineligible under the current pool policy."""

        return frozenset(self._excluded_ids)

    def mark_pending(
        self,
        ids: Iterable[str],
    ) -> None:
        """Move selected objects from unlabeled to pending state."""

        ids_set = self._validate_ids(
            ids,
            "ids",
        )

        unavailable = ids_set - self._unlabeled_ids

        if unavailable:
            raise ValueError(
                "Only unlabeled objects can be marked as pending. "
                f"Unavailable IDs: {sorted(unavailable)}"
            )

        self._unlabeled_ids.difference_update(ids_set)

        self._pending_ids.update(ids_set)

    def resolve(
        self,
        ids: Iterable[str],
    ) -> None:
        """Move successfully labeled objects from pending to labeled."""

        ids_set = self._validate_ids(
            ids,
            "ids",
        )

        not_pending = ids_set - self._pending_ids

        if not_pending:
            raise ValueError(
                f"Only pending objects can be resolved. Non-pending IDs: {sorted(not_pending)}"
            )

        self._pending_ids.difference_update(ids_set)

        self._labeled_ids.update(ids_set)

    def release(
        self,
        ids: Iterable[str],
    ) -> None:
        """Move unresolved objects from pending back to unlabeled."""

        ids_set = self._validate_ids(
            ids,
            "ids",
        )

        not_pending = ids_set - self._pending_ids

        if not_pending:
            raise ValueError(
                f"Only pending objects can be released. Non-pending IDs: {sorted(not_pending)}"
            )

        self._pending_ids.difference_update(ids_set)

        self._unlabeled_ids.update(ids_set)

    def exclude(
        self,
        ids: Iterable[str],
    ) -> None:
        """Move pending objects into the policy-excluded state.

        Exclusion is an operational decision, not a scientific label. An
        outside-taxonomy resolution may be excluded by one workflow or kept
        for anomaly review by another.
        """

        ids_set = self._validate_ids(
            ids,
            "ids",
        )

        not_pending = ids_set - self._pending_ids

        if not_pending:
            raise ValueError(
                f"Only pending objects can be excluded. Non-pending IDs: {sorted(not_pending)}"
            )

        self._pending_ids.difference_update(ids_set)

        self._excluded_ids.update(ids_set)

    def __len__(self) -> int:
        """Return the total number of objects represented by the pool."""

        return (
            len(self._labeled_ids)
            + len(self._unlabeled_ids)
            + len(self._pending_ids)
            + len(self._excluded_ids)
        )

    @staticmethod
    def _validate_ids(
        ids: Iterable[str],
        name: str,
    ) -> set[str]:
        """Validate an iterable of unique string object IDs."""

        ids_list = list(ids)

        if not all(
            isinstance(
                object_id,
                str,
            )
            for object_id in ids_list
        ):
            raise TypeError(f"{name} must contain only strings.")

        if len(ids_list) != len(set(ids_list)):
            raise ValueError(f"{name} must not contain duplicate IDs.")

        return set(ids_list)

    def _validate_disjoint(self) -> None:
        """Ensure that all logical pool states are mutually disjoint."""

        states = {
            "labeled_ids": self._labeled_ids,
            "unlabeled_ids": self._unlabeled_ids,
            "pending_ids": self._pending_ids,
            "excluded_ids": self._excluded_ids,
        }

        names = list(states)

        for index, first_name in enumerate(names):
            for second_name in names[index + 1 :]:
                if states[first_name] & states[second_name]:
                    raise ValueError(f"{first_name} and {second_name} must be disjoint.")
