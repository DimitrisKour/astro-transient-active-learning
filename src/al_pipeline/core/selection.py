"""Selection of objects from acquisition scores."""

from __future__ import annotations

import numpy as np

from typing import Protocol

from al_pipeline.core.pool import Pool
from al_pipeline.core.types import SelectionBatch


class SelectionStrategy(Protocol):
    """Interface implemented by candidate-selection strategies."""

    def select(
        self,
        *,
        ids: list[str] | np.ndarray,
        scores: np.ndarray,
        k: int,
        pool: Pool,
    ) -> SelectionBatch:
        """Select candidate objects from acquisition scores."""
        ...


class TopKSelector(SelectionStrategy):
    """Select the highest-scoring eligible candidate objects.

    Only objects currently belonging to the pool's unlabeled set are
    eligible for selection.

    Ties are resolved deterministically using the candidates' original
    positions in the input arrays.
    """

    def select(
        self,
        *,
        ids: list[str] | np.ndarray,
        scores: np.ndarray,
        k: int,
        pool: Pool,
    ) -> SelectionBatch:
        """Return the ``k`` highest-scoring eligible objects.

        Parameters
        ----------
        ids
            Stable object identifiers corresponding to the scored
            candidates.

        scores
            One-dimensional array containing one acquisition score per
            candidate. Larger scores indicate higher acquisition priority.

        k
            Number of objects to select.

        pool
            Current active-learning pool. Only objects in
            ``pool.unlabeled_ids`` are eligible.

        Returns
        -------
        SelectionBatch
            IDs, original candidate indices, and acquisition scores for the
            selected objects.

        Raises
        ------
        ValueError
            If the inputs are inconsistent, contain invalid scores, or
            fewer than ``k`` eligible objects are available.
        """

        ids_array = np.asarray(ids, dtype=object)
        scores_array = np.asarray(scores, dtype=float)

        self._validate_inputs(
            ids=ids_array,
            scores=scores_array,
            k=k,
        )

        eligible_mask = np.fromiter(
            (object_id in pool.unlabeled_ids for object_id in ids_array),
            dtype=bool,
            count=len(ids_array),
        )

        eligible_indices = np.flatnonzero(eligible_mask)

        if k > len(eligible_indices):
            raise ValueError(
                f"Cannot select {k} objects: only "
                f"{len(eligible_indices)} eligible candidates are available."
            )

        selected_indices = self._top_k_indices(
            eligible_indices=eligible_indices,
            scores=scores_array,
            k=k,
        )

        return SelectionBatch(
            ids=ids_array[selected_indices].tolist(),
            indices=selected_indices,
            scores=scores_array[selected_indices],
        )

    @staticmethod
    def _validate_inputs(
        *,
        ids: np.ndarray,
        scores: np.ndarray,
        k: int,
    ) -> None:
        """Validate candidate identifiers, scores, and selection size."""

        if ids.ndim != 1:
            raise ValueError("ids must be one-dimensional.")

        if scores.ndim != 1:
            raise ValueError("scores must be one-dimensional.")

        if len(ids) != len(scores):
            raise ValueError("ids and scores must contain the same number of candidates.")

        if len(ids) != len(set(ids.tolist())):
            raise ValueError("ids must not contain duplicates.")

        if not all(isinstance(object_id, str) for object_id in ids):
            raise TypeError("ids must contain only strings.")

        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must contain only finite values.")

        if not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer.")

        if k < 0:
            raise ValueError("k must be non-negative.")

    # TODO:
    # For very large candidate pools, benchmark an
    # argpartition-based top-k implementation to avoid sorting every eligible
    # object. Preserve deterministic tie handling when doing so.

    @staticmethod
    def _top_k_indices(
        *,
        eligible_indices: np.ndarray,
        scores: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Return eligible indices ordered by decreasing acquisition score."""

        if k == 0:
            return np.empty(0, dtype=np.int64)

        eligible_scores = scores[eligible_indices]

        order = np.lexsort(
            (
                eligible_indices,
                -eligible_scores,
            )
        )

        return eligible_indices[order[:k]]


# TODO:
# If selection later needs diversity, observational constraints, quotas, or
# cost-aware optimization, introduce additional selector implementations
# rather than adding strategy-specific logic to TopKSelector.
