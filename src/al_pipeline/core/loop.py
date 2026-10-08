"""Core orchestration for candidate scoring and selection."""

from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np

from al_pipeline.core.scoring.base import Scorer
from al_pipeline.core.selection import SelectionStrategy
from al_pipeline.core.pool import Pool
from al_pipeline.core.types import SelectionBatch


ScoringInputT = TypeVar("ScoringInputT")


class ActiveLearningLoop(Generic[ScoringInputT]):
    """Coordinate candidate scoring and selection.

    The loop is deliberately independent of models, datasets, training,
    providers, and oracle implementations. The caller provides the input
    required by the chosen scorer.

    Parameters
    ----------
    scorer
        Scorer used to assign one priority score to each candidate.

    selector
        Strategy used to choose objects from the resulting scores.

    pool
        Current logical active-learning pool state.

    acquisition
        Backward-compatible name for ``scorer``. Existing notebooks may
        continue to pass ``acquisition=...``. New code should use
        ``scorer=...``.
    """

    def __init__(
        self,
        *,
        selector: SelectionStrategy,
        pool: Pool,
        scorer: Scorer[ScoringInputT] | None = None,
        acquisition: Scorer[ScoringInputT] | None = None,
    ) -> None:
        if scorer is None and acquisition is None:
            raise TypeError("Either scorer or acquisition must be provided.")

        if scorer is not None and acquisition is not None:
            raise TypeError("Provide only one of scorer or acquisition, not both.")

        resolved_scorer = scorer if scorer is not None else acquisition

        # The checks above guarantee a scorer is available. Keeping the
        # assignment explicit avoids strategy-specific branching elsewhere.
        assert resolved_scorer is not None

        self.scorer = resolved_scorer
        self.selector = selector
        self.pool = pool

        # Compatibility attribute for existing notebook/code access. New code
        # should use ``loop.scorer``.
        self.acquisition = resolved_scorer

    def query(
        self,
        *,
        ids: list[str] | np.ndarray,
        acquisition_data: ScoringInputT,
        k: int,
    ) -> SelectionBatch:
        """Score candidates and return a proposed selection.

        This method does not modify the pool. It can therefore be used to
        inspect or compare proposed selections before committing them.

        Parameters
        ----------
        ids
            Stable object identifiers corresponding to the candidates being
            scored.

        acquisition_data
            Input required by the chosen scorer. The parameter name is kept
            for backward compatibility; its type and interpretation are
            scorer-specific.

        k
            Number of objects to select.

        Returns
        -------
        SelectionBatch
            Proposed candidate selection.
        """

        scores = self.scorer.score(acquisition_data)

        return self.selector.select(
            ids=ids,
            scores=scores,
            k=k,
            pool=self.pool,
        )

    def acquire(
        self,
        *,
        ids: list[str] | np.ndarray,
        acquisition_data: ScoringInputT,
        k: int,
    ) -> SelectionBatch:
        """Select candidates and move them into the pending pool.

        This performs a committed acquisition step:

        ``unlabeled -> pending``.

        Returns
        -------
        SelectionBatch
            The selected objects and their scores.
        """

        selection = self.query(
            ids=ids,
            acquisition_data=acquisition_data,
            k=k,
        )

        self.pool.mark_pending(selection.ids)

        return selection


# TODO:
# Once existing notebooks have migrated to scoring terminology, consider
# adding ``scoring_data`` as the canonical query/acquire parameter while
# retaining ``acquisition_data`` as a compatibility path if needed.
