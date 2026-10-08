"""Random baseline scoring."""

from __future__ import annotations

import numpy as np

from al_pipeline.core.scoring.base import Scorer


class RandomScorer(Scorer[int]):
    """Assign random scores to candidate objects.

    Random scoring provides a baseline against which informed scorers can be
    compared.

    Parameters
    ----------
    rng
        NumPy random-number generator used to produce scores. Supplying the
        generator externally makes experiments reproducible without relying
        on global random state.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def score(self, n_candidates: int) -> np.ndarray:
        """Return one random score per candidate object.

        Parameters
        ----------
        n_candidates
            Number of candidate objects to score.

        Returns
        -------
        np.ndarray
            One-dimensional array of random scores in the interval
            ``[0, 1)``.

        Raises
        ------
        ValueError
            If ``n_candidates`` is negative.
        """

        if n_candidates < 0:
            raise ValueError("n_candidates must be non-negative.")

        return self._rng.random(n_candidates)
