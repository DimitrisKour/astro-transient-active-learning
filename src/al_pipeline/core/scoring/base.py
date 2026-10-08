"""Base interface for candidate scorers."""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np


ScoringInputT = TypeVar(
    "ScoringInputT",
    contravariant=True,
)


class Scorer(Protocol[ScoringInputT]):
    """Assign one priority score to each candidate object.

    Scorers are deliberately independent of candidate selection. They only
    translate strategy-specific input data into one scalar per candidate.

    Larger scores must always indicate higher priority under the scorer's
    interpretation. The input type is intentionally generic: uncertainty
    scorers may consume classifier predictions, anomaly scorers may consume
    logits or embeddings, and future scientific scorers may consume other
    representations entirely.
    """

    def score(self, data: ScoringInputT) -> np.ndarray:
        """Return one priority score per candidate object."""
        ...


# TODO:
# If several scorers require the same richer model information, introduce a
# dedicated shared input type at that point rather than creating a universal
# container full of optional model-specific fields.
