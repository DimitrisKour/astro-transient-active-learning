"""Backward-compatible acquisition-scoring interface.

New code should import :class:`Scorer` from
``al_pipeline.core.scoring``. The acquisition names are
kept so existing notebooks and downstream code continue to run unchanged.
"""

from al_pipeline.core.scoring.base import Scorer, ScoringInputT

AcquisitionInputT = ScoringInputT
AcquisitionStrategy = Scorer

__all__ = [
    "AcquisitionInputT",
    "AcquisitionStrategy",
]
