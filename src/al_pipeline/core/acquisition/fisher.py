"""Backward-compatible Fisher-acquisition name."""

from al_pipeline.core.scoring.fisher import FisherScorer, FisherScoringInput


class FisherAcquisition(FisherScorer):
    """Compatibility name for :class:`FisherScorer`."""


__all__ = ["FisherAcquisition", "FisherScoringInput"]
