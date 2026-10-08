"""Backward-compatible margin-acquisition name."""

from al_pipeline.core.scoring.margin import MarginScorer


class MarginAcquisition(MarginScorer):
    """Compatibility name for :class:`MarginScorer`."""


__all__ = ["MarginAcquisition"]
