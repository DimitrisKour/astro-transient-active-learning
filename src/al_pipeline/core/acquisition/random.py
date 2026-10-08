"""Backward-compatible random-acquisition name."""

from al_pipeline.core.scoring.random import RandomScorer


class RandomAcquisition(RandomScorer):
    """Compatibility name for :class:`RandomScorer`."""


__all__ = ["RandomAcquisition"]
