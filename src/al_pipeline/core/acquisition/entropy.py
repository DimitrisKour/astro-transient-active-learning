"""Backward-compatible predictive-entropy acquisition name."""

from al_pipeline.core.scoring.entropy import EntropyScorer


class EntropyAcquisition(EntropyScorer):
    """Compatibility name for :class:`EntropyScorer`."""


__all__ = ["EntropyAcquisition"]
