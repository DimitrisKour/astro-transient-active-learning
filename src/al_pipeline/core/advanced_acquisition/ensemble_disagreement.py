"""Ensemble disagreement as a conventional scalar active-learning scorer."""

from __future__ import annotations

import numpy as np

from al_pipeline.core.prediction.ensemble import PredictionEnsemble
from al_pipeline.core.scoring.base import Scorer


class EnsembleDisagreementScorer(Scorer[PredictionEnsemble]):
    """Rank objects by weighted Jensen-Shannon disagreement of ensemble members.

    Uses PredictionEnsemble.disagreement() from Package A. The score is
    entropy(mean member probabilities) minus mean member entropy, not the
    entropy of mean logits. This is an epistemic *proxy*, not proof of a
    Bayesian posterior. Caller must align ``ids`` with ensemble.object_ids.
    """

    def score(self, data: PredictionEnsemble) -> np.ndarray:
        if not isinstance(data, PredictionEnsemble):
            raise TypeError("Expected a PredictionEnsemble")
        values = np.asarray(data.disagreement(), dtype=float)
        if values.ndim != 1 or values.shape[0] != len(data.object_ids):
            raise ValueError("Ensemble disagreement returned misaligned scores")
        if not np.all(np.isfinite(values)):
            raise ValueError("Ensemble disagreement returned nonfinite scores")
        return values


__all__ = ["EnsembleDisagreementScorer"]
