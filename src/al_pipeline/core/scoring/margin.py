"""Margin-based uncertainty scoring."""

from __future__ import annotations

import numpy as np

from al_pipeline.core.scoring.base import Scorer
from al_pipeline.core.types import ClassificationPredictions


class MarginScorer(Scorer[ClassificationPredictions]):
    """Score candidates using classification-margin uncertainty.

    The margin is the difference between the two largest predicted class
    probabilities. Smaller margins correspond to greater uncertainty.

    Scores are negated so that larger scores always indicate higher
    priority.
    """

    def score(
        self,
        data: ClassificationPredictions,
    ) -> np.ndarray:
        """Return margin-based scores for each candidate.

        Parameters
        ----------
        data
            Classifier outputs with shape ``(n_objects, n_classes)``.
            Outputs may be either logits or class probabilities.

        Returns
        -------
        np.ndarray
            One-dimensional array containing one score per candidate.
        """

        probabilities = data.as_probabilities()

        top_two = np.partition(
            probabilities,
            kth=-2,
            axis=1,
        )[:, -2:]

        margins = np.abs(top_two[:, 1] - top_two[:, 0])

        return -margins
