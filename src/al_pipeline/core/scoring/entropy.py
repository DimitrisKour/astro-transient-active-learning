"""Predictive-entropy scoring."""

from __future__ import annotations

import numpy as np

from al_pipeline.core.scoring.base import Scorer
from al_pipeline.core.types import ClassificationPredictions


class EntropyScorer(Scorer[ClassificationPredictions]):
    """Score candidates using predictive entropy.

    Objects with larger predictive entropy receive larger scores and
    therefore have higher priority under a top-k selector.
    """

    def score(
        self,
        data: ClassificationPredictions,
    ) -> np.ndarray:
        """Return predictive entropy for each candidate object.

        Parameters
        ----------
        data
            Classifier outputs with shape ``(n_objects, n_classes)``.
            Outputs may be either logits or class probabilities.

        Returns
        -------
        np.ndarray
            One-dimensional array containing one entropy score per object.
        """

        probabilities = data.as_probabilities()

        terms = np.zeros_like(
            probabilities,
            dtype=float,
        )

        nonzero = probabilities > 0.0

        terms[nonzero] = probabilities[nonzero] * np.log(probabilities[nonzero])

        return -np.sum(terms, axis=1)


# TODO:
# If scores ever need to be compared across classifiers with different
# numbers of output classes, consider optional normalization by
# log(n_classes). Standard entropy is sufficient for ranking candidates
# within a single classifier/pool.
