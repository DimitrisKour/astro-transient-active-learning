"""BALD / mutual-information scorer for coherent posterior predictive draws.

Rows correspond to unlabeled objects, second axis to the SAME sampled model
across *all* objects, last axis to class. No target labels are accessed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BALDInput:
    probabilities: np.ndarray  # (objects, shared_model_draws, classes)

    def __post_init__(self):
        p = np.asarray(self.probabilities, dtype=np.float64)
        if p.ndim != 3 or min(p.shape) < 1 or p.shape[1] < 2 or p.shape[2] < 2:
            raise ValueError('BALD probabilities require (N>=1, K>=2, C>=2)')
        if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
            raise ValueError('BALD probabilities must be finite in [0,1]')
        if not np.allclose(p.sum(axis=2), 1, atol=1e-6, rtol=1e-6):
            raise ValueError('Each stochastic prediction must sum to one')
        object.__setattr__(self, 'probabilities', p)


def entropy(p):
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.), axis=-1)


class BALDScorer:
    def score(self, data: BALDInput) -> np.ndarray:
        if not isinstance(data, BALDInput):
            raise TypeError('Expected BALDInput with coherent stochastic draws')
        p = data.probabilities
        mi = entropy(p.mean(axis=1)) - entropy(p).mean(axis=1)
        if not np.all(np.isfinite(mi)) or np.any(mi < -1e-8):
            raise ValueError('Invalid mutual information estimates')
        return np.maximum(mi, 0.0)


__all__ = ['BALDInput', 'BALDScorer', 'entropy']
