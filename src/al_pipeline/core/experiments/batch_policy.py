"""Serializable settings for the two experimental entropy policies."""

from dataclasses import dataclass

import numpy as np

from al_pipeline.core.scoring.class_balanced import ClassBalancedEntropyScorer


@dataclass(frozen=True, slots=True)
class BatchAcquisitionConfig:
    shortlist_multiplier: float = 3.0
    balance_alpha: float = 1.0
    balance_beta: float = 0.5
    balance_strength: float = 1.0
    balance_cap: float = 3.0
    exploration_fraction: float = 0.2

    def __post_init__(self):
        if not np.isfinite(self.shortlist_multiplier) or self.shortlist_multiplier < 1:
            raise ValueError("shortlist_multiplier must be finite and >= 1.")
        if not np.isfinite(self.exploration_fraction) or not 0 <= self.exploration_fraction <= 1:
            raise ValueError("exploration_fraction must lie in [0, 1].")
        ClassBalancedEntropyScorer(
            alpha=self.balance_alpha,
            beta=self.balance_beta,
            strength=self.balance_strength,
            cap=self.balance_cap,
        )

    def make_scorer(self):
        return ClassBalancedEntropyScorer(
            alpha=self.balance_alpha,
            beta=self.balance_beta,
            strength=self.balance_strength,
            cap=self.balance_cap,
        )
