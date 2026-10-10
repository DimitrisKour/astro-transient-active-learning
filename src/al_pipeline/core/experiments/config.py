"""Run policies independent of datasets, models, and providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from al_pipeline.core.experiments.batch_policy import BatchAcquisitionConfig


StrategyName: TypeAlias = Literal[
    "entropy",
    "entropy_diversity",
    "entropy_balanced_explore",
    "fisher",
    "margin",
    "random",
]


TrainingInitialization: TypeAlias = Literal[
    "warm_start",
    "restart_from_baseline",
    "restart_from_random",
]


@dataclass(frozen=True, slots=True)
class SimulatedALConfig:
    """Strategy-independent and strategy-specific run policy."""

    run_id: str
    strategy: StrategyName
    acquisition_size: int = 50
    max_cycles: int = 6
    seed: int = 135
    training_initialization: TrainingInitialization = "warm_start"
    allow_partial_final_cycle: bool = False
    batch_policy: BatchAcquisitionConfig | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be a non-empty string.")

        if self.strategy not in {
            "entropy",
            "entropy_diversity",
            "entropy_balanced_explore",
            "fisher",
            "margin",
            "random",
        }:
            raise ValueError(f"Unsupported acquisition strategy: {self.strategy!r}.")

        if self.strategy in {"entropy_diversity", "entropy_balanced_explore"}:
            if self.batch_policy is None:
                object.__setattr__(self, "batch_policy", BatchAcquisitionConfig())
        elif self.batch_policy is not None:
            raise ValueError("batch_policy is only used by the two new entropy policies.")

        if self.acquisition_size <= 0:
            raise ValueError("acquisition_size must be positive.")

        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive.")

        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

        if self.training_initialization not in {
            "warm_start",
            "restart_from_baseline",
            "restart_from_random",
        }:
            raise ValueError(
                f"Unsupported training_initialization: {self.training_initialization!r}."
            )
