"""Explicit non-executable boundaries for Tasks 5B and 5C.

These interfaces reserve a place for representation adaptation and subsequent
closed-loop evaluation without prematurely deciding a scientific method.
No training scheduler, optimizer, or acquisition runner is implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class ObjectRepresentationExtractor(Protocol):
    """Feature extraction contract for fixed object-level splits."""

    def extract(self, *, object_ids: tuple[str, ...], checkpoint: Path) -> np.ndarray:
        """Return finite matrix of shape (len(object_ids), embedding_dimension)."""


@dataclass(frozen=True, slots=True)
class RepresentationAdaptationDesign:
    """Task 5B placeholder; validated 5A baseline required before planning."""

    baseline_report: Path
    method: str
    frozen_experts: bool = True
    fixed_acquisition_manifests: bool = True

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("Adaptation method must be scientifically specified")
        if not self.fixed_acquisition_manifests:
            raise ValueError("5B must isolate representation adaptation using fixed acquisition")


@dataclass(frozen=True, slots=True)
class RepresentationClosedLoopDesign:
    """Task 5C placeholder; Task 4 and fixed-manifest 5B evidence required."""

    task4_report: Path
    task5a_report: Path
    task5b_report: Path
    policy: str
    train_budget: int

    def __post_init__(self) -> None:
        if not self.policy or self.train_budget < 1:
            raise ValueError("Closed-loop plan needs policy and positive object budget")
