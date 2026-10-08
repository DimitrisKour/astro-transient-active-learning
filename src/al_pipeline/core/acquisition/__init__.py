"""Backward-compatible acquisition-scoring names.

The canonical scoring API now lives in
``al_pipeline.core.scoring``.
"""

from al_pipeline.core.acquisition.base import AcquisitionStrategy
from al_pipeline.core.acquisition.entropy import EntropyAcquisition
from al_pipeline.core.acquisition.fisher import FisherAcquisition
from al_pipeline.core.acquisition.margin import MarginAcquisition
from al_pipeline.core.acquisition.random import RandomAcquisition

__all__ = [
    "AcquisitionStrategy",
    "EntropyAcquisition",
    "FisherAcquisition",
    "MarginAcquisition",
    "RandomAcquisition",
]
