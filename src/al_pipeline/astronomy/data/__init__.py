"""Data management for adaptive-learning experiments."""

from al_pipeline.astronomy.data.source import CandidateSource
from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest, CandidatePoolRecord
from al_pipeline.astronomy.data.labeled import AcquisitionRecord, LabeledSetManager
from al_pipeline.astronomy.data.registry import (
    DatasetRegistry,
    DatasetSpec,
    ProfileSpec,
    ResolvedProfile,
    StorageSpec,
)
from al_pipeline.core.state.run_state import RunState

__all__ = [
    "CandidateSource",
    "CandidatePoolManifest",
    "CandidatePoolRecord",
    "AcquisitionRecord",
    "DatasetRegistry",
    "DatasetSpec",
    "LabeledSetManager",
    "ProfileSpec",
    "ResolvedProfile",
    "RunState",
    "StorageSpec",
]
