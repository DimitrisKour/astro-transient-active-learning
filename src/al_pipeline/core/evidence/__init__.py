"""Model-independent evidence interfaces and utilities."""

from al_pipeline.core.evidence.types import Evidence
from al_pipeline.core.evidence.types import EvidenceBatch
from al_pipeline.core.evidence.types import EvidenceFailure
from al_pipeline.core.evidence.base import EvidenceProvider

__all__ = ["Evidence", "EvidenceBatch", "EvidenceFailure", "EvidenceProvider"]
