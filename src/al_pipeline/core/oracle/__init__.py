"""Model-independent oracle interfaces and utilities."""

from al_pipeline.core.oracle.rules import EvidenceResolutionRule
from al_pipeline.core.oracle.existing_labels import ExistingLabelsOracle
from al_pipeline.core.oracle.base import LabelProvider
from al_pipeline.core.oracle.resolution import LabelResolution
from al_pipeline.core.oracle.resolution import LabelResolutionBatch
from al_pipeline.core.oracle.base import LabelResolver
from al_pipeline.core.oracle.resolution import LabelStatus
from al_pipeline.core.oracle.rules import ResolutionClaim
from al_pipeline.core.oracle.rule_based import RuleBasedLabelResolver

__all__ = [
    "EvidenceResolutionRule",
    "ExistingLabelsOracle",
    "LabelProvider",
    "LabelResolution",
    "LabelResolutionBatch",
    "LabelResolver",
    "LabelStatus",
    "ResolutionClaim",
    "RuleBasedLabelResolver",
]
