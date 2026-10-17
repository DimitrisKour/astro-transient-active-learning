"""Optional, model-independent advanced acquisition policies.

None of these methods is enabled automatically in existing AL experiments.
"""

from al_pipeline.core.advanced_acquisition.badge import (
    BadgeGradientNormScorer,
    BadgeInput,
    BadgeSelector,
)
from al_pipeline.core.advanced_acquisition.core_set import (
    CoreSetInput,
    CoreSetScorer,
    CoreSetSelector,
)
from al_pipeline.core.advanced_acquisition.ensemble_disagreement import (
    EnsembleDisagreementScorer,
)

__all__ = [
    "BadgeInput", "BadgeGradientNormScorer", "BadgeSelector",
    "CoreSetInput", "CoreSetScorer", "CoreSetSelector",
    "EnsembleDisagreementScorer",
]
