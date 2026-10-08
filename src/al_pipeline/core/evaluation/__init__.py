"""Model-independent evaluation interfaces and utilities."""

from al_pipeline.core.evaluation.history import EvaluationHistory
from al_pipeline.core.evaluation.history import EvaluationRecord
from al_pipeline.core.evaluation.base import ModelEvaluationResult
from al_pipeline.core.evaluation.base import ModelEvaluator
from al_pipeline.core.evaluation.base import SplitEvaluation

__all__ = [
    "EvaluationHistory",
    "EvaluationRecord",
    "ModelEvaluationResult",
    "ModelEvaluator",
    "SplitEvaluation",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_per_class_metrics",
]

_LAZY_EXPORTS = {
    "compute_classification_metrics": "al_pipeline.core.evaluation.metrics",
    "compute_confusion_matrix": "al_pipeline.core.evaluation.metrics",
    "compute_per_class_metrics": "al_pipeline.core.evaluation.metrics",
}


def __getattr__(name: str):
    from importlib import import_module

    try:
        module = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    return getattr(import_module(module), name)
