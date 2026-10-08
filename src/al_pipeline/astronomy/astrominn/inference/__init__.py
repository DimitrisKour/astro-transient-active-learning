"""Model inference adapters for adaptive learning."""

from al_pipeline.astronomy.astrominn.inference.aggregation import (
    AggregationMethod,
    ObjectPredictionResult,
    aggregate_object_predictions,
)
from al_pipeline.astronomy.astrominn.inference.astrominn import (
    AstroMiNNInferenceResult,
    AstroMiNNPredictor,
)
from al_pipeline.astronomy.astrominn.inference.base import ClassificationInferenceResult

__all__ = [
    "AggregationMethod",
    "AstroMiNNInferenceResult",
    "AstroMiNNPredictor",
    "ClassificationInferenceResult",
    "ObjectPredictionResult",
    "aggregate_object_predictions",
]
