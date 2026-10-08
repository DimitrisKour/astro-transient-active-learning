"""Model-independent training interfaces and utilities."""

from al_pipeline.core.training.base import ResumableTrainingBackend
from al_pipeline.core.training.base import TrainingBackend
from al_pipeline.core.training.base import TrainingRequest
from al_pipeline.core.training.base import TrainingResult

__all__ = ["ResumableTrainingBackend", "TrainingBackend", "TrainingRequest", "TrainingResult"]
