"""Inference rows with astronomical alert provenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from al_pipeline.core.types import ClassificationPredictions


@dataclass(frozen=True)
class ClassificationInferenceResult:
    """Classifier predictions together with stable sample provenance."""

    object_ids: tuple[str, ...]
    alerts: tuple[int | None, ...]
    predictions: ClassificationPredictions
    target_indices: tuple[int, ...] | None = None
    embeddings: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_rows = self.predictions.values.shape[0]

        if len(self.object_ids) != n_rows:
            raise ValueError("object_ids and predictions must contain the same rows.")

        if len(self.alerts) != n_rows:
            raise ValueError("alerts and predictions must contain the same rows.")

        if self.target_indices is not None and len(self.target_indices) != n_rows:
            raise ValueError("target_indices and predictions must contain the same rows.")

        if self.embeddings is not None:
            embeddings = np.asarray(self.embeddings)

            if embeddings.ndim != 2:
                raise ValueError(
                    "embeddings must be two-dimensional with shape (n_objects, embedding_dim)."
                )

            if embeddings.shape[0] != n_rows:
                raise ValueError("embeddings and predictions must contain the same rows.")

            if embeddings.shape[1] < 1:
                raise ValueError("embeddings must contain at least one feature.")

            if not np.issubdtype(embeddings.dtype, np.number):
                raise TypeError("embeddings must be numeric.")

            if not np.all(np.isfinite(embeddings)):
                raise ValueError("embeddings must contain only finite values.")

            object.__setattr__(self, "embeddings", embeddings)
