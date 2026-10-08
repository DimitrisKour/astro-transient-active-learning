"""Aggregation of sample-level classifier outputs to object-level outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from al_pipeline.core.types import ClassificationPredictions


AggregationMethod = Literal[
    "latest",
    "mean_probability",
]


@dataclass(frozen=True)
class ObjectPredictionResult:
    """One classifier prediction per unique candidate object."""

    object_ids: tuple[str, ...]
    predictions: ClassificationPredictions
    source_indices: tuple[int, ...] | None = None


def aggregate_object_predictions(
    *,
    object_ids: tuple[str, ...],
    predictions: ClassificationPredictions,
    alerts: tuple[int | None, ...] | None = None,
    method: AggregationMethod = "latest",
) -> ObjectPredictionResult:
    """Aggregate sample-level predictions to one prediction per object.

    Parameters
    ----------
    object_ids
        Object identifier corresponding to every prediction row.

    predictions
        Sample-level classifier outputs.

    alerts
        Alert number corresponding to every prediction row. Required for
        ``method="latest"``.

    method
        ``"latest"`` keeps the prediction from the largest alert number
        for each object.

        ``"mean_probability"`` converts each sample prediction to class
        probabilities and averages probabilities across all samples
        belonging to the same object.
    """

    values = predictions.values

    if len(object_ids) != values.shape[0]:
        raise ValueError("object_ids and prediction rows must have the same length.")

    if alerts is not None and len(alerts) != len(object_ids):
        raise ValueError("alerts and object_ids must have the same length.")

    # Preserve first-seen object ordering.
    grouped_indices: dict[str, list[int]] = {}

    for index, object_id in enumerate(object_ids):
        grouped_indices.setdefault(
            object_id,
            [],
        ).append(index)

    ordered_object_ids = tuple(grouped_indices)

    if method == "latest":
        if alerts is None:
            raise ValueError("Alert numbers are required for latest-alert aggregation.")

        selected_indices: list[int] = []

        for object_id in ordered_object_ids:
            indices = grouped_indices[object_id]

            object_alerts = [alerts[index] for index in indices]

            if any(alert is None for alert in object_alerts):
                raise ValueError(f"Object {object_id!r} is missing an alert number.")

            latest_alert = max(object_alerts)

            latest_indices = [index for index in indices if alerts[index] == latest_alert]

            if len(latest_indices) != 1:
                raise ValueError(
                    f"Object {object_id!r} has multiple samples for latest alert {latest_alert!r}."
                )

            selected_indices.append(latest_indices[0])

        aggregated_values = values[selected_indices]

        return ObjectPredictionResult(
            object_ids=ordered_object_ids,
            predictions=ClassificationPredictions(
                values=aggregated_values,
                output_type=predictions.output_type,
            ),
            source_indices=tuple(selected_indices),
        )

    if method == "mean_probability":
        probabilities = predictions.as_probabilities()

        aggregated_values = np.stack(
            [
                probabilities[grouped_indices[object_id]].mean(axis=0)
                for object_id in ordered_object_ids
            ],
            axis=0,
        )

        return ObjectPredictionResult(
            object_ids=ordered_object_ids,
            predictions=ClassificationPredictions(
                values=aggregated_values,
                output_type="probabilities",
            ),
            source_indices=None,
        )

    raise ValueError(f"Unsupported aggregation method: {method!r}.")
