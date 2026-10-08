import numpy as np
import pytest

from al_pipeline.astronomy.astrominn.inference.aggregation import aggregate_object_predictions
from al_pipeline.core.types import ClassificationPredictions


def test_latest_uses_highest_alert_per_object():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [3.0, 1.0],
                [1.0, 4.0],
                [5.0, 0.0],
            ]
        ),
        output_type="logits",
    )

    result = aggregate_object_predictions(
        object_ids=(
            "object-a",
            "object-a",
            "object-b",
        ),
        alerts=(
            1,
            3,
            2,
        ),
        predictions=predictions,
        method="latest",
    )

    assert result.object_ids == (
        "object-a",
        "object-b",
    )

    assert result.predictions.output_type == "logits"

    assert np.array_equal(
        result.predictions.values,
        np.array(
            [
                [1.0, 4.0],
                [5.0, 0.0],
            ]
        ),
    )
    assert result.source_indices == (1, 2)


def test_mean_probability_averages_samples():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.1, 0.9],
            ]
        ),
        output_type="probabilities",
    )

    result = aggregate_object_predictions(
        object_ids=(
            "object-a",
            "object-a",
            "object-b",
        ),
        predictions=predictions,
        method="mean_probability",
    )

    assert result.object_ids == (
        "object-a",
        "object-b",
    )

    assert result.predictions.output_type == "probabilities"

    assert np.allclose(
        result.predictions.values,
        np.array(
            [
                [0.6, 0.4],
                [0.1, 0.9],
            ]
        ),
    )
    assert result.source_indices is None


def test_latest_requires_alerts():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        output_type="logits",
    )

    with pytest.raises(
        ValueError,
        match="Alert numbers",
    ):
        aggregate_object_predictions(
            object_ids=(
                "object-a",
                "object-b",
            ),
            predictions=predictions,
            method="latest",
        )


def test_latest_rejects_duplicate_latest_alert():
    predictions = ClassificationPredictions(
        values=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        output_type="logits",
    )

    with pytest.raises(
        ValueError,
        match="multiple samples",
    ):
        aggregate_object_predictions(
            object_ids=(
                "object-a",
                "object-a",
            ),
            alerts=(
                5,
                5,
            ),
            predictions=predictions,
            method="latest",
        )
