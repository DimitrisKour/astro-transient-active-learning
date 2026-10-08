"""Tests for adaptive-learning evaluation metrics."""

import numpy as np
import pytest

from al_pipeline.core.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    expected_calibration_error,
    multiclass_brier_score,
)
from al_pipeline.core.types import ClassificationPredictions


def test_perfect_classification_metrics() -> None:
    """Perfect predictions should produce ideal metric values."""
    y_true = np.array([0, 1, 2, 0, 1, 2])

    probabilities = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    predictions = ClassificationPredictions(
        values=probabilities,
        output_type="probabilities",
    )

    results = compute_classification_metrics(
        y_true,
        predictions,
    )

    assert results["accuracy"] == pytest.approx(1.0)
    assert results["balanced_accuracy"] == pytest.approx(1.0)

    assert results["precision_macro"] == pytest.approx(1.0)
    assert results["precision_weighted"] == pytest.approx(1.0)

    assert results["recall_macro"] == pytest.approx(1.0)
    assert results["recall_weighted"] == pytest.approx(1.0)

    assert results["f1_macro"] == pytest.approx(1.0)
    assert results["f1_weighted"] == pytest.approx(1.0)

    assert results["mcc"] == pytest.approx(1.0)

    assert results["roc_auc_macro"] == pytest.approx(1.0)
    assert results["average_precision_macro"] == pytest.approx(1.0)

    assert results["log_loss"] == pytest.approx(0.0, abs=1e-12)
    assert results["brier"] == pytest.approx(0.0)
    assert results["ece"] == pytest.approx(0.0)


def test_requested_metric_subset() -> None:
    """Only explicitly requested metrics should be returned."""
    y_true = np.array([0, 1, 0, 1])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.4, 0.6],
            ]
        ),
        output_type="probabilities",
    )

    results = compute_classification_metrics(
        y_true,
        predictions,
        metrics=("accuracy", "f1_macro"),
    )

    assert set(results) == {
        "accuracy",
        "f1_macro",
    }


def test_logits_and_probabilities_give_same_classification_metrics() -> None:
    """Equivalent logits and probabilities should evaluate identically."""
    y_true = np.array([0, 1, 2, 1])

    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.15, 0.20, 0.65],
            [0.20, 0.60, 0.20],
        ]
    )

    logits = np.log(probabilities)

    probability_predictions = ClassificationPredictions(
        values=probabilities,
        output_type="probabilities",
    )

    logit_predictions = ClassificationPredictions(
        values=logits,
        output_type="logits",
    )

    probability_results = compute_classification_metrics(
        y_true,
        probability_predictions,
    )

    logit_results = compute_classification_metrics(
        y_true,
        logit_predictions,
    )

    assert probability_results.keys() == logit_results.keys()

    for metric in probability_results:
        assert logit_results[metric] == pytest.approx(probability_results[metric])


def test_custom_labels_follow_prediction_column_order() -> None:
    """Evaluation should support labels other than integer class indices."""
    y_true = np.array(["SN_I", "SN_II", "AGN", "SN_I"])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.7, 0.2, 0.1],
            ]
        ),
        output_type="probabilities",
    )

    results = compute_classification_metrics(
        y_true,
        predictions,
        labels=("SN_I", "SN_II", "AGN"),
        metrics=("accuracy", "f1_macro"),
    )

    assert results["accuracy"] == pytest.approx(1.0)
    assert results["f1_macro"] == pytest.approx(1.0)


def test_per_class_metrics() -> None:
    """Per-class results should contain precision, recall, F1, and support."""
    y_true = np.array([0, 0, 1, 1, 2])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.2, 0.7, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
        output_type="probabilities",
    )

    results = compute_per_class_metrics(
        y_true,
        predictions,
    )

    assert set(results) == {0, 1, 2}

    assert results[0]["support"] == 2
    assert results[1]["support"] == 2
    assert results[2]["support"] == 1

    for class_metrics in results.values():
        assert class_metrics["precision"] == pytest.approx(1.0)
        assert class_metrics["recall"] == pytest.approx(1.0)
        assert class_metrics["f1"] == pytest.approx(1.0)


def test_confusion_matrix() -> None:
    """Confusion matrix should follow the prediction-column class order."""
    y_true = np.array([0, 0, 1, 1])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.6, 0.4],
            ]
        ),
        output_type="probabilities",
    )

    matrix = compute_confusion_matrix(
        y_true,
        predictions,
    )

    expected = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    np.testing.assert_array_equal(
        matrix,
        expected,
    )


def test_multiclass_brier_score_known_value() -> None:
    """Brier score should match a directly calculated simple example."""
    y_true = np.array([0, 1])

    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ]
    )

    # Object 1:
    # (0.8 - 1)^2 + (0.2 - 0)^2 = 0.08
    #
    # Object 2:
    # (0.3 - 0)^2 + (0.7 - 1)^2 = 0.18
    #
    # Mean = 0.13
    score = multiclass_brier_score(
        y_true,
        probabilities,
    )

    assert score == pytest.approx(0.13)


def test_expected_calibration_error_perfect_confidence() -> None:
    """Perfectly confident correct predictions should have zero ECE."""
    y_true = np.array([0, 1, 0, 1])

    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = expected_calibration_error(
        y_true,
        probabilities,
        n_bins=10,
    )

    assert result == pytest.approx(0.0)


def test_missing_class_does_not_break_probability_metrics() -> None:
    """A class absent from a split should not make AUC/AP evaluation fail."""
    y_true = np.array([0, 0, 1, 1])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
            ]
        ),
        output_type="probabilities",
    )

    results = compute_classification_metrics(
        y_true,
        predictions,
        metrics=(
            "roc_auc_macro",
            "average_precision_macro",
        ),
    )

    assert np.isfinite(results["roc_auc_macro"])
    assert np.isfinite(results["average_precision_macro"])


def test_unknown_metric_raises() -> None:
    """Unknown metric names should fail explicitly."""
    y_true = np.array([0, 1])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        ),
        output_type="probabilities",
    )

    with pytest.raises(
        ValueError,
        match="Unknown evaluation metric",
    ):
        compute_classification_metrics(
            y_true,
            predictions,
            metrics=("not_a_metric",),
        )


def test_mismatched_number_of_objects_raises() -> None:
    """Targets and predictions must refer to the same objects."""
    y_true = np.array([0, 1, 0])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        ),
        output_type="probabilities",
    )

    with pytest.raises(
        ValueError,
        match="same number of objects",
    ):
        compute_classification_metrics(
            y_true,
            predictions,
        )


def test_unknown_target_label_raises() -> None:
    """Targets must be represented in the declared class labels."""
    y_true = np.array(["SN_I", "UNKNOWN"])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        ),
        output_type="probabilities",
    )

    with pytest.raises(
        ValueError,
        match="not present in labels",
    ):
        compute_classification_metrics(
            y_true,
            predictions,
            labels=("SN_I", "SN_II"),
        )


def test_duplicate_labels_raise() -> None:
    """Prediction-column labels must identify unique classes."""
    y_true = np.array([0, 0])

    predictions = ClassificationPredictions(
        values=np.array(
            [
                [0.8, 0.2],
                [0.7, 0.3],
            ]
        ),
        output_type="probabilities",
    )

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        compute_classification_metrics(
            y_true,
            predictions,
            labels=(0, 0),
        )


def test_invalid_calibration_bin_count_raises() -> None:
    """ECE requires at least one calibration bin."""
    y_true = np.array([0, 1])

    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )

    with pytest.raises(
        ValueError,
        match="n_bins must be at least 1",
    ):
        expected_calibration_error(
            y_true,
            probabilities,
            n_bins=0,
        )
