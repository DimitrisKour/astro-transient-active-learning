"""Common classification metrics for adaptive-learning experiments.

This module evaluates model predictions independently of the acquisition
strategy, active-learning loop, and model implementation. Standard metrics
are delegated to scikit-learn; only small calibration helpers that are not
provided in the required form are implemented locally.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from al_pipeline.core.types import ClassificationPredictions


DEFAULT_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "log_loss",
    "roc_auc_macro",
    "average_precision_macro",
    "brier",
    "ece",
)

SUPPORTED_METRICS = frozenset(DEFAULT_METRICS)


def _resolve_labels(
    labels: Sequence[object] | None,
    *,
    n_classes: int,
) -> tuple[object, ...]:
    """Return one unique label for each prediction column."""
    if labels is None:
        return tuple(range(n_classes))

    resolved = tuple(labels)

    if len(resolved) != n_classes:
        raise ValueError("labels must contain one entry for each prediction column.")

    if len(set(resolved)) != len(resolved):
        raise ValueError("labels must not contain duplicates.")

    return resolved


def _encode_targets(
    y_true: np.ndarray,
    labels: Sequence[object],
) -> np.ndarray:
    """Encode target labels using the prediction-column order."""
    label_to_index = {label: index for index, label in enumerate(labels)}

    try:
        return np.asarray(
            [label_to_index[label] for label in y_true],
            dtype=np.int64,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("y_true contains a label that is not present in labels.") from exc


def _macro_ovr_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Return macro one-vs-rest ROC-AUC over defined classes."""
    class_scores: list[float] = []

    for class_index in range(probabilities.shape[1]):
        binary_targets = y_true == class_index

        if np.all(binary_targets) or not np.any(binary_targets):
            continue

        class_scores.append(
            float(
                roc_auc_score(
                    binary_targets,
                    probabilities[:, class_index],
                )
            )
        )

    if not class_scores:
        return float("nan")

    return float(np.mean(class_scores))


def _macro_average_precision(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Return macro one-vs-rest average precision over defined classes."""
    class_scores: list[float] = []

    for class_index in range(probabilities.shape[1]):
        binary_targets = y_true == class_index

        if np.all(binary_targets) or not np.any(binary_targets):
            continue

        class_scores.append(
            float(
                average_precision_score(
                    binary_targets,
                    probabilities[:, class_index],
                )
            )
        )

    if not class_scores:
        return float("nan")

    return float(np.mean(class_scores))


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Return the multiclass Brier score.

    The score is the mean squared distance between the predicted
    probability vector and the one-hot encoded true class vector.
    Lower values are better.
    """
    targets = np.zeros_like(probabilities, dtype=float)
    targets[np.arange(len(y_true)), y_true] = 1.0

    return float(
        np.mean(
            np.sum(
                (probabilities - targets) ** 2,
                axis=1,
            )
        )
    )


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Return top-label expected calibration error.

    Samples are binned by maximum predicted probability. The absolute
    difference between mean confidence and empirical accuracy is weighted
    by the fraction of samples in each bin. Lower values are better.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predicted == y_true

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.minimum(
        np.digitize(
            confidence,
            bin_edges[1:-1],
            right=True,
        ),
        n_bins - 1,
    )

    ece = 0.0

    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index

        if not np.any(in_bin):
            continue

        weight = np.mean(in_bin)
        accuracy = np.mean(correct[in_bin])
        mean_confidence = np.mean(confidence[in_bin])

        ece += weight * abs(accuracy - mean_confidence)

    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray,
    predictions: ClassificationPredictions,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    labels: Sequence[object] | None = None,
    calibration_bins: int = 15,
) -> dict[str, float]:
    """Compute requested scalar classification metrics.

    Parameters
    ----------
    y_true
        One ground-truth class label per object.
    predictions
        Classifier logits or probabilities. Conversion and validation are
        handled by ``ClassificationPredictions.as_probabilities``.
    metrics
        Names of scalar metrics to calculate.
    labels
        Labels corresponding, in order, to the prediction columns. If
        omitted, ``0, ..., n_classes - 1`` is assumed.
    calibration_bins
        Number of confidence bins used for expected calibration error.

    Returns
    -------
    dict[str, float]
        Mapping from metric name to scalar value.
    """
    y_true = np.asarray(y_true)

    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    if len(y_true) == 0:
        raise ValueError("y_true must contain at least one object.")

    probabilities = predictions.as_probabilities()

    if len(y_true) != probabilities.shape[0]:
        raise ValueError("y_true and predictions must contain the same number of objects.")

    resolved_labels = _resolve_labels(
        labels,
        n_classes=probabilities.shape[1],
    )

    encoded_true = _encode_targets(
        y_true,
        resolved_labels,
    )

    encoded_pred = np.argmax(
        probabilities,
        axis=1,
    )

    class_indices = np.arange(
        probabilities.shape[1],
    )

    requested = tuple(metrics)

    unknown = set(requested) - SUPPORTED_METRICS

    if unknown:
        raise ValueError(f"Unknown evaluation metric(s): {sorted(unknown)}.")

    results: dict[str, float] = {}

    for metric in requested:
        if metric == "accuracy":
            value = accuracy_score(
                encoded_true,
                encoded_pred,
            )

        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(
                encoded_true,
                encoded_pred,
            )

        elif metric == "precision_macro":
            value = precision_score(
                encoded_true,
                encoded_pred,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )

        elif metric == "precision_weighted":
            value = precision_score(
                encoded_true,
                encoded_pred,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )

        elif metric == "recall_macro":
            value = recall_score(
                encoded_true,
                encoded_pred,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )

        elif metric == "recall_weighted":
            value = recall_score(
                encoded_true,
                encoded_pred,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )

        elif metric == "f1_macro":
            value = f1_score(
                encoded_true,
                encoded_pred,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )

        elif metric == "f1_weighted":
            value = f1_score(
                encoded_true,
                encoded_pred,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )

        elif metric == "mcc":
            value = matthews_corrcoef(
                encoded_true,
                encoded_pred,
            )

        elif metric == "log_loss":
            value = log_loss(
                encoded_true,
                probabilities,
                labels=class_indices,
            )

        elif metric == "roc_auc_macro":
            value = _macro_ovr_auc(
                encoded_true,
                probabilities,
            )

        elif metric == "average_precision_macro":
            value = _macro_average_precision(
                encoded_true,
                probabilities,
            )

        elif metric == "brier":
            value = multiclass_brier_score(
                encoded_true,
                probabilities,
            )

        elif metric == "ece":
            value = expected_calibration_error(
                encoded_true,
                probabilities,
                n_bins=calibration_bins,
            )

        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled metric: {metric!r}")

        results[metric] = float(value)

    return results


def compute_per_class_metrics(
    y_true: np.ndarray,
    predictions: ClassificationPredictions,
    *,
    labels: Sequence[object] | None = None,
) -> dict[object, dict[str, float | int]]:
    """Return precision, recall, F1, and support for every class."""
    y_true = np.asarray(y_true)

    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    probabilities = predictions.as_probabilities()

    if len(y_true) != probabilities.shape[0]:
        raise ValueError("y_true and predictions must contain the same number of objects.")

    resolved_labels = _resolve_labels(
        labels,
        n_classes=probabilities.shape[1],
    )

    encoded_true = _encode_targets(
        y_true,
        resolved_labels,
    )

    encoded_pred = np.argmax(
        probabilities,
        axis=1,
    )

    class_indices = np.arange(
        probabilities.shape[1],
    )

    precision = precision_score(
        encoded_true,
        encoded_pred,
        labels=class_indices,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        encoded_true,
        encoded_pred,
        labels=class_indices,
        average=None,
        zero_division=0,
    )

    f1 = f1_score(
        encoded_true,
        encoded_pred,
        labels=class_indices,
        average=None,
        zero_division=0,
    )

    support = np.bincount(
        encoded_true,
        minlength=probabilities.shape[1],
    )

    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(resolved_labels)
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    predictions: ClassificationPredictions,
    *,
    labels: Sequence[object] | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """Return a confusion matrix using prediction-column class order."""
    y_true = np.asarray(y_true)

    probabilities = predictions.as_probabilities()

    if len(y_true) != probabilities.shape[0]:
        raise ValueError("y_true and predictions must contain the same number of objects.")

    resolved_labels = _resolve_labels(
        labels,
        n_classes=probabilities.shape[1],
    )

    encoded_true = _encode_targets(
        y_true,
        resolved_labels,
    )

    encoded_pred = np.argmax(
        probabilities,
        axis=1,
    )

    class_indices = np.arange(
        probabilities.shape[1],
    )

    if normalize not in (
        None,
        "true",
        "pred",
        "all",
    ):
        raise ValueError("normalize must be None, 'true', 'pred', or 'all'.")

    return confusion_matrix(
        encoded_true,
        encoded_pred,
        labels=class_indices,
        normalize=normalize,
    )
