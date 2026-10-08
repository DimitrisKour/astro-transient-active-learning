"""Tests for adaptive-learning evaluation plotting."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from al_pipeline.core.evaluation.history import EvaluationHistory
from al_pipeline.core.evaluation.plotting import (
    plot_confusion_matrix,
    plot_learning_curve,
    plot_per_class_metrics,
    plot_strategy_comparison,
)


def _make_history() -> EvaluationHistory:
    """Return a small history shared by plotting tests."""
    history = EvaluationHistory()

    history.add(
        cycle=0,
        n_labeled=500,
        n_acquired=0,
        split="validation",
        strategy="entropy",
        seed=42,
        metrics={"f1_macro": 0.60},
    )

    history.add(
        cycle=1,
        n_labeled=600,
        n_acquired=100,
        split="validation",
        strategy="entropy",
        seed=42,
        metrics={"f1_macro": 0.70},
    )

    history.add(
        cycle=0,
        n_labeled=500,
        n_acquired=0,
        split="validation",
        strategy="margin",
        seed=42,
        metrics={"f1_macro": 0.55},
    )

    history.add(
        cycle=1,
        n_labeled=600,
        n_acquired=100,
        split="validation",
        strategy="margin",
        seed=42,
        metrics={"f1_macro": 0.65},
    )

    return history


def test_plot_learning_curve_uses_labeled_set_size() -> None:
    """Learning curves should use n_labeled by default."""
    history = _make_history()

    figure, axes = plot_learning_curve(
        history,
        "f1_macro",
        split="validation",
        strategy="entropy",
        seed=42,
    )

    line = axes.lines[0]

    np.testing.assert_array_equal(
        line.get_xdata(),
        np.array([500, 600]),
    )

    np.testing.assert_allclose(
        line.get_ydata(),
        np.array([0.60, 0.70]),
    )

    assert axes.get_xlabel() == "Number of labeled objects"
    assert axes.get_ylabel() == "f1_macro"

    plt.close(figure)


def test_plot_learning_curve_can_use_cycle() -> None:
    """Cycle number should be available as an alternative x-axis."""
    history = _make_history()

    figure, axes = plot_learning_curve(
        history,
        "f1_macro",
        strategy="entropy",
        x_axis="cycle",
    )

    np.testing.assert_array_equal(
        axes.lines[0].get_xdata(),
        np.array([0, 1]),
    )

    assert axes.get_xlabel() == "Active-learning cycle"

    plt.close(figure)


def test_plot_learning_curve_rejects_invalid_x_axis() -> None:
    """Unknown x-axis choices should fail explicitly."""
    history = _make_history()

    with pytest.raises(
        ValueError,
        match="x_axis must be either",
    ):
        plot_learning_curve(
            history,
            "f1_macro",
            x_axis="invalid",
        )


def test_plot_learning_curve_requires_matching_data() -> None:
    """Missing metrics or filters should not produce an empty plot."""
    history = _make_history()

    with pytest.raises(
        ValueError,
        match="No history records contain metric",
    ):
        plot_learning_curve(
            history,
            "accuracy",
        )


def test_plot_strategy_comparison() -> None:
    """Multiple acquisition strategies should produce separate curves."""
    history = _make_history()

    figure, axes = plot_strategy_comparison(
        history,
        "f1_macro",
        strategies=("entropy", "margin"),
        split="validation",
        seed=42,
    )

    assert len(axes.lines) == 2

    labels = {line.get_label() for line in axes.lines}

    assert labels == {
        "entropy",
        "margin",
    }

    plt.close(figure)


def test_plot_strategy_comparison_rejects_empty_strategies() -> None:
    """Strategy comparison requires at least one strategy."""
    history = _make_history()

    with pytest.raises(
        ValueError,
        match="at least one strategy",
    ):
        plot_strategy_comparison(
            history,
            "f1_macro",
            strategies=(),
        )


def test_plot_confusion_matrix() -> None:
    """A square confusion matrix should be displayed with class labels."""
    matrix = np.array(
        [
            [8, 2],
            [1, 9],
        ]
    )

    figure, axes = plot_confusion_matrix(
        matrix,
        labels=("SN_I", "SN_II"),
    )

    assert axes.get_xlabel() == "Predicted class"
    assert axes.get_ylabel() == "True class"

    x_labels = [tick.get_text() for tick in axes.get_xticklabels()]

    y_labels = [tick.get_text() for tick in axes.get_yticklabels()]

    assert x_labels == ["SN_I", "SN_II"]
    assert y_labels == ["SN_I", "SN_II"]

    plt.close(figure)


def test_plot_confusion_matrix_rejects_non_square_matrix() -> None:
    """Confusion matrices must contain one row and column per class."""
    matrix = np.ones((2, 3))

    with pytest.raises(
        ValueError,
        match="must be square",
    ):
        plot_confusion_matrix(matrix)


def test_plot_per_class_metrics() -> None:
    """Per-class precision, recall, and F1 should produce grouped bars."""
    per_class = {
        "SN_I": {
            "precision": 0.90,
            "recall": 0.80,
            "f1": 0.85,
            "support": 100,
        },
        "SN_II": {
            "precision": 0.75,
            "recall": 0.70,
            "f1": 0.72,
            "support": 50,
        },
    }

    figure, axes = plot_per_class_metrics(
        per_class,
    )

    # 2 classes x 3 requested metrics.
    assert len(axes.patches) == 6

    legend_labels = [text.get_text() for text in axes.get_legend().get_texts()]

    assert legend_labels == [
        "precision",
        "recall",
        "f1",
    ]

    plt.close(figure)


def test_plot_per_class_metrics_rejects_missing_metric() -> None:
    """Requested metrics must exist for every supplied class."""
    per_class = {
        "SN_I": {
            "precision": 0.90,
            "recall": 0.80,
        }
    }

    with pytest.raises(
        ValueError,
        match="is missing metric",
    ):
        plot_per_class_metrics(
            per_class,
            metrics=("precision", "f1"),
        )
