"""Plotting utilities for adaptive-learning evaluation.

The functions in this module visualize already-computed evaluation
results. They do not perform model inference or calculate classification
metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from al_pipeline.core.evaluation.history import EvaluationHistory


def plot_learning_curve(
    history: EvaluationHistory,
    metric: str,
    *,
    split: str | None = None,
    strategy: str | None = None,
    seed: int | None = None,
    run_id: str | None = None,
    x_axis: str = "n_labeled",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one metric across an adaptive-learning experiment.

    Parameters
    ----------
    history
        Stored evaluation history.
    metric
        Metric to display.
    split, strategy, seed, run_id
        Optional filters selecting one experiment trajectory.
    x_axis
        Either ``"n_labeled"`` or ``"cycle"``.
    ax
        Existing Matplotlib axes. If omitted, a new figure is created.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the plot.
    """
    if x_axis not in {"n_labeled", "cycle"}:
        raise ValueError("x_axis must be either 'n_labeled' or 'cycle'.")

    values = history.get_metric(
        metric,
        split=split,
        strategy=strategy,
        seed=seed,
        run_id=run_id,
    )

    if not values:
        raise ValueError(f"No history records contain metric {metric!r} for the requested filters.")

    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    cycles = np.asarray(
        [cycle for cycle, _, _ in values],
        dtype=int,
    )

    n_labeled = np.asarray(
        [count for _, count, _ in values],
        dtype=int,
    )

    metric_values = np.asarray(
        [value for _, _, value in values],
        dtype=float,
    )

    x_values = n_labeled if x_axis == "n_labeled" else cycles

    ax.plot(
        x_values,
        metric_values,
        marker="o",
    )

    ax.set_xlabel("Number of labeled objects" if x_axis == "n_labeled" else "Active-learning cycle")
    ax.set_ylabel(metric)

    return figure, ax


def plot_strategy_comparison(
    history: EvaluationHistory,
    metric: str,
    strategies: Sequence[str],
    *,
    split: str | None = None,
    seed: int | None = None,
    run_id: str | None = None,
    x_axis: str = "n_labeled",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Compare acquisition strategies on the same metric."""
    if not strategies:
        raise ValueError("strategies must contain at least one strategy.")

    if x_axis not in {"n_labeled", "cycle"}:
        raise ValueError("x_axis must be either 'n_labeled' or 'cycle'.")

    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    plotted = False

    for strategy in strategies:
        values = history.get_metric(
            metric,
            split=split,
            strategy=strategy,
            seed=seed,
            run_id=run_id,
        )

        if not values:
            continue

        cycles = np.asarray(
            [cycle for cycle, _, _ in values],
            dtype=int,
        )

        n_labeled = np.asarray(
            [count for _, count, _ in values],
            dtype=int,
        )

        metric_values = np.asarray(
            [value for _, _, value in values],
            dtype=float,
        )

        x_values = n_labeled if x_axis == "n_labeled" else cycles

        ax.plot(
            x_values,
            metric_values,
            marker="o",
            label=strategy,
        )

        plotted = True

    if not plotted:
        raise ValueError(
            f"No history records contain metric {metric!r} "
            "for the requested strategies and filters."
        )

    ax.set_xlabel("Number of labeled objects" if x_axis == "n_labeled" else "Active-learning cycle")
    ax.set_ylabel(metric)

    ax.legend(
        frameon=False,
    )

    return figure, ax


def plot_confusion_matrix(
    matrix: np.ndarray,
    *,
    labels: Sequence[object] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot an already-computed confusion matrix."""
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion matrix must be square.")

    n_classes = matrix.shape[0]

    if labels is None:
        resolved_labels = tuple(range(n_classes))
    else:
        resolved_labels = tuple(labels)

        if len(resolved_labels) != n_classes:
            raise ValueError("labels must contain one entry for each class.")

    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    image = ax.imshow(matrix)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(
        np.arange(n_classes),
        labels=resolved_labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(n_classes),
        labels=resolved_labels,
    )

    figure.colorbar(
        image,
        ax=ax,
    )

    return figure, ax


def plot_per_class_metrics(
    per_class: Mapping[
        object,
        Mapping[str, float | int],
    ],
    *,
    metrics: Sequence[str] = (
        "precision",
        "recall",
        "f1",
    ),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot selected metrics for each class."""
    if not per_class:
        raise ValueError("per_class must contain at least one class.")

    if not metrics:
        raise ValueError("metrics must contain at least one metric.")

    labels = tuple(per_class)

    for label in labels:
        missing = [metric for metric in metrics if metric not in per_class[label]]

        if missing:
            raise ValueError(f"Class {label!r} is missing metric(s): {missing}.")

    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    x_positions = np.arange(
        len(labels),
        dtype=float,
    )

    n_metrics = len(metrics)
    total_width = 0.8
    bar_width = total_width / n_metrics

    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * bar_width

    for metric_index, metric in enumerate(metrics):
        values = [float(per_class[label][metric]) for label in labels]

        ax.bar(
            x_positions + offsets[metric_index],
            values,
            width=bar_width,
            label=metric,
        )

    ax.set_xticks(
        x_positions,
        labels=labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Score")

    ax.legend(
        frameon=False,
    )

    return figure, ax


# TODO: Add ROC, precision-recall, and calibration plots once metrics.py
# provides structured diagnostic helpers such as compute_roc_curves(),
# compute_precision_recall_curves(), and compute_calibration_curve().
# Plotting should consume those precomputed outputs rather than duplicate
# target encoding, probability handling, or diagnostic calculations here.
