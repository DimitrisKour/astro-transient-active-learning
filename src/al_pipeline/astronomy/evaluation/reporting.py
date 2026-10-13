"""Multi-seed reporting for simulated active-learning experiments.

The experiment runner deliberately writes simple JSON and CSV artifacts.
This module turns those artifacts into validated, tidy records and aggregated
statistics without depending on a particular model or acquisition strategy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from al_pipeline.core.evaluation.history import EvaluationHistory


DEFAULT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc_macro",
    "average_precision_macro",
    "log_loss",
    "ece",
)

LOWER_IS_BETTER = {
    "brier",
    "ece",
    "log_loss",
}

BOUNDED_METRICS = {
    "accuracy",
    "average_precision_macro",
    "balanced_accuracy",
    "brier",
    "ece",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "roc_auc_macro",
}

STRATEGY_COLORS = {
    "random": "#171717",
    "margin": "#4C78A8",
    "entropy": "#E69F00",
    "fisher": "#7A5195",
    "entropy_diversity": "#7A5195",
    "entropy_balanced_explore": "#6B6ECF",
}

SUITE_LINESTYLES = (
    "-",
    "--",
    ":",
    "-.",
)

CLASS_COLORS = {
    "AGN": "#4C78A8",
    "TDE": "#7A5195",
    "SN I": "#E69F00",
    "SN II": "#6B6ECF",
    "CV": "#666666",
}


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """Locations and display label for one experiment suite."""

    label: str
    state_dir: Path
    artifact_dir: Path

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Suite label must be non-empty.")

        object.__setattr__(self, "state_dir", Path(self.state_dir))
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Validated locations and metadata for one strategy/seed run."""

    suite: str
    strategy: str
    seed: int
    initialization: str
    state_dir: Path
    artifact_dir: Path
    experiment: Mapping[str, object]
    last_completed_cycle: int


@dataclass(frozen=True, slots=True)
class ReportData:
    """All tidy records needed to audit and plot one or more suites."""

    runs: tuple[RunArtifacts, ...]
    evaluation_records: tuple[dict[str, object], ...]
    evaluation_aggregates: tuple[dict[str, object], ...]
    trajectory_records: tuple[dict[str, object], ...]
    trajectory_aggregates: tuple[dict[str, object], ...]
    per_class_records: tuple[dict[str, object], ...]
    per_class_aggregates: tuple[dict[str, object], ...]
    acquisition_records: tuple[dict[str, object], ...]
    acquisition_aggregates: tuple[dict[str, object], ...]
    overlap_records: tuple[dict[str, object], ...]
    overlap_aggregates: tuple[dict[str, object], ...]
    training_records: tuple[dict[str, object], ...]
    training_aggregates: tuple[dict[str, object], ...]
    validation: Mapping[str, object]


def build_multiseed_report(
    suites: Sequence[SuiteSpec],
    *,
    expected_strategies: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
    expected_cycles: int | None = None,
    require_complete: bool = True,
) -> ReportData:
    """Load, validate, and aggregate one or more experiment suites."""
    if not suites:
        raise ValueError("At least one suite must be provided.")

    runs: list[RunArtifacts] = []
    errors: list[str] = []
    warnings: list[str] = []

    for suite in suites:
        discovered, suite_errors, suite_warnings = discover_suite_runs(
            suite,
            expected_strategies=expected_strategies,
            expected_seeds=expected_seeds,
            expected_cycles=expected_cycles,
            require_complete=require_complete,
        )
        runs.extend(discovered)
        errors.extend(suite_errors)
        warnings.extend(suite_warnings)

    errors.extend(_validate_cross_suite_compatibility(runs))

    if errors:
        raise ValueError("Multi-seed report validation failed:\n- " + "\n- ".join(errors))

    evaluation_records = _load_evaluation_records(runs)
    errors.extend(_validate_evaluation_records(evaluation_records))
    if errors:
        raise ValueError("Multi-seed report validation failed:\n- " + "\n- ".join(errors))

    evaluation_aggregates = aggregate_records(
        evaluation_records,
        group_fields=(
            "suite",
            "initialization",
            "strategy",
            "split",
            "cycle",
            "metric",
        ),
        value_field="value",
        passthrough_fields=("n_labeled", "n_acquired_total"),
    )

    trajectory_records = _build_trajectory_records(evaluation_records)
    trajectory_aggregates = _aggregate_trajectory_records(trajectory_records)

    per_class_records = _load_per_class_records(runs)
    per_class_aggregates = aggregate_records(
        per_class_records,
        group_fields=(
            "suite",
            "initialization",
            "strategy",
            "split",
            "level",
            "cycle",
            "class_name",
            "metric",
        ),
        value_field="value",
        passthrough_fields=("support", "n_acquired_total"),
    )

    acquisition_records = _load_acquisition_records(runs)
    acquisition_aggregates = aggregate_records(
        acquisition_records,
        group_fields=(
            "suite",
            "initialization",
            "strategy",
            "cycle",
            "label",
        ),
        value_field="count",
        passthrough_fields=("cumulative_count",),
    )

    overlap_records = _build_overlap_records(runs)
    overlap_aggregates = aggregate_records(
        overlap_records,
        group_fields=(
            "suite",
            "cycle",
            "strategy_a",
            "strategy_b",
        ),
        value_field="jaccard",
        passthrough_fields=("intersection", "union"),
    )

    training_records = _load_training_records(runs)
    training_aggregates = _aggregate_training_records(training_records)

    validation = {
        "passed": True,
        "errors": [],
        "warnings": warnings,
        "suite_count": len(suites),
        "run_count": len(runs),
        "suites": sorted({run.suite for run in runs}),
        "strategies": sorted({run.strategy for run in runs}),
        "seeds": sorted({run.seed for run in runs}),
        "initializations": sorted({run.initialization for run in runs}),
    }

    return ReportData(
        runs=tuple(runs),
        evaluation_records=tuple(evaluation_records),
        evaluation_aggregates=tuple(evaluation_aggregates),
        trajectory_records=tuple(trajectory_records),
        trajectory_aggregates=tuple(trajectory_aggregates),
        per_class_records=tuple(per_class_records),
        per_class_aggregates=tuple(per_class_aggregates),
        acquisition_records=tuple(acquisition_records),
        acquisition_aggregates=tuple(acquisition_aggregates),
        overlap_records=tuple(overlap_records),
        overlap_aggregates=tuple(overlap_aggregates),
        training_records=tuple(training_records),
        training_aggregates=tuple(training_aggregates),
        validation=validation,
    )


def discover_suite_runs(
    suite: SuiteSpec,
    *,
    expected_strategies: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
    expected_cycles: int | None = None,
    require_complete: bool = True,
) -> tuple[list[RunArtifacts], list[str], list[str]]:
    """Discover strategy/seed runs and report validation problems."""
    errors: list[str] = []
    warnings: list[str] = []
    runs: list[RunArtifacts] = []

    if not suite.state_dir.is_dir():
        return [], [f"Missing state suite directory: {suite.state_dir}"], []

    if not suite.artifact_dir.is_dir():
        return [], [f"Missing artifact suite directory: {suite.artifact_dir}"], []

    strategy_dirs = sorted(
        path for path in suite.state_dir.iterdir() if path.is_dir() and path.name != "shared"
    )

    discovered_strategies = {path.name for path in strategy_dirs}
    if expected_strategies is not None:
        missing = set(expected_strategies).difference(discovered_strategies)
        if missing:
            errors.append(f"Suite {suite.label!r} is missing strategies: {sorted(missing)}")

    for strategy_dir in strategy_dirs:
        strategy = strategy_dir.name
        if expected_strategies is not None and strategy not in expected_strategies:
            continue

        seed_dirs = sorted(path for path in strategy_dir.glob("seed_*") if path.is_dir())

        for state_dir in seed_dirs:
            try:
                seed = int(state_dir.name.removeprefix("seed_"))
            except ValueError:
                warnings.append(f"Ignoring malformed seed directory: {state_dir}")
                continue

            if expected_seeds is not None and seed not in expected_seeds:
                continue

            experiment_path = state_dir / "experiment.json"
            run_state_path = state_dir / "run_state.json"
            history_path = state_dir / "evaluation_history.json"

            for required in (
                experiment_path,
                run_state_path,
                history_path,
            ):
                if not required.is_file():
                    errors.append(f"Missing run artifact: {required}")

            if not all(
                path.is_file()
                for path in (
                    experiment_path,
                    run_state_path,
                    history_path,
                )
            ):
                continue

            experiment = _read_json(experiment_path)
            run_state = _read_json(run_state_path)
            config = experiment.get("config")
            if not isinstance(config, dict):
                errors.append(f"experiment.json has no config mapping: {experiment_path}")
                continue

            if config.get("strategy") != strategy:
                errors.append(
                    f"Strategy mismatch in {experiment_path}: "
                    f"directory={strategy!r}, config={config.get('strategy')!r}"
                )

            if config.get("seed") != seed:
                errors.append(
                    f"Seed mismatch in {experiment_path}: "
                    f"directory={seed}, config={config.get('seed')!r}"
                )

            initialization = str(config.get("training_initialization", "unknown"))
            configured_cycles = int(config.get("max_cycles", -1))
            last_completed = int(run_state.get("last_completed_cycle", -1))

            if expected_cycles is not None and configured_cycles != expected_cycles:
                errors.append(
                    f"Run {suite.label}/{strategy}/seed_{seed} has "
                    f"max_cycles={configured_cycles}, expected "
                    f"{expected_cycles}."
                )

            target_cycles = expected_cycles if expected_cycles is not None else configured_cycles
            if require_complete and last_completed != target_cycles:
                errors.append(
                    f"Incomplete run {suite.label}/{strategy}/seed_{seed}: "
                    f"last_completed_cycle={last_completed}, expected "
                    f"{target_cycles}."
                )

            artifact_dir = suite.artifact_dir / strategy / f"seed_{seed}"
            if not artifact_dir.is_dir():
                errors.append(f"Missing heavy-artifact run directory: {artifact_dir}")
                continue

            runs.append(
                RunArtifacts(
                    suite=suite.label,
                    strategy=strategy,
                    seed=seed,
                    initialization=initialization,
                    state_dir=state_dir,
                    artifact_dir=artifact_dir,
                    experiment=experiment,
                    last_completed_cycle=last_completed,
                )
            )

    if expected_strategies is not None and expected_seeds is not None:
        found = {(run.strategy, run.seed) for run in runs}
        requested = {
            (strategy, seed) for strategy in expected_strategies for seed in expected_seeds
        }
        missing = sorted(requested.difference(found))
        if missing:
            errors.append(f"Suite {suite.label!r} is missing strategy/seed runs: {missing}")

    return runs, errors, warnings


def aggregate_records(
    records: Sequence[Mapping[str, object]],
    *,
    group_fields: Sequence[str],
    value_field: str,
    passthrough_fields: Sequence[str],
    value_prefix: str = "value",
) -> list[dict[str, object]]:
    """Aggregate scalar records and retain transparent seed statistics."""
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        key = tuple(record[field] for field in group_fields)
        grouped[key].append(record)

    aggregates: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        values = [float(record[value_field]) for record in group]
        result = dict(zip(group_fields, key, strict=True))
        result.update(_summary(values, prefix=value_prefix))
        result["seeds"] = "|".join(
            str(value)
            for value in sorted({int(record["seed"]) for record in group if "seed" in record})
        )

        for field in passthrough_fields:
            field_values = [float(record[field]) for record in group]
            result[f"{field}_mean"] = statistics.fmean(field_values)
            result[f"{field}_min"] = min(field_values)
            result[f"{field}_max"] = max(field_values)

        aggregates.append(result)

    return aggregates


def write_report_tables(
    report: ReportData,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Write all tidy and aggregated report tables."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    tables = {
        "evaluation_records.csv": report.evaluation_records,
        "evaluation_aggregates.csv": report.evaluation_aggregates,
        "trajectory_records.csv": report.trajectory_records,
        "trajectory_aggregates.csv": report.trajectory_aggregates,
        "per_class_records.csv": report.per_class_records,
        "per_class_aggregates.csv": report.per_class_aggregates,
        "acquisition_records.csv": report.acquisition_records,
        "acquisition_aggregates.csv": report.acquisition_aggregates,
        "selection_overlap_records.csv": report.overlap_records,
        "selection_overlap_aggregates.csv": report.overlap_aggregates,
        "training_records.csv": report.training_records,
        "training_aggregates.csv": report.training_aggregates,
    }

    paths: list[Path] = []
    for name, rows in tables.items():
        path = destination / name
        _write_csv(path, rows)
        paths.append(path)

    validation_path = destination / "validation_report.json"
    _write_json_atomic(validation_path, report.validation)
    paths.append(validation_path)

    inventory = [
        {
            "suite": run.suite,
            "initialization": run.initialization,
            "strategy": run.strategy,
            "seed": run.seed,
            "last_completed_cycle": run.last_completed_cycle,
            "state_dir": str(run.state_dir),
            "artifact_dir": str(run.artifact_dir),
        }
        for run in report.runs
    ]
    inventory_path = destination / "run_inventory.csv"
    _write_csv(inventory_path, inventory)
    paths.append(inventory_path)

    return tuple(paths)


def write_report_plots(
    report: ReportData,
    output_dir: str | Path,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    splits: Sequence[str] = (
        "validation",
        "test",
        "new_holdout",
    ),
    formats: Sequence[str] = ("png", "pdf"),
) -> tuple[Path, ...]:
    """Write multi-seed metric, class-recall, and selection plots."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for split in splits:
        available = {str(record["split"]) for record in report.evaluation_aggregates}
        if split not in available:
            continue

        figure = plot_multiseed_metric_grid(
            report.evaluation_aggregates,
            split=split,
            metrics=metrics,
        )
        paths.extend(
            _save_figure(
                figure,
                destination / f"object_metrics_{split}",
                formats,
            )
        )
        plt.close(figure)

        class_figure = plot_multiseed_class_recall(
            report.per_class_aggregates,
            split=split,
        )
        if class_figure is not None:
            paths.extend(
                _save_figure(
                    class_figure,
                    destination / f"object_recall_{split}",
                    formats,
                )
            )
            plt.close(class_figure)

    if report.overlap_aggregates:
        figure = plot_selection_overlap(report.overlap_aggregates)
        paths.extend(
            _save_figure(
                figure,
                destination / "cumulative_selection_overlap",
                formats,
            )
        )
        plt.close(figure)

    if report.acquisition_aggregates:
        figure = plot_acquisition_composition(report.acquisition_aggregates)
        paths.extend(
            _save_figure(
                figure,
                destination / "cumulative_acquisition_composition",
                formats,
            )
        )
        plt.close(figure)

    return tuple(paths)


def plot_multiseed_metric_grid(
    aggregates: Sequence[Mapping[str, object]],
    *,
    split: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Figure:
    """Plot strategy means with one-standard-deviation seed bands."""
    if not metrics:
        raise ValueError("metrics must contain at least one metric.")

    n_columns = min(3, len(metrics))
    n_rows = math.ceil(len(metrics) / n_columns)

    with plt.rc_context(_plot_style()):
        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(4.5 * n_columns, 3.5 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )

        suite_labels = sorted(
            {str(record["suite"]) for record in aggregates if record["split"] == split}
        )
        suite_styles = {
            label: SUITE_LINESTYLES[index % len(SUITE_LINESTYLES)]
            for index, label in enumerate(suite_labels)
        }

        for metric_index, metric in enumerate(metrics):
            ax = axes.flat[metric_index]
            matching = [
                record
                for record in aggregates
                if record["split"] == split and record["metric"] == metric
            ]

            curves = _group_dicts(
                matching,
                fields=("suite", "strategy"),
            )
            for (suite, strategy), curve in sorted(curves.items()):
                ordered = sorted(
                    curve,
                    key=lambda record: int(record["cycle"]),
                )
                x_values = np.asarray(
                    [record["n_acquired_total_mean"] for record in ordered],
                    dtype=float,
                )
                means = np.asarray(
                    [record["value_mean"] for record in ordered],
                    dtype=float,
                )
                standard_deviations = np.asarray(
                    [record["value_std"] for record in ordered],
                    dtype=float,
                )

                color = STRATEGY_COLORS.get(
                    str(strategy),
                    "#555555",
                )
                label = str(strategy) if len(suite_labels) == 1 else f"{suite} · {strategy}"
                ax.plot(
                    x_values,
                    means,
                    color=color,
                    linestyle=suite_styles[str(suite)],
                    marker="o",
                    markersize=4,
                    linewidth=1.7,
                    label=label,
                )

                if any(int(record["value_n"]) > 1 for record in ordered):
                    ax.fill_between(
                        x_values,
                        means - standard_deviations,
                        means + standard_deviations,
                        color=color,
                        alpha=0.14,
                        linewidth=0,
                    )

            _finish_metric_axis(ax, metric)

        for index in range(len(metrics), n_rows * n_columns):
            axes.flat[index].set_visible(False)

        visible_axes = [ax for ax in axes.flat if ax.get_visible()]
        if visible_axes:
            handles, labels = _combined_legend_entries(visible_axes)
            if handles:
                figure.legend(
                    handles,
                    labels,
                    loc="outside lower center",
                    ncols=min(3, len(labels)),
                    frameon=False,
                )

        figure.suptitle(
            f"{_pretty(split)} object-level performance",
        )

    return figure


def plot_multiseed_class_recall(
    aggregates: Sequence[Mapping[str, object]],
    *,
    split: str,
) -> Figure | None:
    """Plot object-level per-class recall across acquisition cycles."""
    matching = [
        record
        for record in aggregates
        if record["split"] == split
        and record["level"] == "object"
        and record["metric"] == "recall"
        and float(record["support_max"]) > 0
    ]
    if not matching:
        return None

    classes = sorted(
        {str(record["class_name"]) for record in matching},
        key=_class_sort_key,
    )
    n_columns = min(3, len(classes))
    n_rows = math.ceil(len(classes) / n_columns)
    suite_labels = sorted({str(record["suite"]) for record in matching})
    suite_styles = {
        label: SUITE_LINESTYLES[index % len(SUITE_LINESTYLES)]
        for index, label in enumerate(suite_labels)
    }

    with plt.rc_context(_plot_style()):
        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(4.5 * n_columns, 3.4 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )

        for class_index, class_name in enumerate(classes):
            ax = axes.flat[class_index]
            curves = _group_dicts(
                [record for record in matching if record["class_name"] == class_name],
                fields=("suite", "strategy"),
            )

            for (suite, strategy), curve in sorted(curves.items()):
                ordered = sorted(
                    curve,
                    key=lambda record: int(record["cycle"]),
                )
                x_values = np.asarray(
                    [record["n_acquired_total_mean"] for record in ordered],
                    dtype=float,
                )
                means = np.asarray(
                    [record["value_mean"] for record in ordered],
                    dtype=float,
                )
                deviations = np.asarray(
                    [record["value_std"] for record in ordered],
                    dtype=float,
                )
                color = STRATEGY_COLORS.get(str(strategy), "#555555")
                label = str(strategy) if len(suite_labels) == 1 else f"{suite} · {strategy}"
                ax.plot(
                    x_values,
                    means,
                    color=color,
                    linestyle=suite_styles[str(suite)],
                    marker="o",
                    markersize=4,
                    linewidth=1.7,
                    label=label,
                )
                if any(int(record["value_n"]) > 1 for record in ordered):
                    ax.fill_between(
                        x_values,
                        means - deviations,
                        means + deviations,
                        color=color,
                        alpha=0.14,
                        linewidth=0,
                    )

            ax.set_title(class_name)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Acquired objects")
            ax.set_ylabel("Recall")
            _minimal_axis(ax)

        for index in range(len(classes), n_rows * n_columns):
            axes.flat[index].set_visible(False)

        visible_axes = [ax for ax in axes.flat if ax.get_visible()]
        handles, labels = _combined_legend_entries(visible_axes)
        figure.legend(
            handles,
            labels,
            loc="outside lower center",
            ncols=min(3, len(labels)),
            frameon=False,
        )
        figure.suptitle(
            f"{_pretty(split)} class recall",
        )

    return figure


def plot_selection_overlap(
    aggregates: Sequence[Mapping[str, object]],
) -> Figure:
    """Plot cumulative Jaccard overlap between acquisition strategies."""
    suites = sorted({str(record["suite"]) for record in aggregates})

    with plt.rc_context(_plot_style()):
        figure, axes = plt.subplots(
            1,
            len(suites),
            figsize=(5.2 * len(suites), 3.8),
            squeeze=False,
            constrained_layout=True,
        )
        for index, suite in enumerate(suites):
            ax = axes.flat[index]
            curves = _group_dicts(
                [record for record in aggregates if record["suite"] == suite],
                fields=("strategy_a", "strategy_b"),
            )
            for pair_index, ((left, right), curve) in enumerate(sorted(curves.items())):
                ordered = sorted(
                    curve,
                    key=lambda record: int(record["cycle"]),
                )
                x_values = np.asarray(
                    [record["cycle"] for record in ordered],
                    dtype=float,
                )
                means = np.asarray(
                    [record["value_mean"] for record in ordered],
                    dtype=float,
                )
                deviations = np.asarray(
                    [record["value_std"] for record in ordered],
                    dtype=float,
                )
                color = ("#171717", "#4C78A8", "#E69F00", "#7A5195")[pair_index % 4]
                ax.plot(
                    x_values,
                    means,
                    marker="o",
                    color=color,
                    label=f"{left} – {right}",
                )
                if any(int(record["value_n"]) > 1 for record in ordered):
                    ax.fill_between(
                        x_values,
                        means - deviations,
                        means + deviations,
                        color=color,
                        alpha=0.14,
                        linewidth=0,
                    )

            ax.set_title(suite)
            ax.set_xlabel("Active-learning cycle")
            ax.set_ylabel("Cumulative Jaccard overlap")
            ax.set_ylim(0.0, 1.0)
            ax.legend(frameon=False)
            _minimal_axis(ax)

    return figure


def plot_acquisition_composition(
    aggregates: Sequence[Mapping[str, object]],
) -> Figure:
    """Plot mean cumulative acquired-label composition."""
    panels = sorted({(str(record["suite"]), str(record["strategy"])) for record in aggregates})
    n_columns = min(3, len(panels))
    n_rows = math.ceil(len(panels) / n_columns)

    with plt.rc_context(_plot_style()):
        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(4.6 * n_columns, 3.6 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        for panel_index, (suite, strategy) in enumerate(panels):
            ax = axes.flat[panel_index]
            matching = [
                record
                for record in aggregates
                if record["suite"] == suite and record["strategy"] == strategy
            ]
            labels = sorted(
                {
                    str(record["label"])
                    for record in matching
                    if float(record["cumulative_count_max"]) > 0
                },
                key=_class_sort_key,
            )
            for label in labels:
                ordered = sorted(
                    [record for record in matching if record["label"] == label],
                    key=lambda record: int(record["cycle"]),
                )
                ax.plot(
                    [record["cycle"] for record in ordered],
                    [record["cumulative_count_mean"] for record in ordered],
                    marker="o",
                    label=label,
                    color=CLASS_COLORS.get(label),
                )

            ax.set_title(f"{suite} · {strategy}")
            ax.set_xlabel("Active-learning cycle")
            ax.set_ylabel("Cumulative acquired objects")
            ax.legend(frameon=False)
            _minimal_axis(ax)

        for index in range(len(panels), n_rows * n_columns):
            axes.flat[index].set_visible(False)

    return figure


def _load_evaluation_records(
    runs: Sequence[RunArtifacts],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run in runs:
        history = EvaluationHistory.load(run.state_dir / "evaluation_history.json")
        seen: set[tuple[int, str]] = set()
        baseline_n_labeled = min(
            record.n_labeled for record in history.records if record.cycle == 0
        )

        for record in history.records:
            point = (record.cycle, record.split)
            if point in seen:
                raise ValueError(
                    f"Duplicate evaluation point in {run.state_dir}: "
                    f"cycle={record.cycle}, split={record.split!r}."
                )
            seen.add(point)

            if record.strategy not in {None, run.strategy}:
                raise ValueError(f"Evaluation strategy mismatch in {run.state_dir}.")
            if record.seed not in {None, run.seed}:
                raise ValueError(f"Evaluation seed mismatch in {run.state_dir}.")

            for metric, value in record.metrics.items():
                records.append(
                    {
                        "suite": run.suite,
                        "initialization": run.initialization,
                        "strategy": run.strategy,
                        "seed": run.seed,
                        "cycle": record.cycle,
                        "split": record.split,
                        "metric": metric,
                        "value": value,
                        "n_labeled": record.n_labeled,
                        "n_acquired": record.n_acquired,
                        "n_acquired_total": (record.n_labeled - baseline_n_labeled),
                    }
                )
    return records


def _build_trajectory_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped = _group_dicts(
        records,
        fields=(
            "suite",
            "initialization",
            "strategy",
            "seed",
            "split",
            "metric",
        ),
    )
    summaries: list[dict[str, object]] = []

    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(group, key=lambda record: int(record["cycle"]))
        if int(ordered[0]["cycle"]) != 0:
            raise ValueError(f"Trajectory has no cycle-zero baseline: {key}")

        values = np.asarray([record["value"] for record in ordered], dtype=float)
        budgets = np.asarray(
            [record["n_acquired_total"] for record in ordered],
            dtype=float,
        )
        lower_is_better = str(key[-1]) in LOWER_IS_BETTER
        best_index = int(np.argmin(values) if lower_is_better else np.argmax(values))
        post_baseline = values[1:] if len(values) > 1 else values
        normalized_area = (
            float(np.trapezoid(values, budgets) / (budgets[-1] - budgets[0]))
            if len(values) > 1 and budgets[-1] > budgets[0]
            else float(values[0])
        )

        result = dict(
            zip(
                (
                    "suite",
                    "initialization",
                    "strategy",
                    "seed",
                    "split",
                    "metric",
                ),
                key,
                strict=True,
            )
        )
        result.update(
            {
                "baseline": float(values[0]),
                "final": float(values[-1]),
                "final_delta": float(values[-1] - values[0]),
                "best": float(values[best_index]),
                "best_cycle": int(ordered[best_index]["cycle"]),
                "best_budget": int(ordered[best_index]["n_acquired_total"]),
                "mean_post_baseline": float(np.mean(post_baseline)),
                "normalized_area_under_curve": normalized_area,
                "final_cycle": int(ordered[-1]["cycle"]),
                "final_budget": int(ordered[-1]["n_acquired_total"]),
            }
        )
        summaries.append(result)

    return summaries


def _load_per_class_records(
    runs: Sequence[RunArtifacts],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    budgets = _budget_lookup(runs)

    for run in runs:
        for cycle in range(run.last_completed_cycle + 1):
            evaluation_dir = run.artifact_dir / f"cycle_{cycle:03d}" / "evaluation"
            for path in sorted(evaluation_dir.glob("*_per_class_metrics.json")):
                split = path.name.removesuffix("_per_class_metrics.json")
                payload = _read_json(path)
                for level in ("object", "sample"):
                    class_payload = payload.get(level, {})
                    if not isinstance(class_payload, dict):
                        continue
                    for class_name, metrics in class_payload.items():
                        if not isinstance(metrics, dict):
                            continue
                        support = int(metrics.get("support", 0))
                        for metric in ("precision", "recall", "f1"):
                            if metric not in metrics:
                                continue
                            records.append(
                                {
                                    "suite": run.suite,
                                    "initialization": run.initialization,
                                    "strategy": run.strategy,
                                    "seed": run.seed,
                                    "cycle": cycle,
                                    "split": split,
                                    "level": level,
                                    "class_name": str(class_name),
                                    "metric": metric,
                                    "value": float(metrics[metric]),
                                    "support": support,
                                    "n_acquired_total": budgets[
                                        (run.suite, run.strategy, run.seed, cycle)
                                    ],
                                }
                            )
    return records


def _load_acquisition_records(
    runs: Sequence[RunArtifacts],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    class_names_by_suite: dict[str, tuple[str, ...]] = {}
    for run in runs:
        summary_path = run.artifact_dir / "cycle_000" / "evaluation" / "evaluation_summary.json"
        summary = _read_json(summary_path)
        class_names = tuple(str(value) for value in summary.get("model_class_names", ()))
        if not class_names:
            raise ValueError(f"No model_class_names found in {summary_path}.")
        existing = class_names_by_suite.get(run.suite)
        if existing is not None and existing != class_names:
            raise ValueError(f"Model class order differs within suite {run.suite!r}.")
        class_names_by_suite[run.suite] = class_names

    for run in runs:
        path = run.state_dir / "acquisitions.csv"
        rows = _read_csv(path)
        per_cycle: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        seen_ids: set[str] = set()
        for row in rows:
            object_id = row["object_id"]
            if object_id in seen_ids:
                raise ValueError(f"Duplicate acquired object {object_id!r} in {path}.")
            seen_ids.add(object_id)
            per_cycle[int(row["cycle"])][row["label"]] += 1

        labels = class_names_by_suite[run.suite]
        cumulative = {label: 0 for label in labels}
        for cycle in range(1, run.last_completed_cycle + 1):
            counts = per_cycle.get(cycle, {})
            for label in labels:
                count = int(counts.get(label, 0))
                cumulative[label] += count
                records.append(
                    {
                        "suite": run.suite,
                        "initialization": run.initialization,
                        "strategy": run.strategy,
                        "seed": run.seed,
                        "cycle": cycle,
                        "label": label,
                        "count": count,
                        "cumulative_count": cumulative[label],
                    }
                )
    return records


def _aggregate_trajectory_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    fields = (
        "baseline",
        "final",
        "final_delta",
        "best",
        "best_cycle",
        "best_budget",
        "mean_post_baseline",
        "normalized_area_under_curve",
        "final_cycle",
        "final_budget",
    )
    grouped = _group_dicts(
        records,
        fields=(
            "suite",
            "initialization",
            "strategy",
            "split",
            "metric",
        ),
    )
    aggregates: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        result = dict(
            zip(
                (
                    "suite",
                    "initialization",
                    "strategy",
                    "split",
                    "metric",
                ),
                key,
                strict=True,
            )
        )
        result["seeds"] = "|".join(
            str(seed) for seed in sorted({int(row["seed"]) for row in group})
        )
        for field in fields:
            result.update(
                _summary(
                    [float(row[field]) for row in group],
                    prefix=field,
                )
            )
        aggregates.append(result)
    return aggregates


def _build_overlap_records(
    runs: Sequence[RunArtifacts],
) -> list[dict[str, object]]:
    by_suite_seed: dict[tuple[str, int], list[RunArtifacts]] = defaultdict(list)
    for run in runs:
        by_suite_seed[(run.suite, run.seed)].append(run)

    records: list[dict[str, object]] = []
    for (suite, seed), grouped_runs in sorted(by_suite_seed.items()):
        acquired: dict[str, dict[int, set[str]]] = {}
        initialization = grouped_runs[0].initialization
        maximum_cycle = min(run.last_completed_cycle for run in grouped_runs)

        for run in grouped_runs:
            rows = _read_csv(run.state_dir / "acquisitions.csv")
            cumulative: set[str] = set()
            acquired[run.strategy] = {}
            for cycle in range(1, maximum_cycle + 1):
                cumulative.update(row["object_id"] for row in rows if int(row["cycle"]) == cycle)
                acquired[run.strategy][cycle] = set(cumulative)

        strategies = sorted(acquired)
        for left_index, left in enumerate(strategies):
            for right in strategies[left_index + 1 :]:
                for cycle in range(1, maximum_cycle + 1):
                    left_ids = acquired[left][cycle]
                    right_ids = acquired[right][cycle]
                    intersection = len(left_ids.intersection(right_ids))
                    union = len(left_ids.union(right_ids))
                    records.append(
                        {
                            "suite": suite,
                            "initialization": initialization,
                            "seed": seed,
                            "cycle": cycle,
                            "strategy_a": left,
                            "strategy_b": right,
                            "intersection": intersection,
                            "union": union,
                            "jaccard": (intersection / union if union else 1.0),
                        }
                    )
    return records


def _load_training_records(
    runs: Sequence[RunArtifacts],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run in runs:
        for cycle in range(1, run.last_completed_cycle + 1):
            path = run.artifact_dir / f"cycle_{cycle:03d}" / "training" / "training_summary.json"
            if not path.is_file():
                raise ValueError(f"Missing training summary: {path}")
            payload = _read_json(path)
            records.append(
                {
                    "suite": run.suite,
                    "initialization": run.initialization,
                    "strategy": run.strategy,
                    "seed": run.seed,
                    "cycle": cycle,
                    "best_epoch": int(payload["best_epoch"]),
                    "epochs_completed": int(payload["epochs_completed"]),
                    "best_validation_loss": float(payload["best_validation_loss"]),
                    "elapsed_seconds": float(payload["elapsed_seconds"]),
                    "peak_cuda_memory_bytes": (
                        ""
                        if payload.get("peak_cuda_memory_bytes") is None
                        else int(payload["peak_cuda_memory_bytes"])
                    ),
                    "training_samples": int(payload["training_samples"]),
                }
            )
    return records


def _aggregate_training_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    fields = (
        "best_epoch",
        "epochs_completed",
        "best_validation_loss",
        "elapsed_seconds",
        "training_samples",
    )
    grouped = _group_dicts(
        records,
        fields=(
            "suite",
            "initialization",
            "strategy",
            "cycle",
        ),
    )
    aggregates: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        result = dict(
            zip(
                ("suite", "initialization", "strategy", "cycle"),
                key,
                strict=True,
            )
        )
        result["seeds"] = "|".join(
            str(seed) for seed in sorted({int(row["seed"]) for row in group})
        )
        for field in fields:
            result.update(
                _summary(
                    [float(row[field]) for row in group],
                    prefix=field,
                )
            )
        aggregates.append(result)
    return aggregates


def _validate_cross_suite_compatibility(
    runs: Sequence[RunArtifacts],
) -> list[str]:
    if not runs:
        return ["No experiment runs were discovered."]

    errors: list[str] = []
    fingerprint_fields = (
        "candidate_manifest_sha256",
        "hidden_labels_sha256",
        "historical_train_manifest_sha256",
        "pool_id",
        "validation_manifest_sha256",
    )
    reference = runs[0]
    reference_holdout = _stable_json_hash(reference.experiment.get("holdout_ids", []))
    reference_config = reference.experiment.get("config", {})
    baseline_by_seed: dict[int, tuple[str, RunArtifacts]] = {}

    for run in runs:
        baseline_hash = str(run.experiment.get("baseline_checkpoint_sha256", ""))
        previous_baseline = baseline_by_seed.setdefault(
            run.seed,
            (baseline_hash, run),
        )
        if previous_baseline[0] != baseline_hash:
            previous_run = previous_baseline[1]
            errors.append(
                f"Run {run.suite}/{run.strategy}/seed_{run.seed} does "
                "not share its seed-specific baseline checkpoint with "
                f"{previous_run.suite}/{previous_run.strategy}."
            )

    for run in runs[1:]:
        for field in fingerprint_fields:
            if run.experiment.get(field) != reference.experiment.get(field):
                errors.append(
                    f"Run {run.suite}/{run.strategy}/seed_{run.seed} does "
                    f"not match reference field {field!r}."
                )
        if _stable_json_hash(run.experiment.get("holdout_ids", [])) != reference_holdout:
            errors.append(
                f"Run {run.suite}/{run.strategy}/seed_{run.seed} uses a different new-data holdout."
            )

        config = run.experiment.get("config", {})
        for field in ("acquisition_size", "max_cycles"):
            if config.get(field) != reference_config.get(field):
                errors.append(
                    f"Run {run.suite}/{run.strategy}/seed_{run.seed} does "
                    f"not match reference config field {field!r}."
                )

    initializations_by_suite = _group_dicts(
        (
            {
                "suite": run.suite,
                "initialization": run.initialization,
            }
            for run in runs
        ),
        fields=("suite",),
    )
    for (suite,), group in initializations_by_suite.items():
        initializations = {str(record["initialization"]) for record in group}
        if len(initializations) != 1:
            errors.append(
                f"Suite {suite!r} mixes training initializations: {sorted(initializations)}."
            )

    return errors


def _validate_evaluation_records(
    records: Sequence[Mapping[str, object]],
) -> list[str]:
    """Require identical evaluation grids and budgets across comparable runs."""
    if not records:
        return ["No evaluation records were loaded."]

    errors: list[str] = []
    by_run = _group_dicts(
        records,
        fields=("suite", "strategy", "seed"),
    )
    reference_key, reference_records = next(iter(sorted(by_run.items())))
    reference_grid = {
        (
            int(record["cycle"]),
            str(record["split"]),
            str(record["metric"]),
        )
        for record in reference_records
    }
    reference_budgets = {
        int(record["cycle"]): int(record["n_acquired_total"]) for record in reference_records
    }

    for key, run_records in sorted(by_run.items()):
        grid = {
            (
                int(record["cycle"]),
                str(record["split"]),
                str(record["metric"]),
            )
            for record in run_records
        }
        if grid != reference_grid:
            missing = sorted(reference_grid.difference(grid))
            extra = sorted(grid.difference(reference_grid))
            errors.append(
                f"Evaluation grid differs for {key}; reference={reference_key}, "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )

        budgets = {int(record["cycle"]): int(record["n_acquired_total"]) for record in run_records}
        if budgets != reference_budgets:
            errors.append(
                f"Acquisition budgets differ for {key}; "
                f"reference={reference_budgets}, observed={budgets}."
            )

    return errors


def _budget_lookup(
    runs: Sequence[RunArtifacts],
) -> dict[tuple[str, str, int, int], int]:
    result: dict[tuple[str, str, int, int], int] = {}
    for record in _load_evaluation_records(runs):
        key = (
            str(record["suite"]),
            str(record["strategy"]),
            int(record["seed"]),
            int(record["cycle"]),
        )
        budget = int(record["n_acquired_total"])
        existing = result.get(key)
        if existing is not None and existing != budget:
            raise ValueError(f"Inconsistent acquisition budget for {key}.")
        result[key] = budget
    return result


def _summary(
    values: Sequence[float],
    *,
    prefix: str,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty value sequence.")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        f"{prefix}_n": len(values),
        f"{prefix}_mean": mean,
        f"{prefix}_std": standard_deviation,
        f"{prefix}_sem": standard_error,
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def _group_dicts(
    records: Iterable[Mapping[str, object]],
    *,
    fields: Sequence[str],
) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in fields)].append(record)
    return grouped


def _finish_metric_axis(ax: Axes, metric: str) -> None:
    ax.set_title(_pretty(metric))
    ax.set_xlabel("Acquired objects")
    ax.set_ylabel(_pretty(metric))
    if metric in BOUNDED_METRICS:
        ax.set_ylim(0.0, 1.0)
    _minimal_axis(ax)


def _minimal_axis(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _combined_legend_entries(
    axes: Sequence[Axes],
) -> tuple[list[object], list[str]]:
    handles: list[object] = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in axes:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=True):
            if label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    return handles, labels


def _plot_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }


def _save_figure(
    figure: Figure,
    path_without_suffix: Path,
    formats: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []
    for format_name in formats:
        normalized = format_name.lower().lstrip(".")
        if normalized not in {"png", "pdf", "svg"}:
            raise ValueError(f"Plot formats must be png, pdf, or svg; received {format_name!r}.")
        path = path_without_suffix.with_suffix(f".{normalized}")
        figure.savefig(
            path,
            dpi=220 if normalized == "png" else None,
            bbox_inches="tight",
        )
        paths.append(path)
    return paths


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing CSV artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON mapping in {path}.")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _stable_json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pretty(value: str) -> str:
    replacements = {
        "ece": "Expected calibration error",
        "log_loss": "Log loss",
        "roc_auc_macro": "Macro ROC AUC",
        "average_precision_macro": "Macro average precision",
        "balanced_accuracy": "Balanced accuracy",
        "new_holdout": "New-data holdout",
    }
    return replacements.get(value, value.replace("_", " ").title())


def _class_sort_key(value: object) -> tuple[int, str]:
    order = {
        "AGN": 0,
        "TDE": 1,
        "SN I": 2,
        "SN II": 3,
        "CV": 4,
    }
    text = str(value)
    return order.get(text, len(order)), text


__all__ = [
    "DEFAULT_METRICS",
    "ReportData",
    "RunArtifacts",
    "SuiteSpec",
    "aggregate_records",
    "build_multiseed_report",
    "discover_suite_runs",
    "plot_acquisition_composition",
    "plot_multiseed_class_recall",
    "plot_multiseed_metric_grid",
    "plot_selection_overlap",
    "write_report_plots",
    "write_report_tables",
]
