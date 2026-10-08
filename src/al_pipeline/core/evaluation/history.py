"""Evaluation history for adaptive-learning experiments.

This module stores model-evaluation results across active-learning cycles.
It is independent of the model, acquisition strategy, and metric
implementation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Evaluation results from one experiment point.

    Parameters
    ----------
    cycle
        Active-learning cycle number.
    n_labeled
        Total number of labeled training objects at evaluation time.
    n_acquired
        Number of objects acquired during the most recent cycle.
    split
        Dataset split being evaluated, for example ``"validation"`` or
        ``"test"``.
    metrics
        Scalar evaluation metrics recorded at this point.
    strategy
        Optional acquisition-strategy identifier.
    seed
        Optional experiment seed.
    run_id
        Optional identifier distinguishing repeated experiments.
    """

    cycle: int
    n_labeled: int
    n_acquired: int
    split: str
    metrics: dict[str, float]
    strategy: str | None = None
    seed: int | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.cycle < 0:
            raise ValueError("cycle must be non-negative.")

        if self.n_labeled < 0:
            raise ValueError("n_labeled must be non-negative.")

        if self.n_acquired < 0:
            raise ValueError("n_acquired must be non-negative.")

        if not isinstance(self.split, str) or not self.split:
            raise ValueError("split must be a non-empty string.")

        if not self.metrics:
            raise ValueError("metrics must contain at least one value.")

        converted_metrics: dict[str, float] = {}

        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings.")

            try:
                converted_metrics[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Metric {name!r} must be numeric.") from exc

        object.__setattr__(
            self,
            "metrics",
            converted_metrics,
        )


class EvaluationHistory:
    """Collection of evaluation records from adaptive-learning experiments."""

    def __init__(
        self,
        records: list[EvaluationRecord] | None = None,
    ) -> None:
        self._records = list(records) if records is not None else []

    @property
    def records(self) -> tuple[EvaluationRecord, ...]:
        """Return stored records as an immutable tuple."""
        return tuple(self._records)

    def __len__(self) -> int:
        """Return the number of stored evaluation records."""
        return len(self._records)

    def add(
        self,
        *,
        cycle: int,
        n_labeled: int,
        n_acquired: int,
        split: str,
        metrics: Mapping[str, float],
        strategy: str | None = None,
        seed: int | None = None,
        run_id: str | None = None,
    ) -> EvaluationRecord:
        """Create and store one evaluation record."""
        record = EvaluationRecord(
            cycle=cycle,
            n_labeled=n_labeled,
            n_acquired=n_acquired,
            split=split,
            metrics=dict(metrics),
            strategy=strategy,
            seed=seed,
            run_id=run_id,
        )

        self._records.append(record)

        return record

    def get_metric(
        self,
        metric: str,
        *,
        split: str | None = None,
        strategy: str | None = None,
        seed: int | None = None,
        run_id: str | None = None,
    ) -> list[tuple[int, int, float]]:
        """Return one metric across matching history records.

        Returns
        -------
        list[tuple[int, int, float]]
            Tuples containing ``(cycle, n_labeled, metric_value)`` in
            insertion order.
        """
        values: list[tuple[int, int, float]] = []

        for record in self._records:
            if split is not None and record.split != split:
                continue

            if strategy is not None and record.strategy != strategy:
                continue

            if seed is not None and record.seed != seed:
                continue

            if run_id is not None and record.run_id != run_id:
                continue

            if metric not in record.metrics:
                continue

            values.append(
                (
                    record.cycle,
                    record.n_labeled,
                    record.metrics[metric],
                )
            )

        return values

    def available_metrics(self) -> tuple[str, ...]:
        """Return all metric names present in the history."""
        names = {metric for record in self._records for metric in record.metrics}

        return tuple(sorted(names))

    def to_dicts(self) -> list[dict[str, object]]:
        """Return records as plain dictionaries."""
        return [asdict(record) for record in self._records]

    def save(self, path: str | Path) -> None:
        """Save history to a JSON file."""
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = destination.with_suffix(destination.suffix + ".tmp")

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dicts(),
                file,
                indent=2,
            )
            file.write("\n")

        temporary.replace(destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> EvaluationHistory:
        """Load evaluation history from a JSON file."""
        source = Path(path)

        with source.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_records = json.load(file)

        if not isinstance(raw_records, list):
            raise ValueError("Evaluation-history file must contain a list of records.")

        records: list[EvaluationRecord] = []

        for item in raw_records:
            if not isinstance(item, dict):
                raise ValueError("Each evaluation-history entry must be a mapping.")

            records.append(EvaluationRecord(**item))

        return cls(records)
