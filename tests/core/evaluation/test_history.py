"""Tests for adaptive-learning evaluation history."""

import json

import pytest

from al_pipeline.core.evaluation.history import EvaluationHistory, EvaluationRecord


def test_evaluation_record() -> None:
    """A valid evaluation record should preserve experiment metadata."""
    record = EvaluationRecord(
        cycle=2,
        n_labeled=1000,
        n_acquired=100,
        split="validation",
        metrics={
            "accuracy": 0.9,
            "f1_macro": 0.8,
        },
        strategy="entropy",
        seed=42,
        run_id="run_01",
    )

    assert record.cycle == 2
    assert record.n_labeled == 1000
    assert record.n_acquired == 100
    assert record.split == "validation"

    assert record.metrics["accuracy"] == pytest.approx(0.9)
    assert record.metrics["f1_macro"] == pytest.approx(0.8)

    assert record.strategy == "entropy"
    assert record.seed == 42
    assert record.run_id == "run_01"


def test_history_add() -> None:
    """Adding a result should create and store one evaluation record."""
    history = EvaluationHistory()

    record = history.add(
        cycle=0,
        n_labeled=500,
        n_acquired=0,
        split="validation",
        metrics={"accuracy": 0.75},
    )

    assert len(history) == 1
    assert history.records == (record,)

    assert record.cycle == 0
    assert record.n_labeled == 500
    assert record.metrics["accuracy"] == pytest.approx(0.75)


def test_records_property_is_immutable_view() -> None:
    """History should not expose its internal mutable record list."""
    history = EvaluationHistory()

    history.add(
        cycle=0,
        n_labeled=500,
        n_acquired=0,
        split="validation",
        metrics={"accuracy": 0.75},
    )

    assert isinstance(history.records, tuple)

    with pytest.raises(AttributeError):
        history.records.append("invalid")  # type: ignore[attr-defined]


def test_available_metrics() -> None:
    """Metric discovery should include names appearing in any record."""
    history = EvaluationHistory()

    history.add(
        cycle=0,
        n_labeled=500,
        n_acquired=0,
        split="validation",
        metrics={
            "accuracy": 0.75,
            "f1_macro": 0.70,
        },
    )

    history.add(
        cycle=1,
        n_labeled=600,
        n_acquired=100,
        split="validation",
        metrics={
            "accuracy": 0.80,
            "mcc": 0.65,
        },
    )

    assert history.available_metrics() == (
        "accuracy",
        "f1_macro",
        "mcc",
    )


def test_get_metric() -> None:
    """Metric history should contain cycle, labeled size, and value."""
    history = EvaluationHistory()

    history.add(
        cycle=0,
        n_labeled=500,
        n_acquired=0,
        split="validation",
        metrics={"f1_macro": 0.60},
    )

    history.add(
        cycle=1,
        n_labeled=600,
        n_acquired=100,
        split="validation",
        metrics={"f1_macro": 0.70},
    )

    history.add(
        cycle=2,
        n_labeled=700,
        n_acquired=100,
        split="validation",
        metrics={"f1_macro": 0.80},
    )

    assert history.get_metric("f1_macro") == [
        (0, 500, 0.60),
        (1, 600, 0.70),
        (2, 700, 0.80),
    ]


def test_get_metric_filters_records() -> None:
    """History queries should distinguish split, strategy, seed, and run."""
    history = EvaluationHistory()

    history.add(
        cycle=1,
        n_labeled=600,
        n_acquired=100,
        split="validation",
        strategy="entropy",
        seed=42,
        run_id="run_01",
        metrics={"f1_macro": 0.70},
    )

    history.add(
        cycle=1,
        n_labeled=600,
        n_acquired=100,
        split="validation",
        strategy="margin",
        seed=42,
        run_id="run_02",
        metrics={"f1_macro": 0.65},
    )

    history.add(
        cycle=1,
        n_labeled=600,
        n_acquired=100,
        split="test",
        strategy="entropy",
        seed=43,
        run_id="run_03",
        metrics={"f1_macro": 0.75},
    )

    result = history.get_metric(
        "f1_macro",
        split="validation",
        strategy="entropy",
        seed=42,
        run_id="run_01",
    )

    assert result == [
        (1, 600, 0.70),
    ]


def test_get_metric_skips_records_without_metric() -> None:
    """Records need not all contain exactly the same metric set."""
    history = EvaluationHistory()

    history.add(
        cycle=0,
        n_labeled=500,
        n_acquired=0,
        split="validation",
        metrics={"accuracy": 0.75},
    )

    history.add(
        cycle=1,
        n_labeled=600,
        n_acquired=100,
        split="validation",
        metrics={"f1_macro": 0.70},
    )

    assert history.get_metric("f1_macro") == [
        (1, 600, 0.70),
    ]


def test_save_and_load_round_trip(tmp_path) -> None:
    """Saved history should reproduce the original records exactly."""
    history = EvaluationHistory()

    history.add(
        cycle=0,
        n_labeled=500,
        n_acquired=0,
        split="validation",
        strategy="entropy",
        seed=42,
        run_id="run_01",
        metrics={
            "accuracy": 0.75,
            "f1_macro": 0.70,
        },
    )

    history.add(
        cycle=1,
        n_labeled=600,
        n_acquired=100,
        split="validation",
        strategy="entropy",
        seed=42,
        run_id="run_01",
        metrics={
            "accuracy": 0.80,
            "f1_macro": 0.76,
        },
    )

    path = tmp_path / "history.json"

    history.save(path)

    loaded = EvaluationHistory.load(path)

    assert loaded.records == history.records


def test_save_creates_parent_directories(tmp_path) -> None:
    """Saving should create missing output directories."""
    history = EvaluationHistory()

    history.add(
        cycle=0,
        n_labeled=500,
        n_acquired=0,
        split="validation",
        metrics={"accuracy": 0.75},
    )

    path = tmp_path / "nested" / "evaluation" / "history.json"

    history.save(path)

    assert path.exists()


def test_invalid_record_values_raise() -> None:
    """Negative counts, empty splits, and empty metrics are invalid."""
    with pytest.raises(
        ValueError,
        match="cycle must be non-negative",
    ):
        EvaluationRecord(
            cycle=-1,
            n_labeled=100,
            n_acquired=0,
            split="validation",
            metrics={"accuracy": 0.5},
        )

    with pytest.raises(
        ValueError,
        match="n_labeled must be non-negative",
    ):
        EvaluationRecord(
            cycle=0,
            n_labeled=-1,
            n_acquired=0,
            split="validation",
            metrics={"accuracy": 0.5},
        )

    with pytest.raises(
        ValueError,
        match="n_acquired must be non-negative",
    ):
        EvaluationRecord(
            cycle=0,
            n_labeled=100,
            n_acquired=-1,
            split="validation",
            metrics={"accuracy": 0.5},
        )

    with pytest.raises(
        ValueError,
        match="split must be a non-empty string",
    ):
        EvaluationRecord(
            cycle=0,
            n_labeled=100,
            n_acquired=0,
            split="",
            metrics={"accuracy": 0.5},
        )

    with pytest.raises(
        ValueError,
        match="split must be a non-empty string",
    ):
        EvaluationRecord(
            cycle=0,
            n_labeled=100,
            n_acquired=0,
            split=123,  # type: ignore[arg-type]
            metrics={"accuracy": 0.5},
        )

    with pytest.raises(
        ValueError,
        match="metrics must contain at least one value",
    ):
        EvaluationRecord(
            cycle=0,
            n_labeled=100,
            n_acquired=0,
            split="validation",
            metrics={},
        )


def test_non_numeric_metric_raises() -> None:
    """Stored metric values must be convertible to floats."""
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        EvaluationRecord(
            cycle=0,
            n_labeled=100,
            n_acquired=0,
            split="validation",
            metrics={
                "accuracy": "not-a-number",  # type: ignore[dict-item]
            },
        )


def test_invalid_json_structure_raises(tmp_path) -> None:
    """History loading should reject files with the wrong structure."""
    path = tmp_path / "invalid.json"

    path.write_text(
        json.dumps(
            {
                "cycle": 0,
                "metrics": {"accuracy": 1.0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain a list of records",
    ):
        EvaluationHistory.load(path)
