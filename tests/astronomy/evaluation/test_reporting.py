"""Tests for validated multi-seed active-learning reports."""

import csv
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from al_pipeline.astronomy.evaluation.reporting import (
    SuiteSpec,
    build_multiseed_report,
    write_report_plots,
    write_report_tables,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_suite(tmp_path: Path) -> SuiteSpec:
    state_root = tmp_path / "state" / "suite"
    artifact_root = tmp_path / "artifacts" / "suite"
    strategies = ("random", "entropy")
    seeds = (1, 2)

    for strategy in strategies:
        for seed in seeds:
            state = state_root / strategy / f"seed_{seed}"
            artifacts = artifact_root / strategy / f"seed_{seed}"
            experiment = {
                "baseline_checkpoint_sha256": "baseline",
                "candidate_manifest_sha256": "candidate",
                "hidden_labels_sha256": "labels",
                "historical_train_manifest_sha256": "train",
                "pool_id": "pool",
                "validation_manifest_sha256": "validation",
                "holdout_ids": ["holdout_a", "holdout_b"],
                "config": {
                    "strategy": strategy,
                    "seed": seed,
                    "max_cycles": 2,
                    "acquisition_size": 2,
                    "training_initialization": "warm_start",
                },
            }
            _write_json(state / "experiment.json", experiment)
            _write_json(
                state / "run_state.json",
                {
                    "last_completed_cycle": 2,
                    "active_cycle": None,
                    "status": "ready",
                },
            )

            history = []
            strategy_offset = 0.10 if strategy == "entropy" else 0.0
            for cycle in range(3):
                for split in ("test", "new_holdout"):
                    value = 0.50 + 0.01 * seed + strategy_offset + 0.02 * cycle
                    history.append(
                        {
                            "cycle": cycle,
                            "n_labeled": 10 + 2 * cycle,
                            "n_acquired": 0 if cycle == 0 else 2,
                            "split": split,
                            "metrics": {
                                "accuracy": value,
                                "balanced_accuracy": value - 0.02,
                                "roc_auc_macro": value + 0.10,
                                "average_precision_macro": value + 0.05,
                                "log_loss": 1.0 - value,
                                "ece": 0.1,
                            },
                            "strategy": strategy,
                            "seed": seed,
                            "run_id": f"suite_{strategy}_seed{seed}",
                        }
                    )
            _write_json(state / "evaluation_history.json", history)

            acquisitions = [
                {
                    "cycle": 1,
                    "object_id": f"{strategy}_{seed}_a",
                    "survey": "ZTF",
                    "label": "SN I",
                    "strategy": strategy,
                    "data_reference": "a",
                },
                {
                    "cycle": 1,
                    "object_id": f"{strategy}_{seed}_b",
                    "survey": "ZTF",
                    "label": "SN II",
                    "strategy": strategy,
                    "data_reference": "b",
                },
                {
                    "cycle": 2,
                    "object_id": f"shared_{seed}_c",
                    "survey": "ZTF",
                    "label": "SN I",
                    "strategy": strategy,
                    "data_reference": "c",
                },
                {
                    "cycle": 2,
                    "object_id": f"shared_{seed}_d",
                    "survey": "ZTF",
                    "label": "SN I",
                    "strategy": strategy,
                    "data_reference": "d",
                },
            ]
            _write_csv(
                state / "acquisitions.csv",
                (
                    "cycle",
                    "object_id",
                    "survey",
                    "label",
                    "strategy",
                    "data_reference",
                ),
                acquisitions,
            )

            for cycle in range(3):
                evaluation = artifacts / f"cycle_{cycle:03d}" / "evaluation"
                _write_json(
                    evaluation / "evaluation_summary.json",
                    {
                        "model_class_names": [
                            "AGN",
                            "TDE",
                            "SN I",
                            "SN II",
                            "CV",
                        ]
                    },
                )
                for split in ("test", "new_holdout"):
                    _write_json(
                        evaluation / f"{split}_per_class_metrics.json",
                        {
                            "object": {
                                "SN I": {
                                    "precision": 0.7,
                                    "recall": 0.6 + 0.02 * cycle,
                                    "f1": 0.65,
                                    "support": 8,
                                },
                                "SN II": {
                                    "precision": 0.5,
                                    "recall": 0.4 + 0.01 * cycle,
                                    "f1": 0.45,
                                    "support": 2,
                                },
                            },
                            "sample": {},
                        },
                    )

                if cycle > 0:
                    _write_json(
                        artifacts / f"cycle_{cycle:03d}" / "training" / "training_summary.json",
                        {
                            "best_epoch": 2,
                            "epochs_completed": 7,
                            "best_validation_loss": 0.9,
                            "elapsed_seconds": 60.0,
                            "peak_cuda_memory_bytes": 1000,
                            "training_samples": 100 + cycle,
                        },
                    )

    return SuiteSpec(
        label="warm_start",
        state_dir=state_root,
        artifact_dir=artifact_root,
    )


def test_multiseed_report_aggregates_seed_mean_and_std(
    tmp_path: Path,
) -> None:
    suite = _make_suite(tmp_path)
    report = build_multiseed_report(
        (suite,),
        expected_strategies=("random", "entropy"),
        expected_seeds=(1, 2),
        expected_cycles=2,
    )

    match = next(
        record
        for record in report.evaluation_aggregates
        if record["strategy"] == "entropy"
        and record["split"] == "test"
        and record["cycle"] == 1
        and record["metric"] == "accuracy"
    )

    assert match["value_n"] == 2
    assert match["value_mean"] == pytest.approx(0.635)
    assert match["value_std"] == pytest.approx(0.0070710678)
    assert match["n_acquired_total_mean"] == pytest.approx(2.0)


def test_acquisition_aggregation_includes_unselected_classes_as_zero(
    tmp_path: Path,
) -> None:
    suite = _make_suite(tmp_path)
    report = build_multiseed_report(
        (suite,),
        expected_strategies=("random", "entropy"),
        expected_seeds=(1, 2),
        expected_cycles=2,
    )

    match = next(
        record
        for record in report.acquisition_aggregates
        if record["strategy"] == "random" and record["cycle"] == 1 and record["label"] == "TDE"
    )
    assert match["value_n"] == 2
    assert match["value_mean"] == pytest.approx(0.0)
    assert match["cumulative_count_mean"] == pytest.approx(0.0)


def test_report_writes_tables_and_figures(tmp_path: Path) -> None:
    suite = _make_suite(tmp_path)
    report = build_multiseed_report(
        (suite,),
        expected_strategies=("random", "entropy"),
        expected_seeds=(1, 2),
        expected_cycles=2,
    )
    output = tmp_path / "report"

    tables = write_report_tables(report, output)
    plots = write_report_plots(
        report,
        output,
        splits=("test", "new_holdout"),
        formats=("png",),
    )

    assert tables
    assert plots
    assert all(path.is_file() for path in tables)
    assert all(path.is_file() and path.stat().st_size > 0 for path in plots)
    assert (output / "trajectory_aggregates.csv").is_file()
    assert (output / "object_metrics_test.png").is_file()
    assert (output / "object_recall_new_holdout.png").is_file()


def test_report_rejects_different_holdout(tmp_path: Path) -> None:
    suite = _make_suite(tmp_path)
    path = suite.state_dir / "entropy" / "seed_2" / "experiment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["holdout_ids"] = ["different"]
    _write_json(path, payload)

    with pytest.raises(ValueError, match="different new-data holdout"):
        build_multiseed_report(
            (suite,),
            expected_strategies=("random", "entropy"),
            expected_seeds=(1, 2),
            expected_cycles=2,
        )


def test_report_allows_seed_specific_shared_bootstrap_checkpoints(
    tmp_path: Path,
) -> None:
    suite = _make_suite(tmp_path)
    for strategy in ("random", "entropy"):
        path = suite.state_dir / strategy / "seed_2" / "experiment.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["baseline_checkpoint_sha256"] = "seed-two-bootstrap"
        _write_json(path, payload)

    report = build_multiseed_report(
        (suite,),
        expected_strategies=("random", "entropy"),
        expected_seeds=(1, 2),
        expected_cycles=2,
    )

    assert report.validation["passed"] is True


def test_report_rejects_strategy_specific_bootstrap_checkpoint(
    tmp_path: Path,
) -> None:
    suite = _make_suite(tmp_path)
    path = suite.state_dir / "entropy" / "seed_2" / "experiment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["baseline_checkpoint_sha256"] = "wrong-strategy-bootstrap"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="seed-specific baseline checkpoint"):
        build_multiseed_report(
            (suite,),
            expected_strategies=("random", "entropy"),
            expected_seeds=(1, 2),
            expected_cycles=2,
        )
