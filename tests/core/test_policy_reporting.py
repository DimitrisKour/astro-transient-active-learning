import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "policy_report", Path(__file__).resolve().parents[2] / "scripts/report_policy_compare.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def rows(values):
    return [
        {"cycle": i, "n_labeled": n, "metrics": {"f1_macro": y}} for i, (n, y) in enumerate(values)
    ]


def test_report_distinguishes_transient_and_persistent_crossings():
    summary = report.curve_summary(
        rows([(10, 0.4), (30, 0.79), (60, 0.5), (100, 0.8)]), 100, metric="f1_macro"
    )
    assert summary["own_full_reference_threshold"] == pytest.approx(0.78)
    assert summary["first_n_labeled_at_threshold"] == 30
    assert summary["persistent_n_labeled_at_threshold"] == 100
    assert summary["early_fraction_limit_observed"] == 0.5


def test_incomplete_or_non_improving_full_curve_has_no_gain_threshold():
    for values in [[(10, 0.4), (30, 0.7)], [(10, 0.7), (100, 0.6)]]:
        summary = report.curve_summary(rows(values), 100, metric="f1_macro")
        assert summary["own_full_reference_threshold"] is None
        assert summary["first_n_labeled_at_threshold"] is None
