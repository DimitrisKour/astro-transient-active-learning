import numpy as np
import pandas as pd
import pytest

from al_pipeline.core.evaluation.control_sampling import sample_exact_matched_control


def _candidate_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obj_id": [
                "a0",
                "a1",
                "a2",
                "a3",
                "b0",
                "b1",
                "c0",
                "c1",
            ],
            "alert": [
                0,
                0,
                0,
                0,
                3,
                3,
                6,
                6,
            ],
            "value": np.arange(8),
        }
    )


def test_exact_matched_control_matches_selected_distribution():
    candidates = _candidate_table()

    selected = candidates.iloc[[0, 1, 4, 6]].copy()

    control = sample_exact_matched_control(
        candidates,
        selected,
        match_on=("alert",),
        rng=np.random.default_rng(42),
    )

    selected_counts = selected["alert"].value_counts().sort_index()

    control_counts = control["alert"].value_counts().sort_index()

    assert selected_counts.equals(control_counts)

    assert set(selected["obj_id"]).isdisjoint(control["obj_id"])


def test_exact_matched_control_is_reproducible():
    candidates = _candidate_table()

    selected = candidates.iloc[[0, 4, 6]].copy()

    control_1 = sample_exact_matched_control(
        candidates,
        selected,
        match_on=("alert",),
        rng=np.random.default_rng(135),
    )

    control_2 = sample_exact_matched_control(
        candidates,
        selected,
        match_on=("alert",),
        rng=np.random.default_rng(135),
    )

    assert control_1["obj_id"].tolist() == (control_2["obj_id"].tolist())


def test_exact_matched_control_raises_when_stratum_is_too_small():
    candidates = _candidate_table().iloc[:5].copy()

    selected = pd.DataFrame(
        {
            "obj_id": [
                "selected_1",
                "selected_2",
            ],
            "alert": [
                3,
                3,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Insufficient eligible controls",
    ):
        sample_exact_matched_control(
            candidates,
            selected,
            match_on=("alert",),
            rng=np.random.default_rng(1),
        )
