"""Tests for persistent object-level acquisition history."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from al_pipeline.core.oracle.resolution import LabelResolution, LabelResolutionBatch, LabelStatus
from al_pipeline.core.state.history import SQLiteObjectHistoryStore
from al_pipeline.core.types import SelectionBatch


def _selection(
    ids: list[str],
    scores: list[float],
) -> SelectionBatch:
    return SelectionBatch(
        ids=ids,
        indices=np.arange(len(ids)),
        scores=np.asarray(scores, dtype=float),
    )


def test_records_and_reloads_selection_history(tmp_path) -> None:
    path = tmp_path / "object_history.sqlite"
    store = SQLiteObjectHistoryStore(path)
    selected_at = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc)

    events = store.record_selection(
        _selection(
            ["ZTF_A", "ZTF_B"],
            [0.91, 0.73],
        ),
        cycle=1,
        run_id="historical-demo",
        scorer="entropy",
        selected_at=selected_at,
        metadata={"budget": 2},
    )

    assert [event.rank for event in events] == [1, 2]
    assert events[0].candidate_index == 0
    assert events[0].metadata["budget"] == 2

    reopened = SQLiteObjectHistoryStore(path)
    loaded = reopened.acquisition_events()

    assert [event.object_id for event in loaded] == ["ZTF_A", "ZTF_B"]
    assert loaded[0].score == pytest.approx(0.91)
    assert loaded[0].candidate_index == 0
    assert loaded[0].selected_at == selected_at
    assert loaded[0].cycle == 1
    assert loaded[0].run_id == "historical-demo"
    assert loaded[0].scorer == "entropy"


def test_repeated_acquisition_is_preserved_as_history(tmp_path) -> None:
    store = SQLiteObjectHistoryStore(tmp_path / "history.sqlite")
    first = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc)
    second = first + timedelta(days=1)

    store.record_selection(
        _selection(["ZTF_REPEAT"], [0.8]),
        cycle=1,
        selected_at=first,
    )
    store.record_selection(
        _selection(["ZTF_REPEAT", "ZTF_OTHER"], [0.9, 0.7]),
        cycle=2,
        selected_at=second,
    )

    assert store.acquisition_count("ZTF_REPEAT") == 2
    assert store.acquisition_count("ZTF_OTHER") == 1
    assert store.acquired_ids() == ("ZTF_REPEAT", "ZTF_OTHER")
    assert store.acquired_ids(minimum_count=2) == ("ZTF_REPEAT",)

    summary = store.summary("ZTF_REPEAT")

    assert summary.acquisition_count == 2
    assert summary.first_acquired_at == first
    assert summary.last_acquired_at == second
    assert summary.last_score == pytest.approx(0.9)
    assert summary.latest_resolution is None


def test_records_all_resolution_states_without_policy(tmp_path) -> None:
    store = SQLiteObjectHistoryStore(tmp_path / "history.sqlite")
    resolved_at = datetime(2026, 9, 14, 11, 0, tzinfo=timezone.utc)

    batch = LabelResolutionBatch(
        [
            LabelResolution(
                object_id="AGN_OBJECT",
                status=LabelStatus.LABELED,
                label="AGN",
                source="multiple",
                native_label=("Q", "QSO"),
                provenance={"claims": 2},
            ),
            LabelResolution(
                object_id="VARIABLE_OBJECT",
                status=LabelStatus.OUTSIDE_LABEL_SPACE,
                source="VSX",
                native_label="RRAB",
                provenance={"separation_arcsec": 0.2},
            ),
            LabelResolution(
                object_id="UNKNOWN_OBJECT",
                status=LabelStatus.UNRESOLVED,
                provenance={"reason": "no_resolving_evidence"},
            ),
        ]
    )

    store.record_resolutions(
        batch,
        cycle=3,
        run_id="resolver-run",
        resolved_at=resolved_at,
    )

    agn = store.latest_resolution("AGN_OBJECT")
    variable = store.latest_resolution("VARIABLE_OBJECT")
    unknown = store.latest_resolution("UNKNOWN_OBJECT")

    assert agn is not None
    assert agn.status is LabelStatus.LABELED
    assert agn.label == "AGN"
    assert agn.native_label == ["Q", "QSO"]
    assert agn.provenance["claims"] == 2
    assert agn.resolved_at == resolved_at

    assert variable is not None
    assert variable.status is LabelStatus.OUTSIDE_LABEL_SPACE
    assert variable.label is None

    assert unknown is not None
    assert unknown.status is LabelStatus.UNRESOLVED
    assert unknown.provenance["reason"] == "no_resolving_evidence"

    # History records outcomes only. It deliberately does not decide which
    # of these objects may be selected again.
    assert store.summary("UNKNOWN_OBJECT").latest_resolution == unknown


def test_latest_resolution_tracks_newer_attempt(tmp_path) -> None:
    store = SQLiteObjectHistoryStore(tmp_path / "history.sqlite")
    first_time = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc)
    second_time = first_time + timedelta(hours=2)

    store.record_resolutions(
        LabelResolutionBatch(
            [
                LabelResolution(
                    object_id="ZTF_X",
                    status=LabelStatus.UNRESOLVED,
                    provenance={"reason": "no_match"},
                )
            ]
        ),
        cycle=1,
        resolved_at=first_time,
    )

    store.record_resolutions(
        LabelResolutionBatch(
            [
                LabelResolution(
                    object_id="ZTF_X",
                    status=LabelStatus.LABELED,
                    label="AGN",
                    source="DESI_DR1",
                    native_label="QSO",
                )
            ]
        ),
        cycle=2,
        resolved_at=second_time,
    )

    events = store.resolution_events("ZTF_X")

    assert [event.status for event in events] == [
        LabelStatus.UNRESOLVED,
        LabelStatus.LABELED,
    ]
    assert store.latest_resolution("ZTF_X") == events[-1]
    assert store.summary("ZTF_X").latest_resolution == events[-1]


def test_summary_for_unseen_object_is_empty(tmp_path) -> None:
    store = SQLiteObjectHistoryStore(tmp_path / "history.sqlite")

    summary = store.summary("NEVER_SEEN")

    assert summary.object_id == "NEVER_SEEN"
    assert summary.acquisition_count == 0
    assert summary.first_acquired_at is None
    assert summary.last_acquired_at is None
    assert summary.last_score is None
    assert summary.latest_resolution is None


def test_selection_timestamp_must_be_timezone_aware(tmp_path) -> None:
    store = SQLiteObjectHistoryStore(tmp_path / "history.sqlite")

    with pytest.raises(ValueError, match="timezone-aware"):
        store.record_selection(
            _selection(["ZTF_A"], [0.5]),
            selected_at=datetime(2026, 9, 14, 10, 0),
        )


def test_invalid_reacquisition_count_filter_is_rejected(tmp_path) -> None:
    store = SQLiteObjectHistoryStore(tmp_path / "history.sqlite")

    with pytest.raises(ValueError, match="positive integer"):
        store.acquired_ids(minimum_count=0)
