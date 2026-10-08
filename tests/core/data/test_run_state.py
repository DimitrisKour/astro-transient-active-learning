"""Tests for persistent adaptive-learning run state."""

import json
from pathlib import Path

import pytest

from al_pipeline.core.state.run_state import RunState


def test_new_run_starts_at_cycle_zero(
    tmp_path: Path,
) -> None:
    """Historical L0 should be represented by completed cycle zero."""
    path = tmp_path / "run_state.json"

    state = RunState.create(path)

    assert state.last_completed_cycle == 0
    assert state.active_cycle is None
    assert state.status == "ready"
    assert state.next_cycle == 1


def test_start_next_cycle(
    tmp_path: Path,
) -> None:
    """First acquisition cycle should be numbered one."""
    state = RunState.create(tmp_path / "run_state.json")

    cycle = state.start_next_cycle()

    assert cycle == 1
    assert state.active_cycle == 1
    assert state.last_completed_cycle == 0
    assert state.status == "active"


def test_unfinished_cycle_is_resumed(
    tmp_path: Path,
) -> None:
    """Starting again must not skip an unfinished cycle."""
    state = RunState.create(tmp_path / "run_state.json")

    first = state.start_next_cycle()
    second = state.start_next_cycle()

    assert first == 1
    assert second == 1
    assert state.active_cycle == 1


def test_pending_cycle_is_resumed_after_reload(
    tmp_path: Path,
) -> None:
    """Pending state should survive process restarts."""
    path = tmp_path / "run_state.json"

    state = RunState.create(path)
    state.start_next_cycle()
    state.mark_pending()

    restored = RunState.load(path)

    assert restored.last_completed_cycle == 0
    assert restored.active_cycle == 1
    assert restored.status == "pending"
    assert restored.start_next_cycle() == 1


def test_complete_cycle_advances_run(
    tmp_path: Path,
) -> None:
    """Completing one cycle should make the following cycle available."""
    state = RunState.create(tmp_path / "run_state.json")

    state.start_next_cycle()
    completed = state.complete_cycle()

    assert completed == 1
    assert state.last_completed_cycle == 1
    assert state.active_cycle is None
    assert state.status == "ready"
    assert state.next_cycle == 2


def test_multiple_cycles_advance_automatically(
    tmp_path: Path,
) -> None:
    """Repeated cycles should never require hard-coded numbers."""
    state = RunState.create(tmp_path / "run_state.json")

    assert state.start_next_cycle() == 1
    state.complete_cycle()

    assert state.start_next_cycle() == 2
    state.complete_cycle()

    assert state.start_next_cycle() == 3


def test_mark_pending_and_return_active(
    tmp_path: Path,
) -> None:
    """A pending cycle can return to active processing."""
    state = RunState.create(tmp_path / "run_state.json")

    state.start_next_cycle()
    state.mark_pending()

    assert state.status == "pending"

    state.mark_active()

    assert state.status == "active"
    assert state.active_cycle == 1


def test_cycle_number_can_be_verified(
    tmp_path: Path,
) -> None:
    """Operations may explicitly verify the expected active cycle."""
    state = RunState.create(tmp_path / "run_state.json")

    state.start_next_cycle()

    with pytest.raises(
        ValueError,
        match="is not the active cycle",
    ):
        state.complete_cycle(2)


def test_cannot_complete_without_active_cycle(
    tmp_path: Path,
) -> None:
    """A ready run cannot complete a nonexistent cycle."""
    state = RunState.create(tmp_path / "run_state.json")

    with pytest.raises(
        RuntimeError,
        match="no active cycle",
    ):
        state.complete_cycle()


def test_existing_state_is_not_overwritten_by_default(
    tmp_path: Path,
) -> None:
    """Accidental run reinitialization should fail."""
    path = tmp_path / "run_state.json"

    RunState.create(path)

    with pytest.raises(FileExistsError):
        RunState.create(path)


def test_state_is_persisted(
    tmp_path: Path,
) -> None:
    """State changes should immediately appear on disk."""
    path = tmp_path / "run_state.json"

    state = RunState.create(path)
    state.start_next_cycle()
    state.mark_pending()

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {
        "last_completed_cycle": 0,
        "active_cycle": 1,
        "status": "pending",
    }


def test_invalid_state_is_rejected(
    tmp_path: Path,
) -> None:
    """Inconsistent persisted state should not be silently accepted."""
    path = tmp_path / "run_state.json"

    path.write_text(
        json.dumps(
            {
                "last_completed_cycle": 2,
                "active_cycle": 4,
                "status": "active",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="last_completed_cycle",
    ):
        RunState.load(path)
