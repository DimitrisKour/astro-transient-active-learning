"""Persistent state for adaptive-learning experiment cycles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


_VALID_STATUSES = {
    "ready",
    "active",
    "pending",
}


@dataclass(slots=True)
class RunState:
    """Track the current cycle of one adaptive-learning run.

    Cycle 0 represents the historical labeled baseline L0. Acquisition
    therefore begins at cycle 1.
    """

    path: Path
    last_completed_cycle: int = 0
    active_cycle: int | None = None
    status: str = "ready"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._validate()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> RunState:
        """Create and persist a new run state.

        New runs begin at historical baseline cycle 0.
        """
        state_path = Path(path)

        if state_path.exists() and not overwrite:
            raise FileExistsError(f"Run state already exists: {state_path}")

        state = cls(
            path=state_path,
            last_completed_cycle=0,
            active_cycle=None,
            status="ready",
        )
        state.save()

        return state

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> RunState:
        """Load previously persisted run state."""
        state_path = Path(path)

        if not state_path.exists():
            raise FileNotFoundError(f"Run state does not exist: {state_path}")

        with state_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        try:
            last_completed_cycle = payload["last_completed_cycle"]
            active_cycle = payload["active_cycle"]
            status = payload["status"]
        except KeyError as exc:
            raise ValueError(f"Run state is missing required field: {exc.args[0]!r}.") from exc

        return cls(
            path=state_path,
            last_completed_cycle=last_completed_cycle,
            active_cycle=active_cycle,
            status=status,
        )

    @property
    def next_cycle(self) -> int:
        """Return the next cycle number without changing state."""
        if self.active_cycle is not None:
            return self.active_cycle

        return self.last_completed_cycle + 1

    def start_next_cycle(self) -> int:
        """Start a new cycle or resume the currently active one.

        If an unfinished cycle already exists, its number is returned
        instead of silently creating another cycle.
        """
        if self.active_cycle is not None:
            return self.active_cycle

        cycle = self.last_completed_cycle + 1

        self.active_cycle = cycle
        self.status = "active"
        self.save()

        return cycle

    def mark_pending(
        self,
        cycle: int | None = None,
    ) -> None:
        """Mark the active cycle as waiting for oracle resolution."""
        active = self._require_active_cycle(cycle)

        self.active_cycle = active
        self.status = "pending"
        self.save()

    def mark_active(
        self,
        cycle: int | None = None,
    ) -> None:
        """Mark a pending cycle as actively processing again."""
        active = self._require_active_cycle(cycle)

        self.active_cycle = active
        self.status = "active"
        self.save()

    def complete_cycle(
        self,
        cycle: int | None = None,
    ) -> int:
        """Complete the active cycle and return its cycle number."""
        active = self._require_active_cycle(cycle)

        self.last_completed_cycle = active
        self.active_cycle = None
        self.status = "ready"
        self.save()

        return active

    def save(self) -> None:
        """Persist state atomically as JSON."""
        self._validate()

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "last_completed_cycle": self.last_completed_cycle,
            "active_cycle": self.active_cycle,
            "status": self.status,
        }

        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )
            file.write("\n")

        temporary_path.replace(self.path)

        # TODO: Add inter-process locking if multiple workers are ever
        # allowed to advance the same adaptive-learning run concurrently.

    def _require_active_cycle(
        self,
        cycle: int | None,
    ) -> int:
        """Return the active cycle and optionally verify its number."""
        if self.active_cycle is None:
            raise RuntimeError("There is no active cycle.")

        if cycle is not None and cycle != self.active_cycle:
            raise ValueError(f"Cycle {cycle} is not the active cycle {self.active_cycle}.")

        return self.active_cycle

    def _validate(self) -> None:
        """Validate internal cycle/status consistency."""
        if (
            not isinstance(self.last_completed_cycle, int)
            or isinstance(self.last_completed_cycle, bool)
            or self.last_completed_cycle < 0
        ):
            raise ValueError("last_completed_cycle must be a non-negative integer.")

        if self.active_cycle is not None:
            if (
                not isinstance(self.active_cycle, int)
                or isinstance(self.active_cycle, bool)
                or self.active_cycle <= 0
            ):
                raise ValueError("active_cycle must be a positive integer or None.")

            expected = self.last_completed_cycle + 1

            if self.active_cycle != expected:
                raise ValueError("active_cycle must equal last_completed_cycle + 1.")

        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Invalid run status: {self.status!r}.")

        if self.status == "ready" and self.active_cycle is not None:
            raise ValueError("A ready run cannot have an active cycle.")

        if self.status in {"active", "pending"}:
            if self.active_cycle is None:
                raise ValueError(f"A {self.status!r} run must have an active cycle.")
