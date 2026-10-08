"""Persistent object-level history for adaptive candidate workflows.

This module records what happened to an object across acquisition cycles. It
is deliberately observational: it does not decide whether an object may be
selected again, how long an unresolved object should cool down, or whether an
outside-taxonomy object should be retained. Those are workflow-policy choices
that can be made from the recorded history later.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from al_pipeline.core.oracle.resolution import LabelResolutionBatch, LabelStatus
from al_pipeline.core.types import SelectionBatch


@dataclass(frozen=True, slots=True)
class AcquisitionEvent:
    """One committed acquisition of one object."""

    object_id: str
    score: float
    rank: int
    candidate_index: int
    selected_at: datetime
    cycle: int | None = None
    run_id: str | None = None
    scorer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = _normalize_nonempty_string(
            self.object_id,
            "object_id",
        )
        score = float(self.score)

        if not math.isfinite(score):
            raise ValueError("score must be finite.")

        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank <= 0:
            raise ValueError("rank must be a positive integer.")

        if (
            not isinstance(self.candidate_index, int)
            or isinstance(self.candidate_index, bool)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer.")

        cycle = _normalize_cycle(self.cycle)
        selected_at = _normalize_timestamp(self.selected_at)
        run_id = _normalize_optional_string(self.run_id, "run_id")
        scorer = _normalize_optional_string(self.scorer, "scorer")

        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "cycle", cycle)
        object.__setattr__(self, "selected_at", selected_at)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "scorer", scorer)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class ResolutionEvent:
    """One persisted label-resolution outcome for one object."""

    object_id: str
    status: LabelStatus
    resolved_at: datetime
    label: object | None = None
    source: str | None = None
    native_label: object | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    cycle: int | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        object_id = _normalize_nonempty_string(
            self.object_id,
            "object_id",
        )

        if not isinstance(self.status, LabelStatus):
            raise TypeError("status must be a LabelStatus.")

        if self.status is LabelStatus.LABELED:
            if self.label is None:
                raise ValueError("A labeled resolution event must contain a label.")
        elif self.label is not None:
            raise ValueError("Only labeled resolution events may contain a label.")

        resolved_at = _normalize_timestamp(self.resolved_at)
        cycle = _normalize_cycle(self.cycle)
        run_id = _normalize_optional_string(self.run_id, "run_id")
        source = _normalize_optional_string(self.source, "source")

        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "resolved_at", resolved_at)
        object.__setattr__(self, "cycle", cycle)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


@dataclass(frozen=True, slots=True)
class ObjectHistorySummary:
    """Compact derived history for one object.

    This summary intentionally stops short of an eligibility decision. For
    example, ``latest_resolution.status == UNRESOLVED`` does not imply a
    retry schedule, and ``OUTSIDE_LABEL_SPACE`` does not imply permanent
    exclusion.
    """

    object_id: str
    acquisition_count: int
    first_acquired_at: datetime | None
    last_acquired_at: datetime | None
    last_score: float | None
    latest_resolution: ResolutionEvent | None


class ObjectHistoryStore(Protocol):
    """Persistence boundary for object-level acquisition history."""

    def record_selection(
        self,
        selection: SelectionBatch,
        *,
        cycle: int | None = None,
        run_id: str | None = None,
        scorer: str | None = None,
        selected_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[AcquisitionEvent, ...]:
        """Persist one committed selection batch."""
        ...

    def record_resolutions(
        self,
        resolutions: LabelResolutionBatch,
        *,
        cycle: int | None = None,
        run_id: str | None = None,
        resolved_at: datetime | None = None,
    ) -> tuple[ResolutionEvent, ...]:
        """Persist one batch of resolution outcomes."""
        ...

    def summary(self, object_id: str) -> ObjectHistorySummary:
        """Return a compact history summary for one object."""
        ...


class SQLiteObjectHistoryStore:
    """SQLite-backed implementation of :class:`ObjectHistoryStore`.

    SQLite is used only as the local persistence backend. Broker deployment
    can provide another implementation of ``ObjectHistoryStore`` without
    changing acquisition or resolution semantics.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

        if str(self.path) != ":memory:":
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize()

    def record_selection(
        self,
        selection: SelectionBatch,
        *,
        cycle: int | None = None,
        run_id: str | None = None,
        scorer: str | None = None,
        selected_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[AcquisitionEvent, ...]:
        """Persist every object in one committed selection batch.

        The order in ``selection`` becomes a one-based acquisition rank. A
        repeated object ID across different calls is retained as a new event;
        this is the behaviour needed to detect re-acquisition across cycles.
        """
        cycle = _normalize_cycle(cycle)
        run_id = _normalize_optional_string(run_id, "run_id")
        scorer = _normalize_optional_string(scorer, "scorer")
        timestamp = _normalize_timestamp(selected_at)
        metadata_dict = {} if metadata is None else dict(metadata)

        if len(selection.ids) != len(set(selection.ids)):
            raise ValueError("A selection batch must not contain duplicate object IDs.")

        events = tuple(
            AcquisitionEvent(
                object_id=object_id,
                score=float(score),
                rank=rank,
                candidate_index=int(candidate_index),
                selected_at=timestamp,
                cycle=cycle,
                run_id=run_id,
                scorer=scorer,
                metadata=metadata_dict,
            )
            for rank, (object_id, score, candidate_index) in enumerate(
                zip(
                    selection.ids,
                    selection.scores,
                    selection.indices,
                    strict=True,
                ),
                start=1,
            )
        )

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO acquisition_events (
                    object_id,
                    score,
                    rank,
                    candidate_index,
                    selected_at,
                    cycle,
                    run_id,
                    scorer,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.object_id,
                        event.score,
                        event.rank,
                        event.candidate_index,
                        _timestamp_to_text(event.selected_at),
                        event.cycle,
                        event.run_id,
                        event.scorer,
                        _json_dumps(dict(event.metadata)),
                    )
                    for event in events
                ],
            )

        return events

    def record_resolutions(
        self,
        resolutions: LabelResolutionBatch,
        *,
        cycle: int | None = None,
        run_id: str | None = None,
        resolved_at: datetime | None = None,
    ) -> tuple[ResolutionEvent, ...]:
        """Persist every outcome from one resolution batch."""
        cycle = _normalize_cycle(cycle)
        run_id = _normalize_optional_string(run_id, "run_id")
        timestamp = _normalize_timestamp(resolved_at)

        events = tuple(
            ResolutionEvent(
                object_id=result.object_id,
                status=result.status,
                label=result.label,
                source=result.source,
                native_label=result.native_label,
                provenance=({} if result.provenance is None else dict(result.provenance)),
                resolved_at=timestamp,
                cycle=cycle,
                run_id=run_id,
            )
            for result in resolutions.resolutions
        )

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO resolution_events (
                    object_id,
                    status,
                    label_json,
                    source,
                    native_label_json,
                    provenance_json,
                    resolved_at,
                    cycle,
                    run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.object_id,
                        event.status.value,
                        _json_dumps(event.label),
                        event.source,
                        _json_dumps(event.native_label),
                        _json_dumps(dict(event.provenance)),
                        _timestamp_to_text(event.resolved_at),
                        event.cycle,
                        event.run_id,
                    )
                    for event in events
                ],
            )

        return events

    def acquisition_events(
        self,
        object_id: str | None = None,
    ) -> tuple[AcquisitionEvent, ...]:
        """Return acquisition events in insertion order."""
        parameters: Sequence[object]

        if object_id is None:
            where = ""
            parameters = ()
        else:
            object_id = _normalize_nonempty_string(
                object_id,
                "object_id",
            )
            where = "WHERE object_id = ?"
            parameters = (object_id,)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    object_id,
                    score,
                    rank,
                    candidate_index,
                    selected_at,
                    cycle,
                    run_id,
                    scorer,
                    metadata_json
                FROM acquisition_events
                {where}
                ORDER BY id ASC
                """,
                parameters,
            ).fetchall()

        return tuple(
            AcquisitionEvent(
                object_id=row["object_id"],
                score=row["score"],
                rank=row["rank"],
                candidate_index=row["candidate_index"],
                selected_at=_timestamp_from_text(row["selected_at"]),
                cycle=row["cycle"],
                run_id=row["run_id"],
                scorer=row["scorer"],
                metadata=_json_loads(row["metadata_json"]),
            )
            for row in rows
        )

    def resolution_events(
        self,
        object_id: str | None = None,
    ) -> tuple[ResolutionEvent, ...]:
        """Return resolution events in insertion order."""
        parameters: Sequence[object]

        if object_id is None:
            where = ""
            parameters = ()
        else:
            object_id = _normalize_nonempty_string(
                object_id,
                "object_id",
            )
            where = "WHERE object_id = ?"
            parameters = (object_id,)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    object_id,
                    status,
                    label_json,
                    source,
                    native_label_json,
                    provenance_json,
                    resolved_at,
                    cycle,
                    run_id
                FROM resolution_events
                {where}
                ORDER BY id ASC
                """,
                parameters,
            ).fetchall()

        return tuple(
            ResolutionEvent(
                object_id=row["object_id"],
                status=LabelStatus(row["status"]),
                label=_json_loads(row["label_json"]),
                source=row["source"],
                native_label=_json_loads(row["native_label_json"]),
                provenance=_json_loads(row["provenance_json"]),
                resolved_at=_timestamp_from_text(row["resolved_at"]),
                cycle=row["cycle"],
                run_id=row["run_id"],
            )
            for row in rows
        )

    def acquisition_count(self, object_id: str) -> int:
        """Return how many times an object has been acquired."""
        object_id = _normalize_nonempty_string(
            object_id,
            "object_id",
        )

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM acquisition_events
                WHERE object_id = ?
                """,
                (object_id,),
            ).fetchone()

        assert row is not None
        return int(row["count"])

    def acquired_ids(
        self,
        *,
        minimum_count: int = 1,
    ) -> tuple[str, ...]:
        """Return IDs acquired at least ``minimum_count`` times.

        ``minimum_count=2`` is a convenient way to inspect re-acquisitions
        without embedding any policy about whether they should be allowed.
        """
        if (
            not isinstance(minimum_count, int)
            or isinstance(minimum_count, bool)
            or minimum_count <= 0
        ):
            raise ValueError("minimum_count must be a positive integer.")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT object_id, MIN(id) AS first_id
                FROM acquisition_events
                GROUP BY object_id
                HAVING COUNT(*) >= ?
                ORDER BY first_id ASC
                """,
                (minimum_count,),
            ).fetchall()

        return tuple(row["object_id"] for row in rows)

    def latest_resolution(
        self,
        object_id: str,
    ) -> ResolutionEvent | None:
        """Return the most recently recorded resolution for one object."""
        events = self.resolution_events(object_id)

        if not events:
            return None

        return events[-1]

    def summary(self, object_id: str) -> ObjectHistorySummary:
        """Return a compact acquisition/resolution summary for one object."""
        object_id = _normalize_nonempty_string(
            object_id,
            "object_id",
        )
        acquisitions = self.acquisition_events(object_id)
        latest = self.latest_resolution(object_id)

        if acquisitions:
            first_acquired_at = acquisitions[0].selected_at
            last_acquired_at = acquisitions[-1].selected_at
            last_score = acquisitions[-1].score
        else:
            first_acquired_at = None
            last_acquired_at = None
            last_score = None

        return ObjectHistorySummary(
            object_id=object_id,
            acquisition_count=len(acquisitions),
            first_acquired_at=first_acquired_at,
            last_acquired_at=last_acquired_at,
            last_score=last_score,
            latest_resolution=latest,
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])

            if version not in {0, self._SCHEMA_VERSION}:
                raise RuntimeError(f"Unsupported object-history schema version: {version}.")

            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS acquisition_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    candidate_index INTEGER NOT NULL,
                    selected_at TEXT NOT NULL,
                    cycle INTEGER,
                    run_id TEXT,
                    scorer TEXT,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS
                    idx_acquisition_object_id
                ON acquisition_events(object_id);

                CREATE TABLE IF NOT EXISTS resolution_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    label_json TEXT NOT NULL,
                    source TEXT,
                    native_label_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    cycle INTEGER,
                    run_id TEXT
                );

                CREATE INDEX IF NOT EXISTS
                    idx_resolution_object_id
                ON resolution_events(object_id);
                """
            )

            connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection


def _normalize_nonempty_string(
    value: object,
    name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{name} must not be empty.")

    return normalized


def _normalize_optional_string(
    value: object | None,
    name: str,
) -> str | None:
    if value is None:
        return None

    return _normalize_nonempty_string(value, name)


def _normalize_cycle(
    cycle: int | None,
) -> int | None:
    if cycle is None:
        return None

    if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 0:
        raise ValueError("cycle must be a non-negative integer or None.")

    return cycle


def _normalize_timestamp(
    value: datetime | None,
) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)

    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime or None.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware.")

    return value.astimezone(timezone.utc)


def _timestamp_to_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _timestamp_from_text(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return _normalize_timestamp(parsed)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return _timestamp_to_text(_normalize_timestamp(value))

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _json_dumps(value: object) -> str:
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_loads(value: str) -> object:
    return json.loads(value)
