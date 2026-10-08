"""Physical candidate-pool manifests for adaptive learning.

A candidate-pool manifest describes which survey samples physically belong
to a pool. It is intentionally separate from adaptive.pool.Pool, which owns
the logical active-learning states (unlabeled, pending, labeled).
"""

from __future__ import annotations

import csv

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_COLUMNS = (
    "pool_id",
    "survey",
    "object_id",
    "alert",
    "filepath",
    "source_reference",
)

FORBIDDEN_LABEL_COLUMNS = {
    "label",
    "target",
    "class",
    "classification",
}


@dataclass(frozen=True)
class CandidatePoolRecord:
    """One physical sample belonging to an unlabeled candidate pool."""

    survey: str
    object_id: str
    alert: int
    filepath: str
    source_reference: str = ""

    @property
    def qualified_object_id(self) -> str:
        """Survey-qualified ID suitable for globally unique bookkeeping."""

        return f"{self.survey}:{self.object_id}"


class CandidatePoolManifest:
    """A versioned collection of physical candidate samples."""

    def __init__(
        self,
        *,
        pool_id: str,
        records: Iterable[CandidatePoolRecord],
    ) -> None:
        pool_id = str(pool_id).strip()

        if not pool_id:
            raise ValueError("pool_id must be non-empty.")

        self.pool_id = pool_id
        self.records = tuple(records)

        self._validate_records()

    def _validate_records(self) -> None:
        seen: set[tuple[str, str, int]] = set()

        for record in self.records:
            if not record.survey.strip():
                raise ValueError("Candidate survey must be non-empty.")

            if not record.object_id.strip():
                raise ValueError("Candidate object_id must be non-empty.")

            if record.alert < 0:
                raise ValueError("Candidate alert must be >= 0.")

            if not record.filepath.strip():
                raise ValueError("Candidate filepath must be non-empty.")

            key = (
                record.survey,
                record.object_id,
                record.alert,
            )

            if key in seen:
                raise ValueError(
                    "Duplicate candidate sample for "
                    f"{record.survey}:{record.object_id} "
                    f"alert {record.alert}."
                )

            seen.add(key)

    @property
    def object_ids(self) -> tuple[str, ...]:
        """Raw object IDs in first-seen order."""

        return tuple(dict.fromkeys(record.object_id for record in self.records))

    @property
    def qualified_object_ids(
        self,
    ) -> tuple[str, ...]:
        """Survey-qualified object IDs in first-seen order."""

        return tuple(dict.fromkeys(record.qualified_object_id for record in self.records))

    def records_for_object(
        self,
        object_id: str,
        *,
        survey: str | None = None,
    ) -> tuple[CandidatePoolRecord, ...]:
        """Return all alert rows associated with one object."""

        return tuple(
            record
            for record in self.records
            if record.object_id == object_id and (survey is None or record.survey == survey)
        )

    def missing_files(self) -> tuple[str, ...]:
        """Return referenced files that do not currently exist.

        File existence is checked explicitly rather than when parsing the
        manifest so manifests remain portable between storage locations.
        """

        return tuple(
            record.filepath for record in self.records if not Path(record.filepath).exists()
        )

    def to_csv(
        self,
        path: str | Path,
    ) -> Path:
        """Write the manifest to CSV."""

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_suffix(path.suffix + ".tmp")

        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=MANIFEST_COLUMNS,
            )
            writer.writeheader()

            for record in self.records:
                writer.writerow(
                    {
                        "pool_id": self.pool_id,
                        "survey": record.survey,
                        "object_id": record.object_id,
                        "alert": record.alert,
                        "filepath": record.filepath,
                        "source_reference": (record.source_reference),
                    }
                )

        temporary_path.replace(path)

        return path

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
    ) -> "CandidatePoolManifest":
        """Load and validate a candidate-pool manifest."""

        path = Path(path)

        with path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle)

            fieldnames = set(reader.fieldnames or ())

            forbidden = fieldnames & FORBIDDEN_LABEL_COLUMNS

            if forbidden:
                raise ValueError(
                    "Candidate pool manifests must not "
                    "contain label columns: " + ", ".join(sorted(forbidden))
                )

            required = set(MANIFEST_COLUMNS) - {"source_reference"}

            missing = required - fieldnames

            if missing:
                raise ValueError(
                    "Candidate pool manifest is missing "
                    "required columns: " + ", ".join(sorted(missing))
                )

            rows = list(reader)

        pool_ids = {row["pool_id"].strip() for row in rows}

        if not rows:
            raise ValueError("Candidate pool manifest is empty.")

        if len(pool_ids) != 1:
            raise ValueError("A candidate pool manifest must contain exactly one pool_id.")

        pool_id = next(iter(pool_ids))

        records = []

        for row in rows:
            try:
                alert = int(row["alert"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Candidate alert values must be integers.") from exc

            records.append(
                CandidatePoolRecord(
                    survey=row["survey"],
                    object_id=row["object_id"],
                    alert=alert,
                    filepath=row["filepath"],
                    source_reference=row.get(
                        "source_reference",
                        "",
                    ),
                )
            )

        return cls(
            pool_id=pool_id,
            records=records,
        )


# TODO: Add immutable provenance metadata (for example query window or
# source configuration) once the first real external source is integrated
# and we know which provenance fields are actually required.
