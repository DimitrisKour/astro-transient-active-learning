"""Management of cumulative labeled sets for adaptive-learning runs.

Historical training data L0 remain immutable. Accepted acquisitions are
recorded per cycle and combined into reproducible cumulative training
manifests:

    L_t = L0 ∪ A1 ∪ A2 ∪ ... ∪ At

The object-level acquisition ledger is kept separate from model-ready
training rows because one acquired object may correspond to multiple
samples, alerts, spectra, or other rows in a model-specific manifest.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from al_pipeline.astronomy.data.registry import DatasetSpec


_LEDGER_FIELDS = (
    "cycle",
    "object_id",
    "survey",
    "label",
    "strategy",
    "data_reference",
)


@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    """One accepted object acquired during an AL cycle."""

    object_id: str
    survey: str
    label: str
    strategy: str | None = None
    data_reference: str | None = None

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be a non-empty string.")

        if not self.survey:
            raise ValueError("survey must be a non-empty string.")

        if not self.label:
            raise ValueError("label must be a non-empty string.")


class LabeledSetManager:
    """Manage historical and newly acquired labeled training data."""

    def __init__(
        self,
        *,
        dataset: DatasetSpec,
        run_dir: str | Path,
    ) -> None:
        self.dataset = dataset
        self.run_dir = Path(run_dir)

        if dataset.train_manifest is None:
            raise ValueError(f"Dataset {dataset.name!r} has no historical training manifest.")

        if dataset.object_id_column is None:
            raise ValueError(f"Dataset {dataset.name!r} has no object_id_column.")

        if dataset.label_column is None:
            raise ValueError(f"Dataset {dataset.name!r} has no label_column.")

        self.base_manifest = dataset.train_manifest
        self.manifests_dir = self.run_dir / "manifests"
        self.ledger_path = self.run_dir / "acquisitions.csv"

    @property
    def baseline_manifest(self) -> Path:
        """Return the cycle-zero snapshot representing L0."""
        return self.manifests_dir / "train_cycle_000.csv"

    def acquired_manifest(
        self,
        cycle: int,
    ) -> Path:
        """Return the model-ready manifest path for one acquisition cycle."""
        self._validate_cycle(cycle, allow_zero=False)

        return self.manifests_dir / f"acquired_cycle_{cycle:03d}.csv"

    def training_manifest(
        self,
        cycle: int,
    ) -> Path:
        """Return the cumulative training-manifest path for one cycle."""
        self._validate_cycle(cycle, allow_zero=True)

        return self.manifests_dir / f"train_cycle_{cycle:03d}.csv"

    def initialize(self) -> None:
        """Initialize run-owned labeled-data state.

        This creates a small snapshot of the historical training manifest
        for cycle zero but never modifies the original historical file.
        """
        if not self.base_manifest.exists():
            raise FileNotFoundError(
                f"Historical training manifest does not exist: {self.base_manifest}"
            )

        self.manifests_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames, rows = _read_csv(self.base_manifest)

        self._validate_training_schema(fieldnames)

        if not self.baseline_manifest.exists():
            _write_csv_atomic(
                self.baseline_manifest,
                fieldnames,
                rows,
            )

        if not self.ledger_path.exists():
            _write_csv_atomic(
                self.ledger_path,
                list(_LEDGER_FIELDS),
                [],
            )

    def record_cycle(
        self,
        cycle: int,
        *,
        acquisitions: Iterable[AcquisitionRecord],
        training_rows: Iterable[Mapping[str, object]],
    ) -> Path:
        """Record accepted acquisitions and materialize cumulative L_t.

        ``acquisitions`` contains one record per accepted object.

        ``training_rows`` contains the model-ready rows associated with
        those objects. Several training rows may belong to one acquired
        object.

        Returns the cumulative ``train_cycle_XXX.csv`` path.
        """
        self._validate_cycle(cycle, allow_zero=False)
        self.initialize()

        acquired_path = self.acquired_manifest(cycle)

        if acquired_path.exists():
            raise FileExistsError(f"Acquisition cycle {cycle} is already recorded: {acquired_path}")

        if cycle > 1:
            previous = self.acquired_manifest(cycle - 1)

            if not previous.exists():
                raise ValueError(f"Cannot record cycle {cycle} before cycle {cycle - 1}.")

        acquisition_records = tuple(acquisitions)

        self._validate_new_acquisitions(acquisition_records)

        base_fields, _ = _read_csv(self.base_manifest)

        rows = [
            _normalize_training_row(
                row,
                base_fields,
            )
            for row in training_rows
        ]

        self._validate_training_rows(
            rows,
            acquisition_records,
        )

        _write_csv_atomic(
            acquired_path,
            base_fields,
            rows,
        )

        self._append_to_ledger(
            cycle,
            acquisition_records,
        )

        # TODO: If multi-process experiment orchestration is introduced,
        # coordinate this multi-file update with RunState using a run-level
        # lock or transactional recovery mechanism.
        return self.materialize_cycle(cycle)

    def ensure_cycle(
        self,
        cycle: int,
        *,
        acquisitions: Iterable[AcquisitionRecord],
        training_rows: Iterable[Mapping[str, object]],
    ) -> Path:
        """Record a cycle or validate and recover an interrupted record.

        Adaptive runs may be restarted after the per-cycle manifest was
        written but before its ledger entry or cumulative manifest was
        materialized. This method makes that operation idempotent while
        refusing to reuse a cycle whose objects, labels, or rows differ.
        """

        self._validate_cycle(cycle, allow_zero=False)
        self.initialize()

        acquisition_records = tuple(acquisitions)
        base_fields, _ = _read_csv(self.base_manifest)
        normalized_rows = [_normalize_training_row(row, base_fields) for row in training_rows]
        self._validate_training_rows(
            normalized_rows,
            acquisition_records,
        )

        acquired_path = self.acquired_manifest(cycle)

        if not acquired_path.exists():
            return self.record_cycle(
                cycle,
                acquisitions=acquisition_records,
                training_rows=normalized_rows,
            )

        existing_fields, existing_rows = _read_csv(acquired_path)

        if existing_fields != base_fields or existing_rows != normalized_rows:
            raise ValueError(
                f"Existing acquisition manifest for cycle {cycle} does "
                "not match the requested training rows."
            )

        existing_cycle_records = self.get_cycle(cycle)

        if existing_cycle_records:
            if existing_cycle_records != acquisition_records:
                raise ValueError(
                    f"Existing acquisition ledger for cycle {cycle} does "
                    "not match the requested acquisitions."
                )
        else:
            self._validate_new_acquisitions(acquisition_records)
            self._append_to_ledger(cycle, acquisition_records)

        return self.materialize_cycle(cycle)

    def get_cycle(
        self,
        cycle: int,
    ) -> tuple[AcquisitionRecord, ...]:
        """Return accepted objects acquired in exactly one cycle."""
        self._validate_cycle(cycle, allow_zero=False)
        self.initialize()

        rows = self._read_ledger()

        return tuple(_record_from_ledger_row(row) for row in rows if int(row["cycle"]) == cycle)

    def get_acquired_through(
        self,
        cycle: int,
    ) -> tuple[AcquisitionRecord, ...]:
        """Return all accepted objects acquired up to a cycle."""
        self._validate_cycle(cycle, allow_zero=True)
        self.initialize()

        rows = self._read_ledger()

        return tuple(_record_from_ledger_row(row) for row in rows if int(row["cycle"]) <= cycle)

    def get_training_rows_for_cycle(
        self,
        cycle: int,
    ) -> tuple[dict[str, str], ...]:
        """Return model-ready rows introduced during exactly one cycle."""
        path = self.acquired_manifest(cycle)

        if not path.exists():
            raise FileNotFoundError(f"Acquisition manifest does not exist: {path}")

        _, rows = _read_csv(path)

        return tuple(rows)

    def get_labeled_through(
        self,
        cycle: int,
    ) -> tuple[dict[str, str], ...]:
        """Return all model-ready rows in L_t.

        For actual training, ``materialize_cycle`` is preferable because
        model loaders can consume the resulting manifest directly without
        requiring all rows to remain in Python memory.
        """
        path = self.materialize_cycle(cycle)

        _, rows = _read_csv(path)

        return tuple(rows)

    def object_ids_through(
        self,
        cycle: int,
    ) -> frozenset[str]:
        """Return unique object IDs represented in cumulative ``L_t``."""

        rows = self.get_labeled_through(cycle)

        return frozenset(row[self.dataset.object_id_column] for row in rows)

    def materialize_cycle(
        self,
        cycle: int,
    ) -> Path:
        """Create or return the exact cumulative training manifest L_t."""
        self._validate_cycle(cycle, allow_zero=True)
        self.initialize()

        output_path = self.training_manifest(cycle)

        if output_path.exists():
            return output_path

        fieldnames, base_rows = _read_csv(self.base_manifest)

        all_rows = list(base_rows)

        for acquisition_cycle in range(
            1,
            cycle + 1,
        ):
            path = self.acquired_manifest(acquisition_cycle)

            if not path.exists():
                raise FileNotFoundError(
                    f"Cannot materialize cycle {cycle}: "
                    f"missing acquisition manifest for "
                    f"cycle {acquisition_cycle}: {path}"
                )

            acquired_fields, acquired_rows = _read_csv(path)

            if acquired_fields != fieldnames:
                raise ValueError(
                    f"Acquisition manifest for cycle "
                    f"{acquisition_cycle} does not match "
                    "the historical training schema."
                )

            all_rows.extend(acquired_rows)

        _write_csv_atomic(
            output_path,
            fieldnames,
            all_rows,
        )

        return output_path

    def _validate_training_schema(
        self,
        fieldnames: list[str],
    ) -> None:
        """Verify fields needed for acquisition bookkeeping exist."""
        required = {
            self.dataset.object_id_column,
            self.dataset.label_column,
        }

        missing = required.difference(fieldnames)

        if missing:
            raise ValueError(
                f"Historical training manifest is missing required field(s): {sorted(missing)}."
            )

    def _validate_new_acquisitions(
        self,
        records: tuple[AcquisitionRecord, ...],
    ) -> None:
        """Prevent duplicate acquisition of an already labeled object."""
        object_ids = [record.object_id for record in records]

        if len(object_ids) != len(set(object_ids)):
            raise ValueError("An acquisition cycle cannot contain duplicate object IDs.")

        existing = {row["object_id"] for row in self._read_ledger()}

        duplicates = existing.intersection(object_ids)

        if duplicates:
            raise ValueError(f"Objects have already been acquired: {sorted(duplicates)}.")

    def _validate_training_rows(
        self,
        rows: list[dict[str, str]],
        acquisitions: tuple[AcquisitionRecord, ...],
    ) -> None:
        """Ensure training rows belong to the accepted objects."""
        acquired_ids = {record.object_id for record in acquisitions}

        row_ids = {row[self.dataset.object_id_column] for row in rows}

        unknown = row_ids.difference(acquired_ids)

        if unknown:
            raise ValueError(
                "Training rows contain objects that were "
                f"not accepted in this cycle: {sorted(unknown)}."
            )

        missing = acquired_ids.difference(row_ids)

        if missing:
            raise ValueError(
                f"Accepted objects have no model-ready training rows: {sorted(missing)}."
            )

    def _append_to_ledger(
        self,
        cycle: int,
        records: tuple[AcquisitionRecord, ...],
    ) -> None:
        """Append accepted object-level records to the run ledger."""
        existing_rows = self._read_ledger()

        new_rows = [
            {
                "cycle": str(cycle),
                "object_id": record.object_id,
                "survey": record.survey,
                "label": record.label,
                "strategy": record.strategy or "",
                "data_reference": (record.data_reference or ""),
            }
            for record in records
        ]

        _write_csv_atomic(
            self.ledger_path,
            list(_LEDGER_FIELDS),
            [
                *existing_rows,
                *new_rows,
            ],
        )

    def _read_ledger(
        self,
    ) -> list[dict[str, str]]:
        """Read the acquisition ledger."""
        if not self.ledger_path.exists():
            return []

        fieldnames, rows = _read_csv(self.ledger_path)

        if tuple(fieldnames) != _LEDGER_FIELDS:
            raise ValueError("Acquisition ledger has an unexpected schema.")

        return rows

    @staticmethod
    def _validate_cycle(
        cycle: int,
        *,
        allow_zero: bool,
    ) -> None:
        """Validate an AL cycle number."""
        minimum = 0 if allow_zero else 1

        if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < minimum:
            relation = "non-negative" if allow_zero else "positive"

            raise ValueError(f"cycle must be a {relation} integer.")


def _normalize_training_row(
    row: Mapping[str, object],
    fieldnames: list[str],
) -> dict[str, str]:
    """Convert one model-ready row to the historical manifest schema."""
    missing = [field for field in fieldnames if field not in row]

    if missing:
        raise ValueError(f"Training row is missing required field(s): {missing}.")

    return {field: str(row[field]) for field in fieldnames}


def _record_from_ledger_row(
    row: Mapping[str, str],
) -> AcquisitionRecord:
    """Convert a ledger row back to an acquisition record."""
    return AcquisitionRecord(
        object_id=row["object_id"],
        survey=row["survey"],
        label=row["label"],
        strategy=(row["strategy"] or None),
        data_reference=(row["data_reference"] or None),
    )


def _read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV manifest."""
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        fieldnames = list(reader.fieldnames)

        rows = [dict(row) for row in reader]

    return fieldnames, rows


def _write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write a CSV file atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    temporary_path.replace(path)
