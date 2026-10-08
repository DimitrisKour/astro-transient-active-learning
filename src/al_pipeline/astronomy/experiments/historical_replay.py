"""Deterministic historical replay splits for AstroMiNN active learning.

Experiment B treats the original historical training split as the only
source of labels.  A small object-stratified subset becomes the initially
labelled set ``L0`` and every remaining training object becomes the hidden
candidate pool ``U0``.  The published validation and test splits are never
resampled or modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest, CandidatePoolRecord
from al_pipeline.astronomy.data.registry import DatasetSpec
from al_pipeline.astronomy.astrominn.training import LABEL_TO_INDEX, MODEL_CLASS_NAMES


@dataclass(frozen=True, slots=True)
class HistoricalReplaySplit:
    """Materialized, immutable inputs for one historical replay design."""

    split_dir: Path
    initial_manifest: Path
    candidate_pool_dir: Path
    hidden_labels: Path
    summary_path: Path
    initial_object_ids: tuple[str, ...]
    candidate_object_ids: tuple[str, ...]


def materialize_astrominn_historical_replay_split(
    *,
    dataset: DatasetSpec,
    destination: str | Path,
    initial_fraction: float = 0.10,
    minimum_initial_per_class: int = 5,
    split_seed: int = 135,
    pool_id: str = "astrominn_historical_train_replay_v1",
    excluded_object_manifests: Sequence[str | Path] = (),
) -> HistoricalReplaySplit:
    """Create a deterministic balanced ``L0`` and label-hidden ``U0``.

    All alert-stage rows belonging to an object stay together.  ``L0`` gets
    approximately ``initial_fraction`` of every AstroMiNN target class, with
    a small minimum protecting rare-class support; ``U0`` receives every
    other object. The split seed is a data-design choice and should remain
    fixed across model-training seeds.
    """

    if not 0 < initial_fraction < 1:
        raise ValueError("initial_fraction must be between zero and one.")
    if minimum_initial_per_class <= 0:
        raise ValueError("minimum_initial_per_class must be positive.")
    if split_seed < 0:
        raise ValueError("split_seed must be non-negative.")
    if not pool_id:
        raise ValueError("pool_id must be non-empty.")
    if dataset.train_manifest is None:
        raise ValueError("DatasetSpec requires train_manifest.")
    if dataset.validation_manifest is None:
        raise ValueError("DatasetSpec requires validation_manifest.")
    if dataset.test_manifest is None:
        raise ValueError("DatasetSpec requires test_manifest.")
    if dataset.object_id_column is None:
        raise ValueError("DatasetSpec requires object_id_column.")
    if dataset.sample_file_column is None:
        raise ValueError("DatasetSpec requires sample_file_column.")
    if dataset.label_column is None:
        raise ValueError("DatasetSpec requires label_column.")

    fieldnames, rows = _read_csv(dataset.train_manifest)
    maximum_alert = int(dataset.options.get("max_alert", 10))
    required = {
        dataset.object_id_column,
        dataset.sample_file_column,
        dataset.label_column,
        "alert",
    }
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"Historical train manifest is missing fields required for replay: {sorted(missing)}."
        )

    rows_by_object: dict[str, list[dict[str, str]]] = defaultdict(list)
    canonical_label_by_object: dict[str, str] = {}
    native_labels_by_object: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        object_id = row[dataset.object_id_column].strip()
        native_label = row[dataset.label_column].strip()
        if not object_id or not native_label:
            raise ValueError("Historical rows require non-empty object IDs and labels.")
        try:
            canonical_label = MODEL_CLASS_NAMES[LABEL_TO_INDEX[native_label]]
        except KeyError as exc:
            raise ValueError(f"Unsupported historical AstroMiNN label {native_label!r}.") from exc

        previous = canonical_label_by_object.setdefault(
            object_id,
            canonical_label,
        )
        if previous != canonical_label:
            raise ValueError(
                f"Object {object_id!r} maps to conflicting target labels: "
                f"{previous!r} and {canonical_label!r}."
            )
        native_labels_by_object[object_id].add(native_label)
        rows_by_object[object_id].append(row)

        try:
            alert = int(row["alert"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Historical alert is not an integer for {object_id!r}.") from exc
        if not 0 <= alert <= maximum_alert:
            raise ValueError(
                f"Historical alert {alert} for {object_id!r} lies outside "
                f"the configured 0..{maximum_alert} boundary."
            )

    objects_by_class: dict[str, list[str]] = {class_name: [] for class_name in MODEL_CLASS_NAMES}
    for object_id, class_name in canonical_label_by_object.items():
        objects_by_class[class_name].append(object_id)

    initial_ids: set[str] = set()
    for class_index, class_name in enumerate(MODEL_CLASS_NAMES):
        object_ids = sorted(objects_by_class[class_name])
        initial_count = max(
            minimum_initial_per_class,
            math.ceil(len(object_ids) * initial_fraction),
        )
        if len(object_ids) <= initial_count:
            raise ValueError(
                f"Class {class_name!r} has {len(object_ids)} objects; "
                f"cannot place {initial_count} in L0 and retain at "
                "least one object in U0."
            )
        generator = np.random.default_rng(np.random.SeedSequence([split_seed, class_index]))
        chosen = generator.choice(
            np.asarray(object_ids, dtype=object),
            size=initial_count,
            replace=False,
        )
        initial_ids.update(str(value) for value in chosen)

    object_order = tuple(rows_by_object)
    initial_object_ids = tuple(object_id for object_id in object_order if object_id in initial_ids)
    candidate_object_ids = tuple(
        object_id for object_id in object_order if object_id not in initial_ids
    )

    _validate_split_disjointness(
        dataset=dataset,
        initial_ids=set(initial_object_ids),
        candidate_ids=set(candidate_object_ids),
        historical_ids=set(object_order),
    )
    excluded_manifest_audits = _audit_excluded_manifests(
        manifests=excluded_object_manifests,
        dataset=dataset,
        historical_train_ids=set(object_order),
    )

    split_dir = Path(destination)
    initial_manifest = split_dir / "initial_labeled_manifest.csv"
    candidate_pool_dir = split_dir / "candidate_pool"
    candidate_manifest_path = candidate_pool_dir / "candidate_pool.csv"
    ingestion_report_path = candidate_pool_dir / "ingestion_report.json"
    hidden_labels_path = split_dir / "hidden_labels.csv"
    summary_path = split_dir / "split_summary.json"

    initial_rows = [row for row in rows if row[dataset.object_id_column] in initial_ids]
    _write_csv_idempotent(
        initial_manifest,
        fieldnames=fieldnames,
        rows=initial_rows,
    )

    candidate_records: list[CandidatePoolRecord] = []
    for row_index, row in enumerate(rows):
        object_id = row[dataset.object_id_column]
        if object_id in initial_ids:
            continue
        candidate_records.append(
            CandidatePoolRecord(
                survey=dataset.survey,
                object_id=object_id,
                alert=int(row["alert"]),
                filepath=str(
                    _resolve_sample_path(
                        row[dataset.sample_file_column],
                        data_root=dataset.data_root,
                    )
                ),
                source_reference=(
                    f"historical_train:{dataset.train_manifest.name}:row_{row_index}"
                ),
            )
        )

    candidate_manifest = CandidatePoolManifest(
        pool_id=pool_id,
        records=candidate_records,
    )
    _write_candidate_manifest_idempotent(
        candidate_manifest,
        candidate_manifest_path,
    )
    (candidate_pool_dir / "samples").mkdir(parents=True, exist_ok=True)
    _write_json_idempotent(
        ingestion_report_path,
        {
            "schema_version": 1,
            "mode": "historical_manifest_references",
            "source_manifest": str(dataset.train_manifest),
            "pool_id": pool_id,
            "objects": len(candidate_object_ids),
            "samples": len(candidate_records),
            "copied_samples": 0,
            "maximum_alert": maximum_alert,
        },
    )

    hidden_rows = [
        {
            "object_id": object_id,
            "target_label": canonical_label_by_object[object_id],
            "native_labels": "|".join(sorted(native_labels_by_object[object_id])),
        }
        for object_id in candidate_object_ids
    ]
    _write_csv_idempotent(
        hidden_labels_path,
        fieldnames=["object_id", "target_label", "native_labels"],
        rows=hidden_rows,
    )

    initial_sample_counts = Counter(
        canonical_label_by_object[row[dataset.object_id_column]] for row in initial_rows
    )
    candidate_sample_counts = Counter(
        canonical_label_by_object[record.object_id] for record in candidate_records
    )
    summary = {
        "schema_version": 1,
        "design": "historical_train_object_stratified_replay",
        "pool_id": pool_id,
        "split_seed": split_seed,
        "initial_fraction": initial_fraction,
        "minimum_initial_per_class": minimum_initial_per_class,
        "historical_train_manifest": str(dataset.train_manifest),
        "validation_manifest": str(dataset.validation_manifest),
        "test_manifest": str(dataset.test_manifest),
        "initial_manifest": str(initial_manifest),
        "candidate_manifest": str(candidate_manifest_path),
        "ingestion_report": str(ingestion_report_path),
        "hidden_labels": str(hidden_labels_path),
        "historical_objects": len(object_order),
        "historical_samples": len(rows),
        "initial_objects": len(initial_object_ids),
        "initial_samples": len(initial_rows),
        "candidate_objects": len(candidate_object_ids),
        "candidate_samples": len(candidate_records),
        "maximum_alert": maximum_alert,
        "historical_object_label_counts": _counts_by_label(canonical_label_by_object.values()),
        "initial_object_label_counts": _counts_by_label(
            canonical_label_by_object[object_id] for object_id in initial_object_ids
        ),
        "candidate_object_label_counts": _counts_by_label(
            canonical_label_by_object[object_id] for object_id in candidate_object_ids
        ),
        "initial_sample_label_counts": _complete_counts(initial_sample_counts),
        "candidate_sample_label_counts": _complete_counts(candidate_sample_counts),
        "overlap_checks": {
            "initial_candidate": 0,
            "historical_validation": 0,
            "historical_test": 0,
            "validation_test": 0,
        },
        "excluded_manifest_audits": excluded_manifest_audits,
        "sha256": {
            "historical_train_manifest": _sha256_file(dataset.train_manifest),
            "validation_manifest": _sha256_file(dataset.validation_manifest),
            "test_manifest": _sha256_file(dataset.test_manifest),
            "initial_manifest": _sha256_file(initial_manifest),
            "candidate_manifest": _sha256_file(candidate_manifest_path),
            "ingestion_report": _sha256_file(ingestion_report_path),
            "hidden_labels": _sha256_file(hidden_labels_path),
        },
    }
    _write_json_idempotent(summary_path, summary)

    return HistoricalReplaySplit(
        split_dir=split_dir,
        initial_manifest=initial_manifest,
        candidate_pool_dir=candidate_pool_dir,
        hidden_labels=hidden_labels_path,
        summary_path=summary_path,
        initial_object_ids=initial_object_ids,
        candidate_object_ids=candidate_object_ids,
    )


def _validate_split_disjointness(
    *,
    dataset: DatasetSpec,
    initial_ids: set[str],
    candidate_ids: set[str],
    historical_ids: set[str],
) -> None:
    if initial_ids.intersection(candidate_ids):
        raise ValueError("L0 and U0 overlap.")
    if initial_ids.union(candidate_ids) != historical_ids:
        raise ValueError("L0 and U0 do not partition the historical train set.")

    validation_ids = _manifest_object_ids(
        dataset.validation_manifest,
        dataset.object_id_column,
    )
    test_ids = _manifest_object_ids(
        dataset.test_manifest,
        dataset.object_id_column,
    )
    overlaps = {
        "historical/validation": historical_ids.intersection(validation_ids),
        "historical/test": historical_ids.intersection(test_ids),
        "validation/test": validation_ids.intersection(test_ids),
    }
    nonempty = {name: sorted(values)[:10] for name, values in overlaps.items() if values}
    if nonempty:
        raise ValueError(f"Historical split leakage detected: {nonempty}.")


def _manifest_object_ids(path: Path, object_id_column: str) -> set[str]:
    fieldnames, rows = _read_csv(path)
    if object_id_column not in fieldnames:
        raise ValueError(f"Manifest {path} has no object column {object_id_column!r}.")
    return {row[object_id_column] for row in rows}


def _audit_excluded_manifests(
    *,
    manifests: Sequence[str | Path],
    dataset: DatasetSpec,
    historical_train_ids: set[str],
) -> list[dict[str, object]]:
    historical_ids = set(historical_train_ids)
    historical_ids.update(
        _manifest_object_ids(
            dataset.validation_manifest,
            dataset.object_id_column,
        )
    )
    historical_ids.update(
        _manifest_object_ids(
            dataset.test_manifest,
            dataset.object_id_column,
        )
    )

    audits: list[dict[str, object]] = []
    for value in manifests:
        path = Path(value)
        fieldnames, rows = _read_csv(path)
        column = next(
            (
                candidate
                for candidate in (
                    dataset.object_id_column,
                    "object_id",
                    "obj_id",
                )
                if candidate is not None and candidate in fieldnames
            ),
            None,
        )
        if column is None:
            raise ValueError(f"Cannot find an object-ID column in {path}: {fieldnames}.")
        external_ids = {str(row[column]).strip() for row in rows if str(row[column]).strip()}
        overlap = sorted(historical_ids.intersection(external_ids))
        if overlap:
            raise ValueError(
                "Excluded external manifest overlaps historical data: "
                f"{path}; examples={overlap[:10]}."
            )
        audits.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "objects": len(external_ids),
                "historical_overlap": 0,
                "usage": "overlap_audit_only",
            }
        )
    return audits


def _resolve_sample_path(value: object, *, data_root: Path | None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if data_root is None:
        raise ValueError(
            f"Cannot resolve relative historical sample path {path}: DatasetSpec has no data_root."
        )
    return data_root / path


def _counts_by_label(labels: Iterable[object]) -> dict[str, int]:
    return _complete_counts(Counter(str(label) for label in labels))


def _complete_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {class_name: int(counts.get(class_name, 0)) for class_name in MODEL_CLASS_NAMES}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if not fieldnames or not rows:
        raise ValueError(f"Manifest is empty or has no header: {path}.")
    return fieldnames, rows


def _write_csv_idempotent(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _accept_identical_or_install(path, temporary)


def _write_candidate_manifest_idempotent(
    manifest: CandidatePoolManifest,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    manifest.to_csv(temporary)
    _accept_identical_or_install(path, temporary)


def _write_json_idempotent(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _accept_identical_or_install(path, temporary)


def _accept_identical_or_install(path: Path, temporary: Path) -> None:
    if path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f"Existing replay artifact differs from requested design: {path}.")
        temporary.unlink()
        return
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "HistoricalReplaySplit",
    "materialize_astrominn_historical_replay_split",
]
