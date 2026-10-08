"""Tests for deterministic historical active-learning replay manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
from al_pipeline.astronomy.data.registry import DatasetSpec
from al_pipeline.astronomy.experiments.historical_replay import (
    materialize_astrominn_historical_replay_split,
)


NATIVE_LABELS = (
    "AGN",
    "Tidal Disruption Event",
    "SN Ia",
    "SN II",
    "Cataclysmic",
)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["obj_id", "filepath", "target", "alert"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _dataset(
    tmp_path: Path,
    objects_per_class: tuple[int, ...] = (3, 3, 3, 3, 3),
) -> DatasetSpec:
    train_rows = []
    for class_index, label in enumerate(NATIVE_LABELS):
        for object_index in range(objects_per_class[class_index]):
            object_id = f"train_{class_index}_{object_index}"
            for alert in range(2):
                train_rows.append(
                    {
                        "obj_id": object_id,
                        "filepath": f"{object_id}_{alert}.npy",
                        "target": label,
                        "alert": alert,
                    }
                )

    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    test = tmp_path / "test.csv"
    _write_manifest(train, train_rows)
    _write_manifest(
        validation,
        [
            {
                "obj_id": "validation_only",
                "filepath": "validation.npy",
                "target": "SN Ia",
                "alert": 0,
            }
        ],
    )
    _write_manifest(
        test,
        [
            {
                "obj_id": "test_only",
                "filepath": "test.npy",
                "target": "SN II",
                "alert": 0,
            }
        ],
    )
    return DatasetSpec(
        name="astrominn-toy",
        survey="ZTF",
        modality="image_metadata",
        selection_unit="object",
        train_manifest=train,
        validation_manifest=validation,
        test_manifest=test,
        data_root=tmp_path,
        object_id_column="obj_id",
        sample_file_column="filepath",
        label_column="target",
    )


def test_historical_replay_is_stratified_disjoint_and_idempotent(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    destination = tmp_path / "replay"
    first = materialize_astrominn_historical_replay_split(
        dataset=dataset,
        destination=destination,
        initial_fraction=1 / 3,
        minimum_initial_per_class=1,
        split_seed=135,
        pool_id="toy-replay",
    )
    second = materialize_astrominn_historical_replay_split(
        dataset=dataset,
        destination=destination,
        initial_fraction=1 / 3,
        minimum_initial_per_class=1,
        split_seed=135,
        pool_id="toy-replay",
    )

    assert first == second
    assert len(first.initial_object_ids) == 5
    assert len(first.candidate_object_ids) == 10
    assert not set(first.initial_object_ids).intersection(first.candidate_object_ids)

    pool = CandidatePoolManifest.from_csv(first.candidate_pool_dir / "candidate_pool.csv")
    assert len(pool.object_ids) == 10
    assert len(pool.records) == 20
    assert all(record.alert <= 10 for record in pool.records)
    assert (first.candidate_pool_dir / "samples").is_dir()
    assert (first.candidate_pool_dir / "ingestion_report.json").is_file()

    with (first.candidate_pool_dir / "candidate_pool.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        candidate_header = set(next(csv.reader(handle)))
    assert not candidate_header.intersection({"label", "target", "class", "classification"})

    with first.initial_manifest.open(newline="", encoding="utf-8") as handle:
        initial_rows = list(csv.DictReader(handle))
    assert len(initial_rows) == 10

    with first.hidden_labels.open(newline="", encoding="utf-8") as handle:
        hidden_rows = list(csv.DictReader(handle))
    assert {row["target_label"] for row in hidden_rows} == {
        "AGN",
        "TDE",
        "SN I",
        "SN II",
        "CV",
    }

    summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert set(summary["initial_object_label_counts"].values()) == {1}
    assert set(summary["candidate_object_label_counts"].values()) == {2}
    assert all(value == 0 for value in summary["overlap_checks"].values())


def test_historical_replay_rejects_changed_existing_design(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    destination = tmp_path / "replay"
    materialize_astrominn_historical_replay_split(
        dataset=dataset,
        destination=destination,
        initial_fraction=1 / 3,
        minimum_initial_per_class=1,
        split_seed=135,
    )

    with pytest.raises(ValueError, match="differs from requested design"):
        materialize_astrominn_historical_replay_split(
            dataset=dataset,
            destination=destination,
            initial_fraction=2 / 3,
            minimum_initial_per_class=1,
            split_seed=135,
        )


def test_historical_replay_preserves_class_proportions_with_rare_floor(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, (10, 20, 30, 40, 50))
    replay = materialize_astrominn_historical_replay_split(
        dataset=dataset,
        destination=tmp_path / "replay",
        initial_fraction=0.10,
        minimum_initial_per_class=2,
    )
    summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))

    assert summary["initial_object_label_counts"] == {
        "AGN": 2,
        "TDE": 2,
        "SN I": 3,
        "SN II": 4,
        "CV": 5,
    }
    assert summary["candidate_object_label_counts"] == {
        "AGN": 8,
        "TDE": 18,
        "SN I": 27,
        "SN II": 36,
        "CV": 45,
    }


def test_historical_replay_rejects_split_leakage(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    _write_manifest(
        dataset.validation_manifest,
        [
            {
                "obj_id": "train_0_0",
                "filepath": "leaked.npy",
                "target": "AGN",
                "alert": 0,
            }
        ],
    )

    with pytest.raises(ValueError, match="split leakage"):
        materialize_astrominn_historical_replay_split(
            dataset=dataset,
            destination=tmp_path / "replay",
            initial_fraction=1 / 3,
            minimum_initial_per_class=1,
        )


def test_historical_replay_audits_external_labelled_manifest(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    external = tmp_path / "experiment_a_hidden_labels.csv"
    with external.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["object_id", "target_label"],
        )
        writer.writeheader()
        writer.writerow({"object_id": "new_post_2025_object", "target_label": "SN I"})

    replay = materialize_astrominn_historical_replay_split(
        dataset=dataset,
        destination=tmp_path / "replay",
        initial_fraction=1 / 3,
        minimum_initial_per_class=1,
        excluded_object_manifests=(external,),
    )
    summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))

    assert summary["excluded_manifest_audits"] == [
        {
            "historical_overlap": 0,
            "objects": 1,
            "path": str(external),
            "sha256": summary["excluded_manifest_audits"][0]["sha256"],
            "usage": "overlap_audit_only",
        }
    ]


def test_historical_replay_rejects_external_overlap(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    external = tmp_path / "overlapping_external_labels.csv"
    with external.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object_id"])
        writer.writeheader()
        writer.writerow({"object_id": "train_0_0"})

    with pytest.raises(ValueError, match="overlaps historical data"):
        materialize_astrominn_historical_replay_split(
            dataset=dataset,
            destination=tmp_path / "replay",
            initial_fraction=1 / 3,
            minimum_initial_per_class=1,
            excluded_object_manifests=(external,),
        )


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_historical_replay_rejects_invalid_initial_fraction(
    tmp_path: Path,
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="initial_fraction"):
        materialize_astrominn_historical_replay_split(
            dataset=_dataset(tmp_path),
            destination=tmp_path / "replay",
            initial_fraction=fraction,
            minimum_initial_per_class=1,
        )
