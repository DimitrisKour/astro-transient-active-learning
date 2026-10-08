"""Tests for cumulative adaptive-learning labeled sets."""

import csv
from pathlib import Path

import pytest

from al_pipeline.astronomy.data.labeled import AcquisitionRecord, LabeledSetManager
from al_pipeline.astronomy.data.registry import DatasetSpec


def _write_manifest(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    fieldnames = [
        "object_id",
        "filepath",
        "label",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def _manager(
    tmp_path: Path,
) -> LabeledSetManager:
    base = tmp_path / "historical.csv"

    _write_manifest(
        base,
        [
            {
                "object_id": "old_1",
                "filepath": "old_1.npy",
                "label": "A",
            },
            {
                "object_id": "old_2",
                "filepath": "old_2.npy",
                "label": "B",
            },
        ],
    )

    dataset = DatasetSpec(
        name="example",
        survey="ZTF",
        modality="photometry",
        selection_unit="object",
        train_manifest=base,
        object_id_column="object_id",
        sample_file_column="filepath",
        label_column="label",
    )

    return LabeledSetManager(
        dataset=dataset,
        run_dir=tmp_path / "run",
    )


def test_initialize_materializes_l0(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    manager.initialize()

    assert manager.baseline_manifest.exists()
    assert manager.ledger_path.exists()

    rows = manager.get_labeled_through(0)

    assert len(rows) == 2
    assert {row["object_id"] for row in rows} == {
        "old_1",
        "old_2",
    }


def test_record_first_cycle(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    output = manager.record_cycle(
        1,
        acquisitions=[
            AcquisitionRecord(
                object_id="new_1",
                survey="Rubin",
                label="A",
                strategy="entropy",
            )
        ],
        training_rows=[
            {
                "object_id": "new_1",
                "filepath": "new_1.npy",
                "label": "A",
            }
        ],
    )

    assert output == manager.training_manifest(1)
    assert output.exists()

    rows = manager.get_labeled_through(1)

    assert len(rows) == 3


def test_ensure_cycle_is_idempotent(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    acquisition = AcquisitionRecord(
        object_id="new_1",
        survey="ZTF",
        label="A",
        strategy="entropy",
    )
    rows = [
        {
            "object_id": "new_1",
            "filepath": "new_1.npy",
            "label": "A",
        }
    ]

    first = manager.ensure_cycle(
        1,
        acquisitions=[acquisition],
        training_rows=rows,
    )
    second = manager.ensure_cycle(
        1,
        acquisitions=[acquisition],
        training_rows=rows,
    )

    assert first == second
    assert len(manager.get_cycle(1)) == 1


def test_one_object_can_have_multiple_training_rows(
    tmp_path: Path,
) -> None:
    """Object-level acquisition must support alert-level manifests."""
    manager = _manager(tmp_path)

    manager.record_cycle(
        1,
        acquisitions=[
            AcquisitionRecord(
                object_id="new_1",
                survey="ZTF",
                label="A",
            )
        ],
        training_rows=[
            {
                "object_id": "new_1",
                "filepath": "alert_1.npy",
                "label": "A",
            },
            {
                "object_id": "new_1",
                "filepath": "alert_2.npy",
                "label": "A",
            },
        ],
    )

    objects = manager.get_cycle(1)
    rows = manager.get_training_rows_for_cycle(1)

    assert len(objects) == 1
    assert len(rows) == 2


def test_multiple_cycles_are_cumulative(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    manager.record_cycle(
        1,
        acquisitions=[
            AcquisitionRecord(
                object_id="new_1",
                survey="ZTF",
                label="A",
            )
        ],
        training_rows=[
            {
                "object_id": "new_1",
                "filepath": "new_1.npy",
                "label": "A",
            }
        ],
    )

    manager.record_cycle(
        2,
        acquisitions=[
            AcquisitionRecord(
                object_id="new_2",
                survey="Rubin",
                label="B",
            )
        ],
        training_rows=[
            {
                "object_id": "new_2",
                "filepath": "new_2.npy",
                "label": "B",
            }
        ],
    )

    cycle_1 = manager.get_labeled_through(1)
    cycle_2 = manager.get_labeled_through(2)

    assert len(cycle_1) == 3
    assert len(cycle_2) == 4

    assert {row["object_id"] for row in cycle_2} == {
        "old_1",
        "old_2",
        "new_1",
        "new_2",
    }


def test_get_cycle_returns_only_that_cycle(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    manager.record_cycle(
        1,
        acquisitions=[
            AcquisitionRecord(
                object_id="new_1",
                survey="ZTF",
                label="A",
            )
        ],
        training_rows=[
            {
                "object_id": "new_1",
                "filepath": "new_1.npy",
                "label": "A",
            }
        ],
    )

    manager.record_cycle(
        2,
        acquisitions=[
            AcquisitionRecord(
                object_id="new_2",
                survey="Rubin",
                label="B",
            )
        ],
        training_rows=[
            {
                "object_id": "new_2",
                "filepath": "new_2.npy",
                "label": "B",
            }
        ],
    )

    cycle_2 = manager.get_cycle(2)

    assert [record.object_id for record in cycle_2] == ["new_2"]


def test_get_acquired_through_cycle(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    for cycle, object_id in (
        (1, "new_1"),
        (2, "new_2"),
    ):
        manager.record_cycle(
            cycle,
            acquisitions=[
                AcquisitionRecord(
                    object_id=object_id,
                    survey="ZTF",
                    label="A",
                )
            ],
            training_rows=[
                {
                    "object_id": object_id,
                    "filepath": f"{object_id}.npy",
                    "label": "A",
                }
            ],
        )

    records = manager.get_acquired_through(2)

    assert [record.object_id for record in records] == [
        "new_1",
        "new_2",
    ]


def test_survey_provenance_is_preserved(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    manager.record_cycle(
        1,
        acquisitions=[
            AcquisitionRecord(
                object_id="rubin_1",
                survey="Rubin",
                label="A",
            )
        ],
        training_rows=[
            {
                "object_id": "rubin_1",
                "filepath": "rubin_1.npy",
                "label": "A",
            }
        ],
    )

    record = manager.get_cycle(1)[0]

    assert record.survey == "Rubin"


def test_duplicate_object_across_cycles_is_rejected(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    acquisition = AcquisitionRecord(
        object_id="new_1",
        survey="ZTF",
        label="A",
    )

    row = {
        "object_id": "new_1",
        "filepath": "new_1.npy",
        "label": "A",
    }

    manager.record_cycle(
        1,
        acquisitions=[acquisition],
        training_rows=[row],
    )

    with pytest.raises(
        ValueError,
        match="already been acquired",
    ):
        manager.record_cycle(
            2,
            acquisitions=[acquisition],
            training_rows=[row],
        )


def test_cycle_cannot_be_skipped(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    with pytest.raises(
        ValueError,
        match="before cycle 1",
    ):
        manager.record_cycle(
            2,
            acquisitions=[
                AcquisitionRecord(
                    object_id="new_2",
                    survey="ZTF",
                    label="A",
                )
            ],
            training_rows=[
                {
                    "object_id": "new_2",
                    "filepath": "new_2.npy",
                    "label": "A",
                }
            ],
        )


def test_cycle_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    acquisition = AcquisitionRecord(
        object_id="new_1",
        survey="ZTF",
        label="A",
    )

    row = {
        "object_id": "new_1",
        "filepath": "new_1.npy",
        "label": "A",
    }

    manager.record_cycle(
        1,
        acquisitions=[acquisition],
        training_rows=[row],
    )

    with pytest.raises(FileExistsError):
        manager.record_cycle(
            1,
            acquisitions=[acquisition],
            training_rows=[row],
        )


def test_training_rows_must_belong_to_acquired_objects(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    with pytest.raises(
        ValueError,
        match="not accepted",
    ):
        manager.record_cycle(
            1,
            acquisitions=[
                AcquisitionRecord(
                    object_id="new_1",
                    survey="ZTF",
                    label="A",
                )
            ],
            training_rows=[
                {
                    "object_id": "other",
                    "filepath": "other.npy",
                    "label": "A",
                }
            ],
        )


def test_every_acquired_object_needs_training_rows(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    with pytest.raises(
        ValueError,
        match="no model-ready",
    ):
        manager.record_cycle(
            1,
            acquisitions=[
                AcquisitionRecord(
                    object_id="new_1",
                    survey="ZTF",
                    label="A",
                )
            ],
            training_rows=[],
        )


def test_training_row_must_match_historical_schema(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        manager.record_cycle(
            1,
            acquisitions=[
                AcquisitionRecord(
                    object_id="new_1",
                    survey="ZTF",
                    label="A",
                )
            ],
            training_rows=[
                {
                    "object_id": "new_1",
                    "label": "A",
                }
            ],
        )
