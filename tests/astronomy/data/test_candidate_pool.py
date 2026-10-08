import csv

import pytest

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest, CandidatePoolRecord


def make_manifest():
    return CandidatePoolManifest(
        pool_id="ztf_test_night",
        records=[
            CandidatePoolRecord(
                survey="ZTF",
                object_id="ZTF_A",
                alert=0,
                filepath="/data/ZTF_A_alert_0.npy",
            ),
            CandidatePoolRecord(
                survey="ZTF",
                object_id="ZTF_A",
                alert=1,
                filepath="/data/ZTF_A_alert_1.npy",
            ),
            CandidatePoolRecord(
                survey="ZTF",
                object_id="ZTF_B",
                alert=0,
                filepath="/data/ZTF_B_alert_0.npy",
                source_reference="123456",
            ),
        ],
    )


def test_multiple_alerts_per_object_are_allowed():
    manifest = make_manifest()

    assert len(manifest.records) == 3

    records = manifest.records_for_object(
        "ZTF_A",
        survey="ZTF",
    )

    assert len(records) == 2
    assert [record.alert for record in records] == [
        0,
        1,
    ]


def test_object_ids_preserve_first_seen_order():
    manifest = make_manifest()

    assert manifest.object_ids == (
        "ZTF_A",
        "ZTF_B",
    )

    assert manifest.qualified_object_ids == (
        "ZTF:ZTF_A",
        "ZTF:ZTF_B",
    )


def test_csv_round_trip(tmp_path):
    manifest = make_manifest()

    path = tmp_path / "pool.csv"

    manifest.to_csv(path)

    loaded = CandidatePoolManifest.from_csv(path)

    assert loaded.pool_id == manifest.pool_id
    assert loaded.records == manifest.records


def test_duplicate_object_alert_is_rejected():
    with pytest.raises(
        ValueError,
        match="Duplicate candidate sample",
    ):
        CandidatePoolManifest(
            pool_id="duplicate_test",
            records=[
                CandidatePoolRecord(
                    survey="ZTF",
                    object_id="ZTF_A",
                    alert=0,
                    filepath="/data/a.npy",
                ),
                CandidatePoolRecord(
                    survey="ZTF",
                    object_id="ZTF_A",
                    alert=0,
                    filepath="/data/b.npy",
                ),
            ],
        )


def test_label_column_is_rejected(tmp_path):
    path = tmp_path / "bad_pool.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pool_id",
                "survey",
                "object_id",
                "alert",
                "filepath",
                "target",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "pool_id": "bad",
                "survey": "ZTF",
                "object_id": "ZTF_A",
                "alert": 0,
                "filepath": "/data/a.npy",
                "target": "SN Ia",
            }
        )

    with pytest.raises(
        ValueError,
        match="must not contain label columns",
    ):
        CandidatePoolManifest.from_csv(path)


def test_missing_files_are_checked_explicitly(
    tmp_path,
):
    existing = tmp_path / "exists.npy"
    existing.touch()

    missing = tmp_path / "missing.npy"

    manifest = CandidatePoolManifest(
        pool_id="files",
        records=[
            CandidatePoolRecord(
                survey="ZTF",
                object_id="A",
                alert=0,
                filepath=str(existing),
            ),
            CandidatePoolRecord(
                survey="ZTF",
                object_id="B",
                alert=0,
                filepath=str(missing),
            ),
        ],
    )

    assert manifest.missing_files() == (str(missing),)
