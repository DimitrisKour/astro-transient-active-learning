"""Tests for the BOOM ZTF candidate source."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from al_pipeline.astronomy.boom.source import BoomZTFSource
from al_pipeline.astronomy.ztf.eligibility import ZTFKnownSolarSystemRule


class _FakeAdapter:
    """Minimal adapter used to test BOOM ingestion control flow."""

    def __init__(
        self,
        *,
        late_object_ids: set[str] | None = None,
        max_alert: int = 10,
    ) -> None:
        self.late_object_ids = set() if late_object_ids is None else set(late_object_ids)
        self.max_alert = max_alert

    def alert_number(
        self,
        packet,
    ) -> int:
        if packet["objectId"] in self.late_object_ids:
            return self.max_alert + 1

        return 0

    def prepare(
        self,
        packet,
    ):
        return SimpleNamespace(
            object_id=packet["objectId"],
            alert=0,
            image=np.zeros(
                (3, 63, 63),
                dtype=np.float32,
            ),
            metadata=np.zeros(
                34,
                dtype=np.float32,
            ),
        )


def _alert(
    object_id: str,
    candid: int,
    *,
    jd: float = 2461000.5,
    **candidate_fields,
) -> dict:
    return {
        "objectId": object_id,
        "candidate": {
            "candid": candid,
            "jd": jd,
            "ra": 123.4,
            "dec": -12.3,
            **candidate_fields,
        },
    }


def _aux(
    object_id: str,
) -> dict:
    return {
        "_id": object_id,
        "prv_candidates": [],
    }


def _cutout(
    candid: int,
) -> dict:
    return {
        "_id": candid,
        "cutoutScience": b"science",
        "cutoutTemplate": b"template",
        "cutoutDifference": b"difference",
    }


def test_boom_source_stores_basic_configuration():
    source = BoomZTFSource(
        limit=100,
        max_alert=10,
        excluded_object_ids={
            "ZTF_HISTORICAL",
        },
    )

    assert source.limit == 100
    assert source.max_alert == 10
    assert source.save_raw_payloads is False
    assert source.eligibility_policy is None

    assert source.excluded_object_ids == frozenset(
        {
            "ZTF_HISTORICAL",
        }
    )


def test_boom_source_stores_raw_payload_flag():
    source = BoomZTFSource(
        limit=10,
        save_raw_payloads=True,
    )

    assert source.save_raw_payloads is True


def test_boom_source_rejects_nonboolean_raw_payload_flag():
    with pytest.raises(
        TypeError,
        match=("save_raw_payloads must be a boolean"),
    ):
        BoomZTFSource(
            limit=10,
            save_raw_payloads=1,
        )


def test_boom_source_has_no_time_filter_by_default():
    source = BoomZTFSource(
        limit=10,
    )

    assert source.start_jd is None
    assert source.end_jd is None
    assert source._alert_filter() == {}


def test_boom_source_builds_two_sided_jd_filter():
    source = BoomZTFSource(
        limit=10,
        start_jd=2461000.5,
        end_jd=2461010.5,
    )

    assert source._alert_filter() == {
        "candidate.jd": {
            "$gte": 2461000.5,
            "$lt": 2461010.5,
        }
    }


def test_boom_source_supports_start_only():
    source = BoomZTFSource(
        limit=10,
        start_jd=2461000.5,
    )

    assert source._alert_filter() == {
        "candidate.jd": {
            "$gte": 2461000.5,
        }
    }


def test_boom_source_supports_end_only():
    source = BoomZTFSource(
        limit=10,
        end_jd=2461010.5,
    )

    assert source._alert_filter() == {
        "candidate.jd": {
            "$lt": 2461010.5,
        }
    }


def test_boom_source_converts_jd_bounds_to_float():
    source = BoomZTFSource(
        limit=10,
        start_jd=2461000,
        end_jd=2461010,
    )

    assert source.start_jd == 2461000.0
    assert source.end_jd == 2461010.0

    assert isinstance(
        source.start_jd,
        float,
    )

    assert isinstance(
        source.end_jd,
        float,
    )


def test_boom_source_rejects_equal_jd_bounds():
    with pytest.raises(
        ValueError,
        match=("start_jd must be smaller than end_jd"),
    ):
        BoomZTFSource(
            limit=10,
            start_jd=2461000.5,
            end_jd=2461000.5,
        )


def test_boom_source_rejects_reversed_jd_bounds():
    with pytest.raises(
        ValueError,
        match=("start_jd must be smaller than end_jd"),
    ):
        BoomZTFSource(
            limit=10,
            start_jd=2461010.5,
            end_jd=2461000.5,
        )


def test_boom_source_rejects_nonpositive_limit():
    with pytest.raises(
        ValueError,
        match="limit must be > 0",
    ):
        BoomZTFSource(
            limit=0,
        )


def test_boom_source_rejects_nonpositive_query_batch_size():
    with pytest.raises(
        ValueError,
        match=("query_batch_size must be > 0"),
    ):
        BoomZTFSource(
            limit=10,
            query_batch_size=0,
        )


def test_historical_overlap_is_removed_before_aux_and_cutout_queries(
    tmp_path,
    monkeypatch,
):
    source = BoomZTFSource(
        limit=2,
        excluded_object_ids={
            "ZTF_HISTORICAL",
        },
    )

    source.adapter = _FakeAdapter()

    alerts = [
        _alert(
            "ZTF_HISTORICAL",
            1,
        ),
        _alert(
            "ZTF_NEW",
            2,
        ),
    ]

    monkeypatch.setattr(
        source,
        "_find",
        lambda **kwargs: alerts,
    )

    calls = []

    def fake_find_by_ids(
        *,
        catalog_name,
        ids,
        projection=None,
    ):
        ids = list(ids)
        calls.append(
            (
                catalog_name,
                ids,
                projection,
            )
        )

        if catalog_name == "ZTF_alerts_aux":
            return [
                _aux("ZTF_NEW"),
            ]

        if catalog_name == "ZTF_alerts_cutouts":
            return [
                _cutout(2),
            ]

        raise AssertionError(catalog_name)

    monkeypatch.setattr(
        source,
        "_find_by_ids",
        fake_find_by_ids,
    )

    source._fetch(
        pool_id="test_pool",
        destination=tmp_path,
    )

    assert calls == [
        (
            "ZTF_alerts_aux",
            ["ZTF_NEW"],
            {
                "_id": 1,
                "prv_candidates": 1,
            },
        ),
        (
            "ZTF_alerts_cutouts",
            [2],
            None,
        ),
    ]

    assert source.last_report["historical_overlap"] == 1


def test_eligibility_policy_rejects_sso_before_aux_and_cutout_queries(
    tmp_path,
    monkeypatch,
):
    source = BoomZTFSource(
        limit=2,
        eligibility_policy=(
            ZTFKnownSolarSystemRule(
                max_distance_arcsec=5.0,
            )
        ),
    )

    source.adapter = _FakeAdapter()

    alerts = [
        _alert(
            "ZTF_SSO",
            1,
            ssnamenr="12345",
            ssdistnr=0.4,
        ),
        _alert(
            "ZTF_TRANSIENT",
            2,
            ssnamenr="null",
            ssdistnr=-999,
        ),
    ]

    monkeypatch.setattr(
        source,
        "_find",
        lambda **kwargs: alerts,
    )

    calls = []

    def fake_find_by_ids(
        *,
        catalog_name,
        ids,
        projection=None,
    ):
        ids = list(ids)
        calls.append(
            (
                catalog_name,
                ids,
                projection,
            )
        )

        if catalog_name == "ZTF_alerts_aux":
            return [
                _aux("ZTF_TRANSIENT"),
            ]

        if catalog_name == "ZTF_alerts_cutouts":
            return [
                _cutout(2),
            ]

        raise AssertionError(catalog_name)

    monkeypatch.setattr(
        source,
        "_find_by_ids",
        fake_find_by_ids,
    )

    source._fetch(
        pool_id="test_pool",
        destination=tmp_path,
    )

    assert calls == [
        (
            "ZTF_alerts_aux",
            ["ZTF_TRANSIENT"],
            {
                "_id": 1,
                "prv_candidates": 1,
            },
        ),
        (
            "ZTF_alerts_cutouts",
            [2],
            None,
        ),
    ]

    assert source.last_report["eligibility_rejected"] == 1
    assert source.last_report["eligibility_reasons"] == {
        "known_solar_system_object": 1,
    }
    assert source.last_report["aux_objects_requested"] == 1
    assert source.last_report["cutout_alerts_requested"] == 1

    rejection_path = tmp_path / "eligibility_rejections.json"
    assert rejection_path.exists()

    assert json.loads(rejection_path.read_text(encoding="utf-8")) == [
        {
            "candid": 1,
            "evidence": {
                "max_distance_arcsec": 5.0,
                "ssdistnr": 0.4,
                "ssnamenr": "12345",
            },
            "jd": 2461000.5,
            "object_id": "ZTF_SSO",
            "reason": "known_solar_system_object",
            "source": "ztf_packet",
        }
    ]


def test_late_alert_is_removed_before_cutout_query(
    tmp_path,
    monkeypatch,
):
    source = BoomZTFSource(
        limit=2,
        max_alert=10,
    )

    source.adapter = _FakeAdapter(
        late_object_ids={
            "ZTF_LATE",
        },
        max_alert=10,
    )

    alerts = [
        _alert(
            "ZTF_EARLY",
            11,
        ),
        _alert(
            "ZTF_LATE",
            12,
        ),
    ]

    monkeypatch.setattr(
        source,
        "_find",
        lambda **kwargs: alerts,
    )

    calls = []

    def fake_find_by_ids(
        *,
        catalog_name,
        ids,
        projection=None,
    ):
        ids = list(ids)
        calls.append(
            (
                catalog_name,
                ids,
                projection,
            )
        )

        if catalog_name == "ZTF_alerts_aux":
            return [
                _aux("ZTF_EARLY"),
                _aux("ZTF_LATE"),
            ]

        if catalog_name == "ZTF_alerts_cutouts":
            return [
                _cutout(11),
            ]

        raise AssertionError(catalog_name)

    monkeypatch.setattr(
        source,
        "_find_by_ids",
        fake_find_by_ids,
    )

    source._fetch(
        pool_id="test_pool",
        destination=tmp_path,
    )

    assert calls[-1] == (
        "ZTF_alerts_cutouts",
        [11],
        None,
    )

    assert calls[0] == (
        "ZTF_alerts_aux",
        [
            "ZTF_EARLY",
            "ZTF_LATE",
        ],
        {
            "_id": 1,
            "prv_candidates": 1,
        },
    )

    assert source.last_report["late_alert"] == 1

    assert source.last_report["cutout_alerts_requested"] == 1


def test_normal_ingestion_writes_compact_source_index_without_raw_payloads(
    tmp_path,
    monkeypatch,
):
    source = BoomZTFSource(
        limit=1,
    )

    source.adapter = _FakeAdapter()

    monkeypatch.setattr(
        source,
        "_find",
        lambda **kwargs: [
            _alert(
                "ZTF_TEST",
                101,
            )
        ],
    )

    def fake_find_by_ids(
        *,
        catalog_name,
        ids,
        projection=None,
    ):
        if catalog_name == "ZTF_alerts_aux":
            return [
                _aux("ZTF_TEST"),
            ]

        if catalog_name == "ZTF_alerts_cutouts":
            return [
                _cutout(101),
            ]

        raise AssertionError(catalog_name)

    monkeypatch.setattr(
        source,
        "_find_by_ids",
        fake_find_by_ids,
    )

    source._fetch(
        pool_id="test_pool",
        destination=tmp_path,
    )

    assert (tmp_path / "source_index.json").exists()

    assert not (tmp_path / "raw").exists()

    source_index = json.loads((tmp_path / "source_index.json").read_text(encoding="utf-8"))

    assert source_index == [
        {
            "alert": 0,
            "candid": 101,
            "dec": -12.3,
            "jd": 2461000.5,
            "object_id": "ZTF_TEST",
            "ra": 123.4,
            "source_reference": ("boom:101"),
        }
    ]


def test_find_adds_projection_to_boom_payload(
    monkeypatch,
):
    source = BoomZTFSource(
        limit=1,
    )

    captured = {}

    def fake_post(
        endpoint,
        payload,
    ):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {
            "message": "success",
            "data": [],
        }

    monkeypatch.setattr(
        source,
        "_post",
        fake_post,
    )

    source._find(
        catalog_name="ZTF_alerts_aux",
        filter_document={
            "_id": "ZTF_TEST",
        },
        limit=1,
        projection={
            "_id": 1,
            "prv_candidates": 1,
        },
    )

    assert captured["endpoint"] == "/queries/find"

    assert captured["payload"]["projection"] == {
        "_id": 1,
        "prv_candidates": 1,
    }


def test_normal_ingestion_projects_aux_fields(
    tmp_path,
    monkeypatch,
):
    source = BoomZTFSource(
        limit=1,
    )

    source.adapter = _FakeAdapter()

    monkeypatch.setattr(
        source,
        "_find",
        lambda **kwargs: [
            _alert(
                "ZTF_TEST",
                101,
            )
        ],
    )

    projections = {}

    def fake_find_by_ids(
        *,
        catalog_name,
        ids,
        projection=None,
    ):
        projections[catalog_name] = projection

        if catalog_name == "ZTF_alerts_aux":
            return [
                _aux("ZTF_TEST"),
            ]

        if catalog_name == "ZTF_alerts_cutouts":
            return [
                _cutout(101),
            ]

        raise AssertionError(catalog_name)

    monkeypatch.setattr(
        source,
        "_find_by_ids",
        fake_find_by_ids,
    )

    source._fetch(
        pool_id="test_pool",
        destination=tmp_path,
    )

    assert projections["ZTF_alerts_aux"] == {
        "_id": 1,
        "prv_candidates": 1,
    }

    assert projections["ZTF_alerts_cutouts"] is None

    assert source.last_report["aux_projection"] == [
        "_id",
        "prv_candidates",
    ]


def test_debug_ingestion_requests_full_aux_rows(
    tmp_path,
    monkeypatch,
):
    source = BoomZTFSource(
        limit=1,
        save_raw_payloads=True,
    )

    source.adapter = _FakeAdapter()

    monkeypatch.setattr(
        source,
        "_find",
        lambda **kwargs: [
            _alert(
                "ZTF_TEST",
                101,
            )
        ],
    )

    projections = {}

    def fake_find_by_ids(
        *,
        catalog_name,
        ids,
        projection=None,
    ):
        projections[catalog_name] = projection

        if catalog_name == "ZTF_alerts_aux":
            return [
                _aux("ZTF_TEST"),
            ]

        if catalog_name == "ZTF_alerts_cutouts":
            return [
                _cutout(101),
            ]

        raise AssertionError(catalog_name)

    monkeypatch.setattr(
        source,
        "_find_by_ids",
        fake_find_by_ids,
    )

    source._fetch(
        pool_id="test_pool",
        destination=tmp_path,
    )

    assert projections["ZTF_alerts_aux"] is None

    assert source.last_report["aux_projection"] == "full"


def test_ingestion_report_contains_timings_and_storage(
    tmp_path,
    monkeypatch,
):
    source = BoomZTFSource(
        limit=1,
    )

    source.adapter = _FakeAdapter()

    monkeypatch.setattr(
        source,
        "_find",
        lambda **kwargs: [
            _alert(
                "ZTF_TEST",
                101,
            )
        ],
    )

    def fake_find_by_ids(
        *,
        catalog_name,
        ids,
        projection=None,
    ):
        if catalog_name == "ZTF_alerts_aux":
            return [
                _aux("ZTF_TEST"),
            ]

        if catalog_name == "ZTF_alerts_cutouts":
            return [
                _cutout(101),
            ]

        raise AssertionError(catalog_name)

    monkeypatch.setattr(
        source,
        "_find_by_ids",
        fake_find_by_ids,
    )

    source._fetch(
        pool_id="test_pool",
        destination=tmp_path,
    )

    timings = source.last_report["timings_seconds"]

    assert set(timings) == {
        "alerts_query",
        "alert_prefilter",
        "aux_query",
        "alert_history_filter",
        "cutout_query",
        "raw_payload_write",
        "preprocess_and_sample_write",
        "manifest_and_index_write",
        "total_before_report_write",
    }

    assert all(value >= 0 for value in timings.values())

    storage = source.last_report["storage_bytes"]

    assert storage["samples"] > 0
    assert storage["candidate_pool_csv"] > 0
    assert storage["source_index_json"] > 0
    assert storage["eligibility_rejections_json"] > 0


def test_debug_ingestion_preserves_raw_payloads(
    tmp_path,
    monkeypatch,
):
    source = BoomZTFSource(
        limit=1,
        save_raw_payloads=True,
    )

    source.adapter = _FakeAdapter()

    monkeypatch.setattr(
        source,
        "_find",
        lambda **kwargs: [
            _alert(
                "ZTF_TEST",
                101,
            )
        ],
    )

    def fake_find_by_ids(
        *,
        catalog_name,
        ids,
        projection=None,
    ):
        if catalog_name == "ZTF_alerts_aux":
            return [
                _aux("ZTF_TEST"),
            ]

        if catalog_name == "ZTF_alerts_cutouts":
            return [
                _cutout(101),
            ]

        raise AssertionError(catalog_name)

    monkeypatch.setattr(
        source,
        "_find_by_ids",
        fake_find_by_ids,
    )

    source._fetch(
        pool_id="test_pool",
        destination=tmp_path,
    )

    assert (tmp_path / "raw" / "alerts.json").exists()

    assert (tmp_path / "raw" / "aux.json").exists()

    assert (tmp_path / "raw" / "cutouts" / "101_science.fits.gz").exists()
