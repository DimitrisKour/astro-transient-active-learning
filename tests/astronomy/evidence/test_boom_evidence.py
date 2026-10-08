"""Tests for BOOM-backed catalogue evidence gathering."""

from __future__ import annotations

import json

import numpy as np

from al_pipeline.astronomy.boom.evidence import BoomCatalogEvidenceProvider
from al_pipeline.core.types import SelectionBatch


class _FakeBoomClient:
    def __init__(self, responses=None, failures=None):
        self.responses = responses or {}
        self.failures = set(failures or ())
        self.calls = []

    def cone_search(
        self,
        *,
        catalog_name,
        object_coordinates,
        radius_arcsec,
    ):
        self.calls.append(
            {
                "catalog_name": catalog_name,
                "object_coordinates": dict(object_coordinates),
                "radius_arcsec": radius_arcsec,
            }
        )

        if catalog_name in self.failures:
            raise RuntimeError(f"{catalog_name} unavailable")

        catalog_response = self.responses.get(catalog_name, {})

        return {object_id: catalog_response.get(object_id, []) for object_id in object_coordinates}


def _selection(*ids):
    return SelectionBatch(
        ids=list(ids),
        indices=np.arange(len(ids)),
        scores=np.linspace(1.0, 0.5, len(ids)),
    )


def test_boom_evidence_provider_preserves_native_catalog_rows():
    client = _FakeBoomClient(
        responses={
            "VSX": {
                "obj-a": [
                    {
                        "_id": 123,
                        "ra": 10.0001,
                        "dec": 20.0,
                        "types": ["RRAB"],
                    }
                ]
            },
            "NED": {
                "obj-a": [
                    {
                        "_id": "G-1",
                        "ra": 10.0002,
                        "dec": 20.0,
                        "objtype": "G",
                    }
                ]
            },
        }
    )

    provider = BoomCatalogEvidenceProvider(
        coordinates_by_id={
            "obj-a": (10.0, 20.0),
            "obj-b": (30.0, -5.0),
        },
        catalogs=("VSX", "NED"),
        client=client,
    )

    batch = provider.gather(_selection("obj-a", "obj-b"))

    assert batch.requested_ids == ("obj-a", "obj-b")
    assert batch.failures == ()
    assert batch.objects_without_evidence == ("obj-b",)

    evidence = batch.for_object("obj-a")
    assert [item.source for item in evidence] == ["VSX", "NED"]
    assert all(item.kind == "catalog_match" for item in evidence)
    assert evidence[0].value["types"] == ["RRAB"]
    assert evidence[1].value["objtype"] == "G"
    assert evidence[0].provenance["provider"] == "BOOM"
    assert evidence[0].provenance["query_radius_arcsec"] == 5.0
    assert evidence[0].provenance["separation_arcsec"] > 0.0


def test_boom_evidence_provider_chunks_queries():
    client = _FakeBoomClient()

    provider = BoomCatalogEvidenceProvider(
        coordinates_by_id={
            "a": (1.0, 1.0),
            "b": (2.0, 2.0),
            "c": (3.0, 3.0),
        },
        catalogs=("VSX",),
        query_chunk_size=2,
        client=client,
    )

    provider.gather(_selection("a", "b", "c"))

    assert len(client.calls) == 2
    assert list(client.calls[0]["object_coordinates"]) == ["a", "b"]
    assert list(client.calls[1]["object_coordinates"]) == ["c"]


def test_boom_evidence_provider_records_catalog_failure_and_continues():
    client = _FakeBoomClient(
        responses={
            "NED": {
                "a": [
                    {
                        "_id": "G-1",
                        "ra": 1.0,
                        "dec": 1.0,
                    }
                ]
            }
        },
        failures={"VSX"},
    )

    provider = BoomCatalogEvidenceProvider(
        coordinates_by_id={"a": (1.0, 1.0)},
        catalogs=("VSX", "NED"),
        client=client,
    )

    batch = provider.gather(_selection("a"))

    assert len(batch.evidence) == 1
    assert batch.evidence[0].source == "NED"
    assert len(batch.failures) == 1
    assert batch.failures[0].source == "VSX"
    assert batch.failures[0].reason == "query_failed"
    assert "unavailable" in batch.failures[0].provenance["error"]


def test_boom_evidence_provider_marks_missing_coordinates_as_failure():
    client = _FakeBoomClient()

    provider = BoomCatalogEvidenceProvider(
        coordinates_by_id={"a": (1.0, 1.0)},
        catalogs=("VSX",),
        client=client,
    )

    batch = provider.gather(_selection("a", "missing"))

    assert batch.objects_with_failures == frozenset({"missing"})
    assert batch.failures_for_object("missing")[0].reason == "missing_coordinates"
    assert list(client.calls[0]["object_coordinates"]) == ["a"]


def test_boom_evidence_provider_uses_catalog_supplied_separation():
    client = _FakeBoomClient(
        responses={
            "VSX": {
                "a": [
                    {
                        "_id": 1,
                        "distance_arcsec": 0.42,
                    }
                ]
            }
        }
    )

    provider = BoomCatalogEvidenceProvider(
        coordinates_by_id={"a": (1.0, 1.0)},
        catalogs=("VSX",),
        client=client,
    )

    batch = provider.gather(_selection("a"))

    assert batch.evidence[0].provenance["separation_arcsec"] == 0.42


def test_boom_evidence_provider_can_load_pool_source_index(tmp_path):
    path = tmp_path / "source_index.json"
    path.write_text(
        json.dumps(
            [
                {
                    "object_id": "a",
                    "ra": 10.0,
                    "dec": 20.0,
                },
                {
                    "object_id": "a",
                    "ra": 10.1,
                    "dec": 20.1,
                },
                {
                    "object_id": "b",
                    "ra": 30.0,
                    "dec": -10.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    provider = BoomCatalogEvidenceProvider.from_source_index(
        path,
        catalogs=("VSX",),
        client=_FakeBoomClient(),
    )

    assert provider.coordinates_by_id == {
        "a": (10.0, 20.0),
        "b": (30.0, -10.0),
    }
