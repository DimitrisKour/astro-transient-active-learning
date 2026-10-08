"""Tests for reusable BOOM REST query helpers."""

from __future__ import annotations

from al_pipeline.astronomy.boom.client import BoomClient


def test_boom_client_cone_search_builds_expected_payload(monkeypatch):
    client = BoomClient()
    captured = {}

    def fake_post(endpoint, payload):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {
            "data": {
                "obj-a": [{"_id": 1}],
                "obj-b": [],
            }
        }

    monkeypatch.setattr(client, "post", fake_post)

    result = client.cone_search(
        catalog_name="VSX",
        object_coordinates={
            "obj-a": (10.0, 20.0),
            "obj-b": (30.0, -5.0),
        },
        radius_arcsec=5.0,
    )

    assert captured["endpoint"] == "/queries/cone_search"
    assert captured["payload"] == {
        "catalog_name": "VSX",
        "radius": 5.0,
        "unit": "Arcseconds",
        "object_coordinates": {
            "obj-a": [10.0, 20.0],
            "obj-b": [30.0, -5.0],
        },
    }
    assert result == {
        "obj-a": [{"_id": 1}],
        "obj-b": [],
    }
