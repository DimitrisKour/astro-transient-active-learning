"""Tests for generic adaptive evidence types."""

import pytest

from al_pipeline.core.evidence.types import Evidence, EvidenceBatch


def test_evidence_preserves_provider_native_information():
    provenance = {
        "separation_arcsec": 0.42,
        "catalog_id": "QSO-1",
    }

    item = Evidence(
        object_id="obj-1",
        source="SIMBAD",
        kind="classification",
        value="QSO",
        provenance=provenance,
    )

    provenance["catalog_id"] = "changed"

    assert item.object_id == "obj-1"
    assert item.source == "SIMBAD"
    assert item.kind == "classification"
    assert item.value == "QSO"
    assert item.provenance["separation_arcsec"] == 0.42
    assert item.provenance["catalog_id"] == "QSO-1"

    with pytest.raises(TypeError):
        item.provenance["new"] = 1


def test_evidence_requires_nonempty_identity_fields():
    with pytest.raises(ValueError, match="object_id"):
        Evidence(
            object_id="",
            source="SIMBAD",
            kind="classification",
        )

    with pytest.raises(ValueError, match="source"):
        Evidence(
            object_id="obj-1",
            source="",
            kind="classification",
        )

    with pytest.raises(ValueError, match="kind"):
        Evidence(
            object_id="obj-1",
            source="SIMBAD",
            kind="",
        )


def test_evidence_batch_retains_objects_without_matches():
    batch = EvidenceBatch(
        requested_ids=["a", "b", "c"],
        evidence=[
            Evidence(
                object_id="a",
                source="VSX",
                kind="classification",
                value="RRAB",
            ),
            Evidence(
                object_id="a",
                source="Gaia",
                kind="classification",
                value="RR",
            ),
            Evidence(
                object_id="c",
                source="NED",
                kind="context",
                value="G",
            ),
        ],
    )

    assert batch.requested_ids == ("a", "b", "c")
    assert [item.source for item in batch.for_object("a")] == ["VSX", "Gaia"]
    assert batch.objects_with_evidence == frozenset({"a", "c"})
    assert batch.objects_without_evidence == ("b",)


def test_evidence_batch_rejects_unrequested_object():
    with pytest.raises(ValueError, match="not requested"):
        EvidenceBatch(
            requested_ids=["a"],
            evidence=[
                Evidence(
                    object_id="b",
                    source="TNS",
                    kind="classification",
                    value="SN Ia",
                )
            ],
        )


def test_evidence_batch_rejects_duplicate_requested_ids():
    with pytest.raises(ValueError, match="duplicates"):
        EvidenceBatch(
            requested_ids=["a", "a"],
        )


def test_evidence_batches_merge_across_providers():
    simbad = EvidenceBatch(
        requested_ids=["a", "b"],
        evidence=[
            Evidence(
                object_id="a",
                source="SIMBAD",
                kind="classification",
                value="AGN",
            )
        ],
    )

    desi = EvidenceBatch(
        requested_ids=["a", "b"],
        evidence=[
            Evidence(
                object_id="a",
                source="DESI_DR1",
                kind="spectroscopy",
                value="QSO",
            ),
            Evidence(
                object_id="b",
                source="DESI_DR1",
                kind="spectroscopy",
                value="STAR",
            ),
        ],
    )

    merged = simbad.merge(desi)

    assert [(item.object_id, item.source) for item in merged.evidence] == [
        ("a", "SIMBAD"),
        ("a", "DESI_DR1"),
        ("b", "DESI_DR1"),
    ]


def test_evidence_batch_merge_requires_same_request_set_and_order():
    first = EvidenceBatch(
        requested_ids=["a", "b"],
    )
    different = EvidenceBatch(
        requested_ids=["b", "a"],
    )

    with pytest.raises(ValueError, match="identical requested_ids"):
        first.merge(different)


def test_evidence_batch_preserves_query_failures_separately():
    from al_pipeline.core.evidence.types import EvidenceFailure

    batch = EvidenceBatch(
        requested_ids=["a", "b"],
        evidence=[
            Evidence(
                object_id="a",
                source="VSX",
                kind="catalog_match",
                value={"types": ["RRAB"]},
            )
        ],
        failures=[
            EvidenceFailure(
                object_id="b",
                source="DESI_DR1",
                reason="query_failed",
                provenance={"error": "timeout"},
            )
        ],
    )

    assert batch.objects_with_failures == frozenset({"b"})
    assert batch.objects_without_evidence == ("b",)
    assert batch.failures_for_object("b")[0].source == "DESI_DR1"
    assert batch.failures_for_object("b")[0].provenance["error"] == "timeout"


def test_evidence_batch_merge_preserves_failures():
    from al_pipeline.core.evidence.types import EvidenceFailure

    first = EvidenceBatch(
        requested_ids=["a"],
        failures=[
            EvidenceFailure(
                object_id="a",
                source="VSX",
                reason="query_failed",
            )
        ],
    )

    second = EvidenceBatch(
        requested_ids=["a"],
        evidence=[
            Evidence(
                object_id="a",
                source="NED",
                kind="catalog_match",
                value={"objtype": "G"},
            )
        ],
    )

    merged = first.merge(second)

    assert len(merged.evidence) == 1
    assert len(merged.failures) == 1
    assert merged.failures[0].source == "VSX"
