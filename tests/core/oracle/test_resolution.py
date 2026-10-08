"""Tests for real-label resolution result types."""

import numpy as np
import pytest

from al_pipeline.core.oracle.resolution import LabelResolution, LabelResolutionBatch, LabelStatus


def test_labeled_resolution_requires_label():
    with pytest.raises(
        ValueError,
        match="must contain a target label",
    ):
        LabelResolution(
            object_id="obj-1",
            status=LabelStatus.LABELED,
        )


def test_unresolved_resolution_rejects_label():
    with pytest.raises(
        ValueError,
        match="Only labeled resolutions",
    ):
        LabelResolution(
            object_id="obj-1",
            status=LabelStatus.UNRESOLVED,
            label="SN II",
        )


def test_outside_label_space_can_preserve_native_label():
    result = LabelResolution(
        object_id="obj-1",
        status=LabelStatus.OUTSIDE_LABEL_SPACE,
        source="VSX",
        native_label="RRAB",
    )

    assert result.label is None
    assert result.source == "VSX"
    assert result.native_label == "RRAB"


def test_resolution_batch_rejects_duplicate_ids():
    result = LabelResolution(
        object_id="obj-1",
        status=LabelStatus.UNRESOLVED,
    )

    with pytest.raises(
        ValueError,
        match="duplicate object IDs",
    ):
        LabelResolutionBatch([result, result])


def test_labeled_batch_keeps_only_resolved_objects():
    results = LabelResolutionBatch(
        [
            LabelResolution(
                object_id="obj-1",
                status=LabelStatus.LABELED,
                label="SN II",
                source="TNS",
                native_label="SN IIP",
            ),
            LabelResolution(
                object_id="obj-2",
                status=LabelStatus.UNRESOLVED,
            ),
            LabelResolution(
                object_id="obj-3",
                status=LabelStatus.OUTSIDE_LABEL_SPACE,
                source="VSX",
                native_label="RRAB",
            ),
            LabelResolution(
                object_id="obj-4",
                status=LabelStatus.LABELED,
                label="AGN",
                source="milliquas_v8",
                native_label="Q",
            ),
        ]
    )

    batch = results.labeled_batch()

    assert batch.ids == [
        "obj-1",
        "obj-4",
    ]

    np.testing.assert_array_equal(
        batch.labels,
        np.asarray(
            [
                "SN II",
                "AGN",
            ]
        ),
    )


def test_by_status_preserves_order():
    results = LabelResolutionBatch(
        [
            LabelResolution(
                object_id="obj-1",
                status=LabelStatus.UNRESOLVED,
            ),
            LabelResolution(
                object_id="obj-2",
                status=LabelStatus.LABELED,
                label="CV",
            ),
            LabelResolution(
                object_id="obj-3",
                status=LabelStatus.UNRESOLVED,
            ),
        ]
    )

    unresolved = results.by_status(LabelStatus.UNRESOLVED)

    assert [result.object_id for result in unresolved] == [
        "obj-1",
        "obj-3",
    ]


def test_label_resolver_can_interpret_evidence_batch():
    from al_pipeline.core.evidence.types import Evidence, EvidenceBatch

    class ExampleResolver:
        def resolve(self, evidence):
            resolutions = []

            for object_id in evidence.requested_ids:
                matches = evidence.for_object(object_id)

                if any(item.value == "QSO" for item in matches):
                    resolutions.append(
                        LabelResolution(
                            object_id=object_id,
                            status=LabelStatus.LABELED,
                            label="AGN",
                            source="example",
                            native_label="QSO",
                        )
                    )
                else:
                    resolutions.append(
                        LabelResolution(
                            object_id=object_id,
                            status=LabelStatus.UNRESOLVED,
                        )
                    )

            return LabelResolutionBatch(resolutions)

    evidence = EvidenceBatch(
        requested_ids=["a", "b"],
        evidence=[
            Evidence(
                object_id="a",
                source="SIMBAD",
                kind="classification",
                value="QSO",
            )
        ],
    )

    results = ExampleResolver().resolve(evidence)

    assert results.by_status(LabelStatus.LABELED)[0].object_id == "a"
    assert results.by_status(LabelStatus.UNRESOLVED)[0].object_id == "b"
