"""Tests for end-to-end catalogue-backed acquisition orchestration."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.scoring.entropy import EntropyScorer
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.evidence.types import Evidence, EvidenceBatch
from al_pipeline.astronomy.labels.applecider_catalog import AppleCiDErCatalogResolver
from al_pipeline.core.pool import Pool
from al_pipeline.core.state.history import SQLiteObjectHistoryStore
from al_pipeline.core.types import ClassificationPredictions
from al_pipeline.core.workflow import ActiveLearningResolutionWorkflow


class _CatalogProvider:
    def gather(self, selection):
        rows = {
            "agn": Evidence(
                object_id="agn",
                source="milliquas_v8",
                kind="catalog_match",
                value={"objtype": "Q"},
                provenance={"separation_arcsec": 0.2},
            ),
            "outside": Evidence(
                object_id="outside",
                source="VSX",
                kind="catalog_match",
                value={"types": ["RRAB"]},
                provenance={"separation_arcsec": 0.2},
            ),
        }

        return EvidenceBatch(
            requested_ids=selection.ids,
            evidence=[rows[object_id] for object_id in selection.ids if object_id in rows],
        )


class _ReorderedProvider:
    def gather(self, selection):
        return EvidenceBatch(
            requested_ids=list(reversed(selection.ids)),
        )


def _workflow(
    *,
    outside_action="exclude",
    unresolved_action="keep_pending",
    history_store=None,
    provider=None,
):
    pool = Pool.from_ids(["agn", "outside", "unknown"])
    loop = ActiveLearningLoop(
        scorer=EntropyScorer(),
        selector=TopKSelector(),
        pool=pool,
    )

    return ActiveLearningResolutionWorkflow(
        loop=loop,
        evidence_provider=provider or _CatalogProvider(),
        resolver=AppleCiDErCatalogResolver(),
        history_store=history_store,
        outside_action=outside_action,
        unresolved_action=unresolved_action,
    )


def _predictions():
    return ClassificationPredictions(
        values=np.array(
            [
                [0.50, 0.50],
                [0.60, 0.40],
                [0.70, 0.30],
            ]
        ),
        output_type="probabilities",
    )


def _run(workflow):
    return workflow.run(
        ids=["agn", "outside", "unknown"],
        acquisition_data=_predictions(),
        k=3,
        cycle=1,
        run_id="test-cycle",
    )


def test_default_workflow_resolves_excludes_and_keeps_unresolved_pending():
    workflow = _workflow()

    result = _run(workflow)

    assert result.labeled_ids == ("agn",)
    assert result.outside_ids == ("outside",)
    assert result.unresolved_ids == ("unknown",)
    assert workflow.pool.labeled_ids == frozenset({"agn"})
    assert workflow.pool.excluded_ids == frozenset({"outside"})
    assert workflow.pool.pending_ids == frozenset({"unknown"})
    assert workflow.pool.unlabeled_ids == frozenset()


def test_workflow_can_release_unresolved_and_outside_objects():
    workflow = _workflow(
        outside_action="release",
        unresolved_action="release",
    )

    _run(workflow)

    assert workflow.pool.labeled_ids == frozenset({"agn"})
    assert workflow.pool.excluded_ids == frozenset()
    assert workflow.pool.pending_ids == frozenset()
    assert workflow.pool.unlabeled_ids == frozenset({"outside", "unknown"})


def test_workflow_records_selection_and_resolution_history():
    with TemporaryDirectory() as directory:
        store = SQLiteObjectHistoryStore(Path(directory) / "history.sqlite")
        workflow = _workflow(history_store=store)

        _run(workflow)

        assert store.acquired_ids() == ("agn", "outside", "unknown")
        assert store.latest_resolution("agn").label == "AGN"
        assert store.latest_resolution("outside").status.value == "outside_label_space"
        assert store.latest_resolution("unknown").status.value == "unresolved"


def test_provider_contract_failure_leaves_selection_pending():
    workflow = _workflow(provider=_ReorderedProvider())

    with np.testing.assert_raises_regex(
        ValueError,
        "different object sequence",
    ):
        _run(workflow)

    assert workflow.pool.pending_ids == frozenset({"agn", "outside", "unknown"})


def test_workflow_rejects_invalid_pool_actions():
    with np.testing.assert_raises_regex(
        ValueError,
        "outside_action",
    ):
        _workflow(outside_action="discard")

    with np.testing.assert_raises_regex(
        ValueError,
        "unresolved_action",
    ):
        _workflow(unresolved_action="exclude")
