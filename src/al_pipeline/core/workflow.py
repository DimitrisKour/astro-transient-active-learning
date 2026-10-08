"""End-to-end orchestration for one catalogue-backed acquisition cycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.evidence.types import EvidenceBatch
from al_pipeline.core.evidence.base import EvidenceProvider
from al_pipeline.core.oracle.resolution import LabelResolutionBatch, LabelStatus
from al_pipeline.core.oracle.base import LabelResolver
from al_pipeline.core.pool import Pool
from al_pipeline.core.state.history import ObjectHistoryStore
from al_pipeline.core.types import LabeledBatch, SelectionBatch


ScoringInputT = TypeVar("ScoringInputT")
OutsideAction: TypeAlias = Literal[
    "exclude",
    "release",
    "keep_pending",
]
UnresolvedAction: TypeAlias = Literal[
    "release",
    "keep_pending",
]


@dataclass(frozen=True, slots=True)
class ActiveLearningCycleResult:
    """Artifacts and outcomes produced by one acquisition-resolution cycle."""

    selection: SelectionBatch
    evidence: EvidenceBatch
    resolutions: LabelResolutionBatch
    labeled: LabeledBatch

    @property
    def labeled_ids(self) -> tuple[str, ...]:
        return tuple(self.labeled.ids)

    @property
    def unresolved_ids(self) -> tuple[str, ...]:
        return tuple(
            result.object_id for result in self.resolutions.by_status(LabelStatus.UNRESOLVED)
        )

    @property
    def outside_ids(self) -> tuple[str, ...]:
        return tuple(
            result.object_id
            for result in self.resolutions.by_status(LabelStatus.OUTSIDE_LABEL_SPACE)
        )


class ActiveLearningResolutionWorkflow(Generic[ScoringInputT]):
    """Connect acquisition, evidence gathering, resolution, and pool state.

    The workflow deliberately makes the two non-label pool decisions explicit:
    outside-taxonomy objects can be excluded, released, or kept pending, while
    unresolved objects can be released or kept pending for later inspection.
    Successfully labelled objects always move from pending to labelled.

    If a history store is supplied, the committed selection is recorded before
    evidence gathering and every resolution outcome is recorded before pool
    transitions are applied. A provider exception therefore leaves the selected
    objects pending and preserves their acquisition event.
    """

    def __init__(
        self,
        *,
        loop: ActiveLearningLoop[ScoringInputT],
        evidence_provider: EvidenceProvider,
        resolver: LabelResolver,
        history_store: ObjectHistoryStore | None = None,
        outside_action: OutsideAction = "exclude",
        unresolved_action: UnresolvedAction = "keep_pending",
    ) -> None:
        if outside_action not in {
            "exclude",
            "release",
            "keep_pending",
        }:
            raise ValueError("outside_action must be 'exclude', 'release', or 'keep_pending'.")

        if unresolved_action not in {"release", "keep_pending"}:
            raise ValueError("unresolved_action must be 'release' or 'keep_pending'.")

        self.loop = loop
        self.evidence_provider = evidence_provider
        self.resolver = resolver
        self.history_store = history_store
        self.outside_action = outside_action
        self.unresolved_action = unresolved_action

    @property
    def pool(self) -> Pool:
        return self.loop.pool

    def run(
        self,
        *,
        ids: list[str],
        acquisition_data: ScoringInputT,
        k: int,
        cycle: int | None = None,
        run_id: str | None = None,
        scorer: str | None = None,
        history_metadata: Mapping[str, Any] | None = None,
    ) -> ActiveLearningCycleResult:
        """Commit and resolve one active-learning acquisition cycle."""

        selection = self.loop.acquire(
            ids=ids,
            acquisition_data=acquisition_data,
            k=k,
        )

        if self.history_store is not None:
            self.history_store.record_selection(
                selection,
                cycle=cycle,
                run_id=run_id,
                scorer=(scorer if scorer is not None else type(self.loop.scorer).__name__),
                metadata=history_metadata,
            )

        evidence = self.evidence_provider.gather(selection)
        expected_ids = tuple(selection.ids)

        if evidence.requested_ids != expected_ids:
            raise ValueError(
                "Evidence provider returned a different object sequence from "
                "the committed selection."
            )

        resolutions = self.resolver.resolve(evidence)
        resolution_ids = tuple(result.object_id for result in resolutions.resolutions)

        if resolution_ids != expected_ids:
            raise ValueError(
                "Label resolver returned a different object sequence from the committed selection."
            )

        if self.history_store is not None:
            self.history_store.record_resolutions(
                resolutions,
                cycle=cycle,
                run_id=run_id,
            )

        labeled = resolutions.labeled_batch()
        outside_ids = [
            result.object_id for result in resolutions.by_status(LabelStatus.OUTSIDE_LABEL_SPACE)
        ]
        unresolved_ids = [
            result.object_id for result in resolutions.by_status(LabelStatus.UNRESOLVED)
        ]

        if labeled.ids:
            self.pool.resolve(labeled.ids)

        self._apply_outside_action(outside_ids)
        self._apply_unresolved_action(unresolved_ids)

        return ActiveLearningCycleResult(
            selection=selection,
            evidence=evidence,
            resolutions=resolutions,
            labeled=labeled,
        )

    def _apply_outside_action(self, object_ids: list[str]) -> None:
        if not object_ids or self.outside_action == "keep_pending":
            return

        if self.outside_action == "exclude":
            self.pool.exclude(object_ids)
        else:
            self.pool.release(object_ids)

    def _apply_unresolved_action(self, object_ids: list[str]) -> None:
        if not object_ids or self.unresolved_action == "keep_pending":
            return

        self.pool.release(object_ids)
