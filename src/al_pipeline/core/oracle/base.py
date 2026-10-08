"""Base interfaces for active-learning label providers."""

from __future__ import annotations

from typing import Protocol

from al_pipeline.core.evidence.types import EvidenceBatch
from al_pipeline.core.oracle.resolution import LabelResolutionBatch
from al_pipeline.core.types import LabeledBatch, SelectionBatch


class LabelProvider(Protocol):
    """Interface implemented by active-learning label providers.

    A label provider receives objects selected for acquisition and returns
    labels for any objects that it can resolve.

    This interface is appropriate when the provider's output can already be
    represented directly as training labels.
    """

    def query(
        self,
        selection: SelectionBatch,
    ) -> LabeledBatch:
        """Return labels obtained for selected objects."""
        ...


class LabelResolver(Protocol):
    """Interpret gathered evidence in the current target-label context.

    Evidence retrieval and scientific interpretation are deliberately
    separate. A resolver receives provider-native evidence and decides whether
    each requested object is labeled, unresolved, or outside the current
    target label space.
    """

    def resolve(
        self,
        evidence: EvidenceBatch,
    ) -> LabelResolutionBatch:
        """Return one resolution outcome per requested object."""
        ...


# TODO:
# If the real oracle workflow is asynchronous (for example human inspection
# or spectroscopic follow-up), determine whether separate request/retrieve
# operations are needed. Do not force asynchronous behavior into either
# synchronous protocol until the workflow is known.
