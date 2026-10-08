"""Interfaces for gathering evidence about selected candidates."""

from __future__ import annotations

from typing import Protocol

from al_pipeline.core.evidence.types import EvidenceBatch
from al_pipeline.core.types import SelectionBatch


class EvidenceProvider(Protocol):
    """Gather provider-native evidence for selected objects.

    Providers are responsible only for retrieving or constructing evidence.
    They must not decide how that evidence maps into the current target label
    space. That scientific policy belongs to a resolver.
    """

    def gather(
        self,
        selection: SelectionBatch,
    ) -> EvidenceBatch:
        """Return evidence for the selected objects."""
        ...


# TODO:
# If real deployment requires asynchronous catalogue/human/spectroscopic
# requests, introduce request/retrieve semantics here rather than making the
# generic EvidenceProvider protocol asynchronous pre-emptively.
