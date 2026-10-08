"""Generic rule primitives for evidence-backed label resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from al_pipeline.core.evidence.types import Evidence
from al_pipeline.core.oracle.resolution import LabelStatus


@dataclass(frozen=True, slots=True)
class ResolutionClaim:
    """One rule's resolving claim about one object.

    Claims are intentionally intermediate. A single object may accumulate
    several independent claims, which the final resolver can combine or mark
    as conflicting before emitting a ``LabelResolution``.
    """

    object_id: str
    status: LabelStatus
    source: str
    native_label: object | None = None
    label: object | None = None
    rule: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = str(self.object_id).strip()
        source = str(self.source).strip()

        if not object_id:
            raise ValueError("object_id must not be empty.")

        if not source:
            raise ValueError("source must not be empty.")

        if not isinstance(self.status, LabelStatus):
            raise TypeError("status must be a LabelStatus.")

        if self.status is LabelStatus.UNRESOLVED:
            raise ValueError(
                "ResolutionClaim must represent resolving evidence, not an unresolved outcome."
            )

        if self.status is LabelStatus.LABELED:
            if self.label is None:
                raise ValueError("A labeled claim must contain a target label.")
        elif self.label is not None:
            raise ValueError("Only labeled claims may contain a target label.")

        if self.rule is not None:
            rule = str(self.rule).strip()
            if not rule:
                raise ValueError("rule must not be an empty string.")
            object.__setattr__(self, "rule", rule)

        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


class EvidenceResolutionRule(Protocol):
    """Interpret one evidence record as an optional resolving claim."""

    def evaluate(
        self,
        evidence: Evidence,
    ) -> ResolutionClaim | None:
        """Return a claim when this rule considers the evidence decisive."""
        ...
