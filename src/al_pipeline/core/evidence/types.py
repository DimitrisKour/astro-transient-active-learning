"""Generic evidence records for adaptive resolution workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class Evidence:
    """One piece of information gathered about one selected object.

    Evidence is deliberately weaker than a resolved label. It records what a
    provider knows without deciding how that information maps into a model's
    current target taxonomy.
    """

    object_id: str
    source: str
    kind: str
    value: object | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = str(self.object_id).strip()
        source = str(self.source).strip()
        kind = str(self.kind).strip()

        if not object_id:
            raise ValueError("object_id must not be empty.")

        if not source:
            raise ValueError("source must not be empty.")

        if not kind:
            raise ValueError("kind must not be empty.")

        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


@dataclass(frozen=True, slots=True)
class EvidenceFailure:
    """Failure to gather one source of evidence for one requested object.

    A failed query must remain distinguishable from a successful query that
    returned no catalogue match. A failed source query must not be interpreted as a successful request
    that found no matching evidence.
    """

    object_id: str
    source: str
    reason: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = str(self.object_id).strip()
        source = str(self.source).strip()
        reason = str(self.reason).strip()

        if not object_id:
            raise ValueError("object_id must not be empty.")

        if not source:
            raise ValueError("source must not be empty.")

        if not reason:
            raise ValueError("reason must not be empty.")

        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


@dataclass(frozen=True, slots=True)
class EvidenceBatch:
    """Evidence gathered for one requested collection of objects.

    ``requested_ids`` is retained even when no evidence is found. ``failures``
    separately records requests that could not be completed, so downstream
    resolution never has to confuse "no match" with "query failed".
    """

    requested_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    failures: tuple[EvidenceFailure, ...]

    def __init__(
        self,
        *,
        requested_ids: Sequence[str],
        evidence: Sequence[Evidence] = (),
        failures: Sequence[EvidenceFailure] = (),
    ) -> None:
        normalized_ids = tuple(requested_ids)
        normalized_evidence = tuple(evidence)
        normalized_failures = tuple(failures)

        if not all(isinstance(object_id, str) for object_id in normalized_ids):
            raise TypeError("requested_ids must contain only strings.")

        if any(not object_id for object_id in normalized_ids):
            raise ValueError("requested_ids must not contain empty strings.")

        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("requested_ids must not contain duplicates.")

        if not all(isinstance(item, Evidence) for item in normalized_evidence):
            raise TypeError("evidence must contain only Evidence objects.")

        if not all(isinstance(item, EvidenceFailure) for item in normalized_failures):
            raise TypeError("failures must contain only EvidenceFailure objects.")

        requested = set(normalized_ids)

        unexpected_evidence = sorted(
            {item.object_id for item in normalized_evidence if item.object_id not in requested}
        )

        if unexpected_evidence:
            raise ValueError(
                f"Evidence contains object IDs that were not requested: {unexpected_evidence}"
            )

        unexpected_failures = sorted(
            {item.object_id for item in normalized_failures if item.object_id not in requested}
        )

        if unexpected_failures:
            raise ValueError(
                "Evidence failures contain object IDs that were not requested: "
                f"{unexpected_failures}"
            )

        object.__setattr__(
            self,
            "requested_ids",
            normalized_ids,
        )
        object.__setattr__(
            self,
            "evidence",
            normalized_evidence,
        )
        object.__setattr__(
            self,
            "failures",
            normalized_failures,
        )

    def for_object(
        self,
        object_id: str,
    ) -> tuple[Evidence, ...]:
        """Return all evidence associated with one requested object."""

        if object_id not in self.requested_ids:
            raise KeyError(f"Object ID was not requested: {object_id!r}")

        return tuple(item for item in self.evidence if item.object_id == object_id)

    def failures_for_object(
        self,
        object_id: str,
    ) -> tuple[EvidenceFailure, ...]:
        """Return all gathering failures associated with one object."""

        if object_id not in self.requested_ids:
            raise KeyError(f"Object ID was not requested: {object_id!r}")

        return tuple(item for item in self.failures if item.object_id == object_id)

    @property
    def objects_with_evidence(self) -> frozenset[str]:
        """Return requested object IDs with at least one evidence record."""

        return frozenset(item.object_id for item in self.evidence)

    @property
    def objects_with_failures(self) -> frozenset[str]:
        """Return requested object IDs affected by at least one query failure."""

        return frozenset(item.object_id for item in self.failures)

    @property
    def objects_without_evidence(self) -> tuple[str, ...]:
        """Return requested IDs with no evidence records.

        Callers that need to distinguish a genuine no-match result from a
        provider failure should also inspect ``objects_with_failures`` or
        ``failures_for_object``.
        """

        present = self.objects_with_evidence

        return tuple(object_id for object_id in self.requested_ids if object_id not in present)

    def merge(
        self,
        *others: "EvidenceBatch",
    ) -> "EvidenceBatch":
        """Combine provider results queried on the same object set."""

        merged_evidence = list(self.evidence)
        merged_failures = list(self.failures)

        for other in others:
            if not isinstance(other, EvidenceBatch):
                raise TypeError("Can only merge with another EvidenceBatch.")

            if other.requested_ids != self.requested_ids:
                raise ValueError("Evidence batches must have identical requested_ids to be merged.")

            merged_evidence.extend(other.evidence)
            merged_failures.extend(other.failures)

        return EvidenceBatch(
            requested_ids=self.requested_ids,
            evidence=merged_evidence,
            failures=merged_failures,
        )
