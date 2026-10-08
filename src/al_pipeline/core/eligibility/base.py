"""Generic eligibility decisions for adaptive candidate workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar


EligibilityInputT = TypeVar("EligibilityInputT")


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    """Result of evaluating whether one candidate may enter a workflow.

    Eligibility is deliberately distinct from scientific label resolution.
    A candidate can be a valid object outside the current model
    taxonomy and still remain eligible for anomaly review. Conversely, a
    provider may already know that an object is irrelevant to a particular
    acquisition task and reject it before expensive preprocessing/inference.
    """

    eligible: bool
    reason: str | None = None
    source: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reason = None if self.reason is None else str(self.reason).strip()
        source = None if self.source is None else str(self.source).strip()

        if self.eligible and reason is not None:
            raise ValueError("Eligible decisions must not carry a rejection reason.")

        if not self.eligible and not reason:
            raise ValueError("Ineligible decisions must carry a non-empty reason.")

        if source == "":
            source = None

        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )

    @classmethod
    def allow(cls) -> "EligibilityDecision":
        """Return an eligible decision with no rejection provenance."""

        return cls(eligible=True)

    @classmethod
    def reject(
        cls,
        *,
        reason: str,
        source: str | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> "EligibilityDecision":
        """Return an ineligible decision with structured provenance."""

        return cls(
            eligible=False,
            reason=reason,
            source=source,
            evidence=({} if evidence is None else evidence),
        )


class EligibilityRule(Protocol[EligibilityInputT]):
    """Protocol for one task-specific eligibility rule."""

    def evaluate(
        self,
        data: EligibilityInputT,
    ) -> EligibilityDecision:
        """Return whether ``data`` is eligible for the downstream task."""

        ...


class EligibilityPolicy(Generic[EligibilityInputT]):
    """Ordered composition of independent eligibility rules.

    The first rejecting rule wins. This keeps policy behaviour deterministic
    and lets inexpensive/provider-native rules run before more costly ones.
    An empty policy allows every candidate.
    """

    def __init__(
        self,
        rules: Iterable[EligibilityRule[EligibilityInputT]] = (),
    ) -> None:
        self.rules = tuple(rules)

    def evaluate(
        self,
        data: EligibilityInputT,
    ) -> EligibilityDecision:
        """Evaluate rules in order and return the first rejection."""

        for rule in self.rules:
            decision = rule.evaluate(data)

            if not isinstance(
                decision,
                EligibilityDecision,
            ):
                raise TypeError("Eligibility rules must return EligibilityDecision.")

            if not decision.eligible:
                return decision

        return EligibilityDecision.allow()
