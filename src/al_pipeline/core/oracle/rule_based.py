"""Generic rule-based resolver over provider-native evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from al_pipeline.core.evidence.types import Evidence, EvidenceBatch, EvidenceFailure
from al_pipeline.core.oracle.resolution import LabelResolution, LabelResolutionBatch, LabelStatus
from al_pipeline.core.oracle.rules import EvidenceResolutionRule, ResolutionClaim


class RuleBasedLabelResolver:
    """Combine independent evidence rules into final resolution outcomes.

    The resolver is deliberately conservative. Independent claims that agree
    are combined, while claims that imply different final outcomes leave the
    object unresolved rather than silently choosing one source over another.
    Query failures are retained in provenance and never masquerade as a clean
    no-match result.
    """

    def __init__(
        self,
        rules: Sequence[EvidenceResolutionRule],
    ) -> None:
        self.rules = tuple(rules)

        if not self.rules:
            raise ValueError("At least one resolution rule is required.")

    def resolve(
        self,
        evidence: EvidenceBatch,
    ) -> LabelResolutionBatch:
        """Resolve one outcome for every requested object."""

        resolutions: list[LabelResolution] = []

        for object_id in evidence.requested_ids:
            object_evidence = evidence.for_object(object_id)
            failures = evidence.failures_for_object(object_id)
            claims = self._claims_for_object(
                object_id=object_id,
                evidence=object_evidence,
            )

            resolutions.append(
                self._resolve_object(
                    object_id=object_id,
                    evidence=object_evidence,
                    failures=failures,
                    claims=claims,
                )
            )

        return LabelResolutionBatch(resolutions)

    def _claims_for_object(
        self,
        *,
        object_id: str,
        evidence: Sequence[Evidence],
    ) -> tuple[ResolutionClaim, ...]:
        claims: list[ResolutionClaim] = []

        for item in evidence:
            for rule in self.rules:
                claim = rule.evaluate(item)

                if claim is None:
                    continue

                if claim.object_id != object_id:
                    raise ValueError(
                        "Resolution rule returned a claim for a different "
                        f"object: expected {object_id!r}, got "
                        f"{claim.object_id!r}."
                    )

                claims.append(claim)

        return tuple(claims)

    def _resolve_object(
        self,
        *,
        object_id: str,
        evidence: Sequence[Evidence],
        failures: Sequence[EvidenceFailure],
        claims: Sequence[ResolutionClaim],
    ) -> LabelResolution:
        if not claims:
            return LabelResolution(
                object_id=object_id,
                status=LabelStatus.UNRESOLVED,
                provenance=self._unresolved_provenance(
                    evidence=evidence,
                    failures=failures,
                ),
            )

        outcome_keys = {self._claim_outcome_key(claim) for claim in claims}

        if len(outcome_keys) != 1:
            return LabelResolution(
                object_id=object_id,
                status=LabelStatus.UNRESOLVED,
                provenance={
                    "reason": "conflicting_resolving_evidence",
                    "claims": [self._serialize_claim(claim) for claim in claims],
                    "evidence_count": len(evidence),
                    "failures": [self._serialize_failure(failure) for failure in failures],
                },
            )

        first = claims[0]
        sources = tuple(dict.fromkeys(claim.source for claim in claims))
        native_labels = tuple(
            dict.fromkeys(claim.native_label for claim in claims if claim.native_label is not None)
        )

        source = sources[0] if len(sources) == 1 else "multiple"

        if not native_labels:
            native_label: object | None = None
        elif len(native_labels) == 1:
            native_label = native_labels[0]
        else:
            native_label = native_labels

        return LabelResolution(
            object_id=object_id,
            status=first.status,
            label=first.label,
            source=source,
            native_label=native_label,
            provenance={
                "reason": "resolving_evidence",
                "claims": [self._serialize_claim(claim) for claim in claims],
                "evidence_count": len(evidence),
                "failures": [self._serialize_failure(failure) for failure in failures],
            },
        )

    @staticmethod
    def _claim_outcome_key(
        claim: ResolutionClaim,
    ) -> tuple[LabelStatus, object | None]:
        return claim.status, claim.label

    @staticmethod
    def _unresolved_provenance(
        *,
        evidence: Sequence[Evidence],
        failures: Sequence[EvidenceFailure],
    ) -> dict[str, Any]:
        if not evidence and failures:
            reason = "evidence_gathering_failed"
        elif failures:
            reason = "no_resolving_evidence_with_partial_failures"
        else:
            reason = "no_resolving_evidence"

        return {
            "reason": reason,
            "evidence_count": len(evidence),
            "evidence_sources": list(dict.fromkeys(item.source for item in evidence)),
            "failures": [
                RuleBasedLabelResolver._serialize_failure(failure) for failure in failures
            ],
        }

    @staticmethod
    def _serialize_claim(
        claim: ResolutionClaim,
    ) -> dict[str, Any]:
        return {
            "status": claim.status.value,
            "label": claim.label,
            "source": claim.source,
            "native_label": claim.native_label,
            "rule": claim.rule,
            "provenance": dict(claim.provenance),
        }

    @staticmethod
    def _serialize_failure(
        failure: EvidenceFailure,
    ) -> dict[str, Any]:
        return {
            "source": failure.source,
            "reason": failure.reason,
            "provenance": dict(failure.provenance),
        }
