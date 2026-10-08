"""Tests for generic rule-based evidence resolution."""

from al_pipeline.core.evidence.types import Evidence, EvidenceBatch, EvidenceFailure
from al_pipeline.core.oracle.resolution import LabelStatus
from al_pipeline.core.oracle.rules import ResolutionClaim
from al_pipeline.core.oracle.rule_based import RuleBasedLabelResolver


class _SourceRule:
    def __init__(self, source, *, status, label=None, native_label=None):
        self.source = source
        self.status = status
        self.label = label
        self.native_label = native_label

    def evaluate(self, evidence):
        if evidence.source != self.source:
            return None

        return ResolutionClaim(
            object_id=evidence.object_id,
            status=self.status,
            label=self.label,
            source=evidence.source,
            native_label=self.native_label,
            rule=type(self).__name__,
        )


def test_rule_based_resolver_combines_agreeing_target_claims():
    resolver = RuleBasedLabelResolver(
        [
            _SourceRule(
                "A",
                status=LabelStatus.LABELED,
                label="AGN",
                native_label="Q",
            ),
            _SourceRule(
                "B",
                status=LabelStatus.LABELED,
                label="AGN",
                native_label="QSO",
            ),
        ]
    )

    batch = EvidenceBatch(
        requested_ids=["obj"],
        evidence=[
            Evidence(object_id="obj", source="A", kind="x"),
            Evidence(object_id="obj", source="B", kind="x"),
        ],
    )

    result = resolver.resolve(batch).resolutions[0]

    assert result.status is LabelStatus.LABELED
    assert result.label == "AGN"
    assert result.source == "multiple"
    assert result.native_label == ("Q", "QSO")
    assert len(result.provenance["claims"]) == 2


def test_rule_based_resolver_leaves_conflicting_claims_unresolved():
    resolver = RuleBasedLabelResolver(
        [
            _SourceRule(
                "A",
                status=LabelStatus.LABELED,
                label="AGN",
            ),
            _SourceRule(
                "B",
                status=LabelStatus.OUTSIDE_LABEL_SPACE,
                native_label="RRAB",
            ),
        ]
    )

    batch = EvidenceBatch(
        requested_ids=["obj"],
        evidence=[
            Evidence(object_id="obj", source="A", kind="x"),
            Evidence(object_id="obj", source="B", kind="x"),
        ],
    )

    result = resolver.resolve(batch).resolutions[0]

    assert result.status is LabelStatus.UNRESOLVED
    assert result.provenance["reason"] == "conflicting_resolving_evidence"


def test_rule_based_resolver_distinguishes_query_failure_from_no_match():
    resolver = RuleBasedLabelResolver(
        [
            _SourceRule(
                "A",
                status=LabelStatus.LABELED,
                label="AGN",
            )
        ]
    )

    failed = resolver.resolve(
        EvidenceBatch(
            requested_ids=["obj"],
            failures=[
                EvidenceFailure(
                    object_id="obj",
                    source="A",
                    reason="query_failed",
                )
            ],
        )
    ).resolutions[0]

    clean_no_match = resolver.resolve(EvidenceBatch(requested_ids=["obj"])).resolutions[0]

    assert failed.status is LabelStatus.UNRESOLVED
    assert failed.provenance["reason"] == "evidence_gathering_failed"
    assert clean_no_match.provenance["reason"] == "no_resolving_evidence"


def test_valid_claim_can_resolve_despite_unrelated_source_failure():
    resolver = RuleBasedLabelResolver(
        [
            _SourceRule(
                "A",
                status=LabelStatus.LABELED,
                label="AGN",
                native_label="Q",
            )
        ]
    )

    result = resolver.resolve(
        EvidenceBatch(
            requested_ids=["obj"],
            evidence=[Evidence(object_id="obj", source="A", kind="x")],
            failures=[
                EvidenceFailure(
                    object_id="obj",
                    source="B",
                    reason="query_failed",
                )
            ],
        )
    ).resolutions[0]

    assert result.status is LabelStatus.LABELED
    assert result.label == "AGN"
    assert len(result.provenance["failures"]) == 1
