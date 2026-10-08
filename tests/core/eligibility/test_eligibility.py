"""Tests for generic candidate eligibility interfaces."""

import pytest

from al_pipeline.core.eligibility.base import EligibilityDecision, EligibilityPolicy


class _AllowRule:
    def evaluate(self, data):
        return EligibilityDecision.allow()


class _RejectRule:
    def __init__(self, reason):
        self.reason = reason

    def evaluate(self, data):
        return EligibilityDecision.reject(
            reason=self.reason,
            source="test",
            evidence={"value": data},
        )


class _InvalidRule:
    def evaluate(self, data):
        return False


def test_eligible_decision_has_no_rejection_reason():
    decision = EligibilityDecision.allow()

    assert decision.eligible is True
    assert decision.reason is None
    assert decision.source is None
    assert dict(decision.evidence) == {}


def test_rejection_requires_reason():
    with pytest.raises(
        ValueError,
        match="must carry a non-empty reason",
    ):
        EligibilityDecision(
            eligible=False,
        )


def test_eligible_decision_rejects_reason():
    with pytest.raises(
        ValueError,
        match="must not carry a rejection reason",
    ):
        EligibilityDecision(
            eligible=True,
            reason="not_used",
        )


def test_rejection_copies_evidence():
    evidence = {"value": 1}

    decision = EligibilityDecision.reject(
        reason="example",
        source="unit_test",
        evidence=evidence,
    )

    evidence["value"] = 2

    assert decision.eligible is False
    assert decision.reason == "example"
    assert decision.source == "unit_test"
    assert dict(decision.evidence) == {
        "value": 1,
    }


def test_empty_policy_allows_candidate():
    policy = EligibilityPolicy()

    assert policy.evaluate("x").eligible is True


def test_policy_returns_first_rejection():
    policy = EligibilityPolicy(
        [
            _AllowRule(),
            _RejectRule("first"),
            _RejectRule("second"),
        ]
    )

    decision = policy.evaluate("candidate")

    assert decision.eligible is False
    assert decision.reason == "first"
    assert dict(decision.evidence) == {
        "value": "candidate",
    }


def test_policy_rejects_invalid_rule_return_type():
    policy = EligibilityPolicy([_InvalidRule()])

    with pytest.raises(
        TypeError,
        match=("Eligibility rules must return EligibilityDecision"),
    ):
        policy.evaluate("candidate")
