"""Tests for ZTF-specific eligibility rules."""

import pytest

from al_pipeline.astronomy.ztf.eligibility import ZTFKnownSolarSystemRule


def test_sso_rule_allows_candidate_without_name():
    rule = ZTFKnownSolarSystemRule()

    decision = rule.evaluate(
        {
            "ssnamenr": None,
            "ssdistnr": 0.2,
        }
    )

    assert decision.eligible is True


@pytest.mark.parametrize(
    "name",
    [
        "",
        "null",
        "None",
        "unknown",
        -999,
        -1,
        0,
    ],
)
def test_sso_rule_allows_missing_or_sentinel_name(name):
    rule = ZTFKnownSolarSystemRule()

    decision = rule.evaluate(
        {
            "ssnamenr": name,
            "ssdistnr": 0.2,
        }
    )

    assert decision.eligible is True


def test_sso_rule_rejects_close_named_object():
    rule = ZTFKnownSolarSystemRule(
        max_distance_arcsec=5.0,
    )

    decision = rule.evaluate(
        {
            "ssnamenr": "12345",
            "ssdistnr": 1.2,
        }
    )

    assert decision.eligible is False
    assert decision.reason == "known_solar_system_object"
    assert decision.source == "ztf_packet"
    assert dict(decision.evidence) == {
        "ssnamenr": "12345",
        "ssdistnr": 1.2,
        "max_distance_arcsec": 5.0,
    }


def test_sso_rule_allows_distant_named_object():
    rule = ZTFKnownSolarSystemRule(
        max_distance_arcsec=5.0,
    )

    decision = rule.evaluate(
        {
            "ssnamenr": "12345",
            "ssdistnr": 5.1,
        }
    )

    assert decision.eligible is True


@pytest.mark.parametrize(
    "distance",
    [None, "bad", -1, float("nan")],
)
def test_sso_rule_allows_unusable_distance(distance):
    rule = ZTFKnownSolarSystemRule()

    decision = rule.evaluate(
        {
            "ssnamenr": "12345",
            "ssdistnr": distance,
        }
    )

    assert decision.eligible is True


def test_sso_rule_rejects_invalid_threshold():
    with pytest.raises(
        ValueError,
        match=("max_distance_arcsec must be finite and >= 0"),
    ):
        ZTFKnownSolarSystemRule(
            max_distance_arcsec=-1,
        )
