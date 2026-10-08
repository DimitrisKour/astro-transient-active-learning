"""ZTF-specific eligibility rules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from al_pipeline.core.eligibility.base import EligibilityDecision


class ZTFKnownSolarSystemRule:
    """Reject candidates already associated with a nearby known SSO.

    ZTF alert packets carry the nearest known Solar-System-object name and
    distance through ``ssnamenr`` and ``ssdistnr``. The rule acts only when
    both fields provide a usable association within the configured angular
    distance. Missing, sentinel, invalid, or more distant associations remain
    eligible.

    This is an eligibility rule for a transient-classification task, not a
    statement that Solar-System objects are scientifically uninteresting.
    """

    reason = "known_solar_system_object"
    source = "ztf_packet"

    def __init__(
        self,
        *,
        max_distance_arcsec: float = 5.0,
    ) -> None:
        max_distance_arcsec = float(max_distance_arcsec)

        if not math.isfinite(max_distance_arcsec) or max_distance_arcsec < 0:
            raise ValueError("max_distance_arcsec must be finite and >= 0.")

        self.max_distance_arcsec = max_distance_arcsec

    def evaluate(
        self,
        candidate: Mapping[str, Any],
    ) -> EligibilityDecision:
        """Evaluate one ZTF candidate mapping."""

        name = candidate.get("ssnamenr")

        if not self._valid_name(name):
            return EligibilityDecision.allow()

        distance_value = candidate.get("ssdistnr")

        try:
            distance = float(distance_value)
        except (TypeError, ValueError):
            return EligibilityDecision.allow()

        if not math.isfinite(distance) or distance < 0 or distance > self.max_distance_arcsec:
            return EligibilityDecision.allow()

        return EligibilityDecision.reject(
            reason=self.reason,
            source=self.source,
            evidence={
                "ssnamenr": name,
                "ssdistnr": distance,
                "max_distance_arcsec": (self.max_distance_arcsec),
            },
        )

    @staticmethod
    def _valid_name(
        value: Any,
    ) -> bool:
        if value is None:
            return False

        text = str(value).strip()

        if not text:
            return False

        if text.lower() in {
            "none",
            "null",
            "nan",
            "unknown",
        }:
            return False

        try:
            numeric = float(text)
        except ValueError:
            return True

        return math.isfinite(numeric) and numeric > 0
