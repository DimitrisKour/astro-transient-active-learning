"""BOOM catalogue evidence provider."""

from __future__ import annotations

import json
import math

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from al_pipeline.astronomy.boom.client import BoomClient
from al_pipeline.core.evidence.types import Evidence, EvidenceBatch, EvidenceFailure
from al_pipeline.core.types import SelectionBatch


DEFAULT_BOOM_EVIDENCE_CATALOGS: tuple[str, ...] = (
    "TNS",
    "VSX",
    "milliquas_v8",
    "DESI_DR1",
    "NED",
)


class BoomCatalogEvidenceProvider:
    """Gather catalogue matches from BOOM for selected sky positions.

    The provider retrieves provider-native catalogue rows and preserves them as
    evidence. It intentionally does *not* map native catalogue values into an
    AppleCiDEr class or resolution status; that scientific policy belongs to a
    resolver.

    Parameters
    ----------
    coordinates_by_id
        Mapping from object ID to ``(ra_deg, dec_deg)``.

    catalogs
        BOOM catalogues to query. The default set contains the catalogues that
        produced label- or identity-relevant evidence in the historical audit.

    radius_arcsec
        Cone-search radius applied to every catalogue.

    query_chunk_size
        Number of selected positions sent in one BOOM cone-search request.

    client
        Optional reusable BOOM REST client. Injecting a client makes the
        provider straightforward to test and allows broker deployments to
        share transport configuration.
    """

    def __init__(
        self,
        *,
        coordinates_by_id: Mapping[
            str,
            tuple[float, float] | list[float],
        ],
        catalogs: Sequence[str] = DEFAULT_BOOM_EVIDENCE_CATALOGS,
        radius_arcsec: float = 5.0,
        query_chunk_size: int = 100,
        client: BoomClient | None = None,
    ) -> None:
        if radius_arcsec <= 0:
            raise ValueError("radius_arcsec must be > 0.")

        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be > 0.")

        normalized_catalogs = tuple(str(catalog).strip() for catalog in catalogs)

        if not normalized_catalogs:
            raise ValueError("catalogs must not be empty.")

        if any(not catalog for catalog in normalized_catalogs):
            raise ValueError("catalog names must not be empty.")

        if len(normalized_catalogs) != len(set(normalized_catalogs)):
            raise ValueError("catalogs must not contain duplicates.")

        normalized_coordinates: dict[
            str,
            tuple[float, float],
        ] = {}

        for object_id, position in coordinates_by_id.items():
            object_id = str(object_id).strip()

            if not object_id:
                raise ValueError("coordinates_by_id contains an empty object ID.")

            if len(position) != 2:
                raise ValueError("Each coordinate must contain exactly RA and Dec.")

            ra = float(position[0])
            dec = float(position[1])

            if not math.isfinite(ra) or not math.isfinite(dec):
                raise ValueError(f"Coordinates for {object_id!r} must be finite.")

            if dec < -90.0 or dec > 90.0:
                raise ValueError(f"Declination for {object_id!r} must lie in [-90, 90].")

            normalized_coordinates[object_id] = (
                ra % 360.0,
                dec,
            )

        self.coordinates_by_id = normalized_coordinates
        self.catalogs = normalized_catalogs
        self.radius_arcsec = float(radius_arcsec)
        self.query_chunk_size = int(query_chunk_size)
        self.client = client or BoomClient()

    @classmethod
    def from_source_index(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> "BoomCatalogEvidenceProvider":
        """Construct a provider from a BOOM pool ``source_index.json``."""

        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(payload, list):
            raise ValueError("BOOM source index must contain a JSON list.")

        coordinates: dict[
            str,
            tuple[float, float],
        ] = {}

        for row in payload:
            if not isinstance(row, Mapping):
                continue

            object_id = str(row.get("object_id") or "").strip()

            ra = row.get("ra")
            dec = row.get("dec")

            if not object_id or ra is None or dec is None:
                continue

            try:
                position = (
                    float(ra),
                    float(dec),
                )
            except (TypeError, ValueError):
                continue

            if not all(math.isfinite(value) for value in position):
                continue

            # One BOOM pool can contain multiple alerts for an object. Their
            # sky positions should be effectively identical for catalogue
            # association; retaining the first valid position is sufficient
            # and deterministic for this provider convenience constructor.
            coordinates.setdefault(
                object_id,
                position,
            )

        return cls(
            coordinates_by_id=coordinates,
            **kwargs,
        )

    def gather(
        self,
        selection: SelectionBatch,
    ) -> EvidenceBatch:
        """Cone-search configured BOOM catalogues for selected objects."""

        requested_ids = tuple(selection.ids)
        evidence: list[Evidence] = []
        failures: list[EvidenceFailure] = []

        missing_coordinates = [
            object_id for object_id in requested_ids if object_id not in self.coordinates_by_id
        ]

        for object_id in missing_coordinates:
            failures.append(
                EvidenceFailure(
                    object_id=object_id,
                    source="BOOM",
                    reason="missing_coordinates",
                )
            )

        queryable_ids = [
            object_id for object_id in requested_ids if object_id in self.coordinates_by_id
        ]

        for catalog in self.catalogs:
            for start in range(
                0,
                len(queryable_ids),
                self.query_chunk_size,
            ):
                object_ids = queryable_ids[start : start + self.query_chunk_size]

                if not object_ids:
                    continue

                coordinates = {
                    object_id: self.coordinates_by_id[object_id] for object_id in object_ids
                }

                try:
                    matches_by_id = self.client.cone_search(
                        catalog_name=catalog,
                        object_coordinates=coordinates,
                        radius_arcsec=self.radius_arcsec,
                    )
                except Exception as exc:
                    for object_id in object_ids:
                        failures.append(
                            EvidenceFailure(
                                object_id=object_id,
                                source=catalog,
                                reason="query_failed",
                                provenance={
                                    "provider": "BOOM",
                                    "error": repr(exc),
                                    "query_radius_arcsec": self.radius_arcsec,
                                },
                            )
                        )
                    continue

                for object_id in object_ids:
                    matches = matches_by_id.get(
                        object_id,
                        [],
                    )

                    for match_index, match in enumerate(matches):
                        separation_arcsec = self._match_separation_arcsec(
                            object_id=object_id,
                            match=match,
                        )

                        provenance: dict[str, Any] = {
                            "provider": "BOOM",
                            "catalog": catalog,
                            "match_index": match_index,
                            "query_radius_arcsec": self.radius_arcsec,
                        }

                        if separation_arcsec is not None:
                            provenance["separation_arcsec"] = separation_arcsec

                        evidence.append(
                            Evidence(
                                object_id=object_id,
                                source=catalog,
                                kind="catalog_match",
                                value=dict(match),
                                provenance=provenance,
                            )
                        )

        return EvidenceBatch(
            requested_ids=requested_ids,
            evidence=evidence,
            failures=failures,
        )

    def _match_separation_arcsec(
        self,
        *,
        object_id: str,
        match: Mapping[str, Any],
    ) -> float | None:
        for key in (
            "distance_arcsec",
            "separation_arcsec",
            "dist_arcsec",
            "distance",
        ):
            value = match.get(key)

            if value is None:
                continue

            try:
                separation = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(separation):
                return separation

        match_ra = match.get("ra")
        match_dec = match.get("dec")

        if match_ra is None or match_dec is None:
            return None

        try:
            source_ra, source_dec = self.coordinates_by_id[object_id]
            return self._angular_separation_arcsec(
                source_ra,
                source_dec,
                float(match_ra),
                float(match_dec),
            )
        except (TypeError, ValueError, KeyError):
            return None

    @staticmethod
    def _angular_separation_arcsec(
        ra1_deg: float,
        dec1_deg: float,
        ra2_deg: float,
        dec2_deg: float,
    ) -> float:
        """Return great-circle angular separation in arcseconds."""

        ra1 = math.radians(ra1_deg)
        dec1 = math.radians(dec1_deg)
        ra2 = math.radians(ra2_deg)
        dec2 = math.radians(dec2_deg)

        cosine = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(
            ra1 - ra2
        )

        angle = math.acos(max(-1.0, min(1.0, cosine)))

        return math.degrees(angle) * 3600.0
