"""Reusable BOOM REST client for adaptive data and evidence workflows."""

from __future__ import annotations

import json
import os

from collections.abc import Iterable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BOOM_API_URL = "https://babamul.umn.edu/api"


class BoomAPIError(RuntimeError):
    """Raised when the BOOM REST API request fails or is malformed."""


class BoomClient:
    """Small reusable client for the BOOM REST API.

    The client contains provider-specific transport/query mechanics only. It
    deliberately knows nothing about active-learning eligibility, scoring,
    label taxonomies, or scientific evidence interpretation.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BOOM_API_URL,
        token_env: str = "BABAMUL_API_TOKEN",
        query_batch_size: int = 500,
        timeout_seconds: float = 120.0,
    ) -> None:
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be > 0.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0.")

        self.base_url = str(base_url).rstrip("/")
        self.token_env = str(token_env)
        self.query_batch_size = int(query_batch_size)
        self.timeout_seconds = float(timeout_seconds)

    def post(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """POST one JSON request to BOOM and return the decoded mapping."""

        token = os.environ.get(self.token_env)

        if not token:
            raise RuntimeError(f"{self.token_env} is not set.")

        body = json.dumps(payload).encode("utf-8")

        request = Request(
            self.base_url + endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as handle:
                response = json.loads(handle.read().decode("utf-8"))

        except HTTPError as exc:
            try:
                detail = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                detail = ""

            raise BoomAPIError(f"BOOM API returned HTTP {exc.code}: {detail}") from exc

        except URLError as exc:
            raise BoomAPIError("Could not connect to the BOOM API.") from exc

        if not isinstance(response, dict):
            raise BoomAPIError("BOOM API returned an unexpected response.")

        return response

    def find(
        self,
        *,
        catalog_name: str,
        filter_document: Mapping[str, Any],
        limit: int,
        sort: Mapping[str, Any] | None = None,
        projection: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run BOOM's ``/queries/find`` endpoint."""

        payload: dict[str, Any] = {
            "catalog_name": catalog_name,
            "filter": dict(filter_document),
            "limit": int(limit),
        }

        if sort is not None:
            payload["sort"] = dict(sort)

        if projection is not None:
            payload["projection"] = dict(projection)

        response = self.post(
            "/queries/find",
            payload,
        )

        data = response.get("data")

        if not isinstance(data, list):
            raise BoomAPIError("BOOM find response does not contain a data list.")

        return data

    def find_by_ids(
        self,
        *,
        catalog_name: str,
        ids: Iterable[str | int],
        projection: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find rows by ``_id`` while respecting the configured chunk size."""

        ids = list(dict.fromkeys(ids))
        rows: list[dict[str, Any]] = []

        for start in range(
            0,
            len(ids),
            self.query_batch_size,
        ):
            batch = ids[start : start + self.query_batch_size]

            if not batch:
                continue

            rows.extend(
                self.find(
                    catalog_name=catalog_name,
                    filter_document={
                        "_id": {
                            "$in": batch,
                        }
                    },
                    limit=len(batch),
                    projection=projection,
                )
            )

        return rows

    def cone_search(
        self,
        *,
        catalog_name: str,
        object_coordinates: Mapping[
            str,
            tuple[float, float] | list[float],
        ],
        radius_arcsec: float = 5.0,
    ) -> dict[str, list[dict[str, Any]]]:
        """Cone-search one BOOM catalogue around named object positions."""

        radius_arcsec = float(radius_arcsec)

        if radius_arcsec <= 0:
            raise ValueError("radius_arcsec must be > 0.")

        coordinates = {
            str(object_id): [
                float(position[0]),
                float(position[1]),
            ]
            for object_id, position in object_coordinates.items()
        }

        response = self.post(
            "/queries/cone_search",
            {
                "catalog_name": str(catalog_name),
                "radius": radius_arcsec,
                "unit": "Arcseconds",
                "object_coordinates": coordinates,
            },
        )

        data = response.get("data")

        if not isinstance(data, Mapping):
            raise BoomAPIError("BOOM cone-search response data is not a mapping.")

        normalized: dict[str, list[dict[str, Any]]] = {}

        for object_id in coordinates:
            matches = data.get(object_id, [])

            if matches is None:
                matches = []

            if not isinstance(matches, list):
                raise BoomAPIError(
                    "BOOM cone-search matches must be lists; "
                    f"got {type(matches).__name__} for {object_id!r}."
                )

            if not all(isinstance(match, Mapping) for match in matches):
                raise BoomAPIError("BOOM cone-search matches must contain mappings.")

            normalized[object_id] = [dict(match) for match in matches]

        return normalized
