"""BOOM REST source for finite ZTF active-learning candidate pools."""

from __future__ import annotations

import json
import time

from collections.abc import (
    Collection,
    Iterable,
    Mapping,
)
from pathlib import Path
from typing import Any
import numpy as np

from al_pipeline.core.eligibility.base import EligibilityRule
from al_pipeline.astronomy.boom.client import BoomAPIError, BoomClient, DEFAULT_BOOM_API_URL
from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest, CandidatePoolRecord
from al_pipeline.astronomy.data.source import CandidateSource
from al_pipeline.astronomy.ztf.packet import ZTFAstroMiNNAlertAdapter, ZTFPacketError


class BoomZTFSource(CandidateSource):
    """Materialize a finite BOOM ZTF query as an unlabeled pool.

    The BOOM main API stores one logical ZTF alert across three catalogs:

    - ZTF_alerts
    - ZTF_alerts_aux
    - ZTF_alerts_cutouts

    They are joined through objectId and candid before provider-neutral
    ZTF preprocessing is applied.

    Full BOOM payloads are optional because historical auxiliary rows can
    be very large. A compact source index is always persisted so accepted
    samples can be traced back to their BOOM alert and sky position.
    """

    def __init__(
        self,
        *,
        limit: int,
        start_jd: float | None = None,
        end_jd: float | None = None,
        excluded_object_ids: Collection[str] = (),
        max_alert: int = 10,
        base_url: str = DEFAULT_BOOM_API_URL,
        token_env: str = "BABAMUL_API_TOKEN",
        query_batch_size: int = 500,
        save_raw_payloads: bool = False,
        eligibility_policy: (EligibilityRule[Mapping[str, Any]] | None) = None,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be > 0.")

        if not isinstance(
            save_raw_payloads,
            bool,
        ):
            raise TypeError("save_raw_payloads must be a boolean.")

        if start_jd is not None:
            start_jd = float(start_jd)

        if end_jd is not None:
            end_jd = float(end_jd)

        if start_jd is not None and end_jd is not None and start_jd >= end_jd:
            raise ValueError("start_jd must be smaller than end_jd.")

        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be > 0.")

        self.limit = int(limit)
        self.start_jd = start_jd
        self.end_jd = end_jd
        self.max_alert = int(max_alert)

        self.base_url = str(base_url).rstrip("/")

        self.token_env = str(token_env)

        self.query_batch_size = int(query_batch_size)

        self.client = BoomClient(
            base_url=self.base_url,
            token_env=self.token_env,
            query_batch_size=self.query_batch_size,
        )

        self.save_raw_payloads = save_raw_payloads

        self.eligibility_policy = eligibility_policy

        self.excluded_object_ids = frozenset(str(object_id) for object_id in excluded_object_ids)

        self.adapter = ZTFAstroMiNNAlertAdapter(max_alert=max_alert)

        self.last_report: dict[str, Any] | None = None

    @property
    def source_name(self) -> str:
        return "boom_ztf_rest"

    def _fetch(
        self,
        *,
        pool_id: str,
        destination: Path,
    ) -> CandidatePoolManifest:
        total_start = time.perf_counter()
        timings: dict[str, float] = {}

        samples_dir = destination / "samples"

        samples_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        alert_filter = self._alert_filter()

        step_start = time.perf_counter()
        alerts = self._find(
            catalog_name="ZTF_alerts",
            filter_document=alert_filter,
            limit=self.limit,
            sort={
                "candidate.jd": -1,
            },
        )
        timings["alerts_query"] = time.perf_counter() - step_start

        # TODO: Add stable cursor-based pagination if BOOM pools need to
        # exhaust large time windows. Repeating the same start_jd/end_jd and
        # limit currently returns the same newest alerts because results are
        # sorted by candidate.jd descending. For finite experiments, use
        # explicit non-overlapping JD windows. A future cursor should use a
        # stable ordering such as (candidate.jd, candidate.candid), rather
        # than JD alone, to avoid skipping alerts that share a timestamp.

        alert_jds = [
            float(candidate["jd"])
            for alert in alerts
            if isinstance(
                (candidate := alert.get("candidate")),
                Mapping,
            )
            and candidate.get("jd") is not None
        ]

        report: dict[str, Any] = {
            "source": self.source_name,
            "pool_id": pool_id,
            "requested": self.limit,
            "start_jd": self.start_jd,
            "end_jd": self.end_jd,
            "max_alert": self.max_alert,
            "save_raw_payloads": (self.save_raw_payloads),
            "total": len(alerts),
            "returned_first_jd": (min(alert_jds) if alert_jds else None),
            "returned_last_jd": (max(alert_jds) if alert_jds else None),
            "accepted": 0,
            "late_alert": 0,
            "historical_overlap": 0,
            "eligibility_rejected": 0,
            "eligibility_reasons": {},
            "duplicate_sample": 0,
            "missing_aux": 0,
            "missing_cutouts": 0,
            "invalid_cutout": 0,
            "invalid_packet": 0,
            "errors": [],
        }

        # Apply every eligibility check that needs only the alert row first.
        # This avoids AUX/cutout requests for known historical objects and
        # malformed candidates.
        step_start = time.perf_counter()

        eligibility_rejections: list[dict[str, Any]] = []

        eligible_alerts: list[
            tuple[
                dict[str, Any],
                str,
                Mapping[str, Any],
                int,
            ]
        ] = []

        for alert in alerts:
            object_id = str(alert.get("objectId") or "")

            candidate = alert.get("candidate")

            if not object_id or not isinstance(
                candidate,
                Mapping,
            ):
                self._record_error(
                    report,
                    reason="invalid_packet",
                    object_id=object_id,
                    message=("BOOM alert is missing objectId or candidate."),
                )
                continue

            candid_value = candidate.get("candid")

            if candid_value is None:
                self._record_error(
                    report,
                    reason="invalid_packet",
                    object_id=object_id,
                    message=("BOOM candidate is missing candid."),
                )
                continue

            try:
                candid = int(candid_value)
            except (
                TypeError,
                ValueError,
            ):
                self._record_error(
                    report,
                    reason="invalid_packet",
                    object_id=object_id,
                    message=("BOOM candidate candid must be an integer."),
                )
                continue

            if object_id in self.excluded_object_ids:
                report["historical_overlap"] += 1
                continue

            if self.eligibility_policy is not None:
                decision = self.eligibility_policy.evaluate(candidate)

                if not decision.eligible:
                    reason = str(decision.reason)

                    report["eligibility_rejected"] += 1

                    reason_counts = report["eligibility_reasons"]
                    reason_counts[reason] = (
                        reason_counts.get(
                            reason,
                            0,
                        )
                        + 1
                    )

                    eligibility_rejections.append(
                        {
                            "object_id": object_id,
                            "candid": candid,
                            "jd": candidate.get("jd"),
                            "reason": reason,
                            "source": (decision.source),
                            "evidence": dict(decision.evidence),
                        }
                    )
                    continue

            eligible_alerts.append(
                (
                    alert,
                    object_id,
                    candidate,
                    candid,
                )
            )

        object_ids = list(
            dict.fromkeys(
                object_id
                for (
                    _,
                    object_id,
                    _,
                    _,
                ) in eligible_alerts
            )
        )

        timings["alert_prefilter"] = time.perf_counter() - step_start

        # In normal experiment mode, request only the AUX fields needed by
        # the AstroMiNN adapter. Historical AUX documents can otherwise be
        # extremely large. Debug mode intentionally keeps the full AUX rows
        # so older/raw-data diagnostic notebooks retain their original data.
        aux_projection = (
            None
            if self.save_raw_payloads
            else {
                "_id": 1,
                "prv_candidates": 1,
            }
        )

        step_start = time.perf_counter()
        aux_rows = self._find_by_ids(
            catalog_name=("ZTF_alerts_aux"),
            ids=object_ids,
            projection=aux_projection,
        )
        timings["aux_query"] = time.perf_counter() - step_start

        step_start = time.perf_counter()

        aux_by_id = {str(row["_id"]): row for row in aux_rows if row.get("_id") is not None}

        # Alert-number eligibility needs AUX history but not image cutouts.
        # Reject late candidates before requesting the heavier cutout rows.
        cutout_candidates: list[
            tuple[
                dict[str, Any],
                int,
                int,
            ]
        ] = []

        for (
            _alert,
            object_id,
            candidate,
            candid,
        ) in eligible_alerts:
            aux = aux_by_id.get(object_id)

            if aux is None:
                self._record_error(
                    report,
                    reason="missing_aux",
                    object_id=object_id,
                    message=("No matching ZTF_alerts_aux row."),
                )
                continue

            packet = {
                "objectId": object_id,
                "candidate": dict(candidate),
                "prv_candidates": (aux.get("prv_candidates") or []),
            }

            try:
                alert_number = self.adapter.alert_number(packet)

            except ZTFPacketError as exc:
                self._record_error(
                    report,
                    reason="invalid_packet",
                    object_id=object_id,
                    message=str(exc),
                )
                continue

            if alert_number > self.adapter.max_alert:
                report["late_alert"] += 1
                continue

            cutout_candidates.append(
                (
                    packet,
                    candid,
                    alert_number,
                )
            )

        candids = list(
            dict.fromkeys(
                candid
                for (
                    _,
                    candid,
                    _,
                ) in cutout_candidates
            )
        )

        timings["alert_history_filter"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        cutout_rows = self._find_by_ids(
            catalog_name=("ZTF_alerts_cutouts"),
            ids=candids,
        )
        timings["cutout_query"] = time.perf_counter() - step_start

        cutouts_by_id = {int(row["_id"]): row for row in cutout_rows if row.get("_id") is not None}

        report["aux_objects_requested"] = len(object_ids)

        report["aux_projection"] = "full" if aux_projection is None else list(aux_projection.keys())

        report["cutout_alerts_requested"] = len(candids)

        # Full raw BOOM payloads are useful for debugging but can consume
        # several GB for a modest historical query. Normal experiment runs
        # therefore persist only canonical samples, the compact source index,
        # manifest, and ingestion report.
        step_start = time.perf_counter()

        if self.save_raw_payloads:
            raw_dir = destination / "raw"

            raw_cutouts_dir = raw_dir / "cutouts"

            raw_cutouts_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._write_json(
                raw_dir / "alerts.json",
                alerts,
            )

            self._write_json(
                raw_dir / "aux.json",
                aux_rows,
            )

            self._write_raw_cutouts(
                cutouts_by_id,
                raw_cutouts_dir,
            )

        timings["raw_payload_write"] = time.perf_counter() - step_start

        step_start = time.perf_counter()

        records: list[CandidatePoolRecord] = []

        source_index: list[dict[str, Any]] = []

        sample_bytes = 0

        seen: set[tuple[str, int]] = set()

        for (
            packet,
            candid,
            alert_number,
        ) in cutout_candidates:
            object_id = str(packet["objectId"])

            key = (
                object_id,
                alert_number,
            )

            if key in seen:
                report["duplicate_sample"] += 1
                continue

            cutouts = cutouts_by_id.get(candid)

            if cutouts is None:
                self._record_error(
                    report,
                    reason="missing_cutouts",
                    object_id=object_id,
                    message=("No matching ZTF_alerts_cutouts row."),
                )
                continue

            complete_packet = {
                **packet,
                "cutoutScience": (cutouts.get("cutoutScience")),
                "cutoutTemplate": (cutouts.get("cutoutTemplate")),
                "cutoutDifference": (cutouts.get("cutoutDifference")),
            }

            try:
                sample = self.adapter.prepare(complete_packet)

            except ZTFPacketError as exc:
                message = str(exc)

                reason = "invalid_cutout" if "cutout" in (message.lower()) else "invalid_packet"

                self._record_error(
                    report,
                    reason=reason,
                    object_id=object_id,
                    message=message,
                )
                continue

            sample_path = samples_dir / (f"{sample.object_id}_alert_{sample.alert}.npy")

            np.save(
                sample_path,
                {
                    "obj_id": (sample.object_id),
                    "alerte": (sample.alert),
                    "image": (sample.image),
                    "metadata": (sample.metadata),
                },
                allow_pickle=True,
            )

            sample_bytes += sample_path.stat().st_size

            source_reference = f"boom:{candid}"

            records.append(
                CandidatePoolRecord(
                    survey="ZTF",
                    object_id=(sample.object_id),
                    alert=sample.alert,
                    filepath=str(sample_path),
                    source_reference=(source_reference),
                )
            )

            candidate = packet["candidate"]

            source_index.append(
                {
                    "object_id": (sample.object_id),
                    "candid": candid,
                    "alert": (sample.alert),
                    "jd": (candidate.get("jd")),
                    "ra": (candidate.get("ra")),
                    "dec": (candidate.get("dec")),
                    "source_reference": (source_reference),
                }
            )

            seen.add(key)

        timings["preprocess_and_sample_write"] = time.perf_counter() - step_start

        if not records:
            raise RuntimeError("BOOM query produced no usable candidate samples.")

        report["accepted"] = len(records)

        report["rejected"] = report["total"] - report["accepted"]

        manifest = CandidatePoolManifest(
            pool_id=pool_id,
            records=records,
        )

        step_start = time.perf_counter()

        manifest_path = destination / "candidate_pool.csv"
        manifest.to_csv(manifest_path)

        source_index_path = destination / "source_index.json"
        self._write_json(
            source_index_path,
            source_index,
        )

        eligibility_path = destination / "eligibility_rejections.json"
        self._write_json(
            eligibility_path,
            eligibility_rejections,
        )

        timings["manifest_and_index_write"] = time.perf_counter() - step_start

        report["storage_bytes"] = {
            "samples": sample_bytes,
            "candidate_pool_csv": (manifest_path.stat().st_size),
            "source_index_json": (source_index_path.stat().st_size),
            "eligibility_rejections_json": (eligibility_path.stat().st_size),
        }

        timings["total_before_report_write"] = time.perf_counter() - total_start

        report["timings_seconds"] = {name: round(value, 6) for name, value in timings.items()}

        self._write_json(
            destination / "ingestion_report.json",
            report,
        )

        self.last_report = report

        return manifest

    def _find(
        self,
        *,
        catalog_name: str,
        filter_document: Mapping[str, Any],
        limit: int,
        sort: Mapping[str, Any] | None = None,
        projection: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "catalog_name": (catalog_name),
            "filter": dict(filter_document),
            "limit": int(limit),
        }

        if sort is not None:
            payload["sort"] = dict(sort)

        if projection is not None:
            payload["projection"] = dict(projection)

        response = self._post(
            "/queries/find",
            payload,
        )

        data = response.get("data")

        if not isinstance(
            data,
            list,
        ):
            raise BoomAPIError("BOOM find response does not contain a data list.")

        return data

    def _find_by_ids(
        self,
        *,
        catalog_name: str,
        ids: Iterable[str | int],
        projection: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
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
                self._find(
                    catalog_name=(catalog_name),
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

    def _alert_filter(
        self,
    ) -> dict[str, Any]:
        """Build the BOOM filter for the configured alert time window."""

        jd_filter: dict[
            str,
            float,
        ] = {}

        if self.start_jd is not None:
            jd_filter["$gte"] = self.start_jd

        if self.end_jd is not None:
            jd_filter["$lt"] = self.end_jd

        if not jd_filter:
            return {}

        return {
            "candidate.jd": (jd_filter),
        }

    def _post(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Delegate BOOM transport to the reusable REST client."""

        return self.client.post(
            endpoint,
            payload,
        )

    @staticmethod
    def _write_json(
        path: Path,
        value: Any,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(path.suffix + ".tmp")

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        temporary.replace(path)

    @staticmethod
    def _write_raw_cutouts(
        cutouts_by_id: Mapping[
            int,
            Mapping[str, Any],
        ],
        destination: Path,
    ) -> None:
        names = {
            "cutoutScience": ("science"),
            "cutoutTemplate": ("template"),
            "cutoutDifference": ("difference"),
        }

        for candid, row in cutouts_by_id.items():
            for field, suffix in names.items():
                value = row.get(field)

                if value is None:
                    continue

                try:
                    content = bytes(value)
                except Exception:
                    continue

                path = destination / (f"{candid}_{suffix}.fits.gz")

                path.write_bytes(content)

    @staticmethod
    def _record_error(
        report: dict[str, Any],
        *,
        reason: str,
        object_id: str,
        message: str,
    ) -> None:
        report[reason] += 1

        report["errors"].append(
            {
                "object_id": (object_id),
                "reason": reason,
                "message": message,
            }
        )


# TODO: If AUX transfer remains a bottleneck after projection, investigate
# whether BOOM can server-side trim/filter prv_candidates itself without
# changing the historical alert semantics. Do not assume ordering or slice
# semantics without validating them against full AUX histories.

# TODO: Add a Kafka-backed BOOM source only if continuous online active
# learning becomes necessary. Finite REST-backed pools are sufficient for
# the current batch active-learning experiments.
