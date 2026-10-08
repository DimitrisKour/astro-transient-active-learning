#!/usr/bin/env python
"""Expand the labelled BOOM pool to every retrievable early-alert stage.

The source pool produced by notebook 12 contains one validated latest sample
per object. Notebook 11 cached every compact BOOM alert row and the complete
object-level auxiliary history. This script reconstructs all stages at or
before the source acquisition stage, constrained to the historical AstroMiNN
alert limit, and queries only the missing cutouts.

Without ``--materialize`` the script performs a read-only reconstruction audit
and prints the planned pool. ``--run-parity`` additionally verifies that
latest-alert inference is unchanged between the original and expanded pools.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest, CandidatePoolRecord
from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.boom.client import BoomClient
from al_pipeline.astronomy.ztf.packet import ZTFAstroMiNNAlertAdapter, ZTFPacketError


@dataclass(frozen=True, slots=True)
class StageContext:
    """One exact object/alert stage to materialize."""

    object_id: str
    alert: int
    candid: int
    jd: float
    packet: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or materialize every <=10 alert stage for the labelled BOOM/TNS AstroMiNN pool."
        )
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--profile", default="astrominn_ztf")
    parser.add_argument(
        "--audit-id",
        default="boom_new_labelled_2025_to_2026_v1",
    )
    parser.add_argument(
        "--source-pool-id",
        default="boom_tns_new_labelled_early10_v1",
    )
    parser.add_argument(
        "--source-run-id",
        default="boom_tns_new_labelled_inference_v1",
    )
    parser.add_argument(
        "--target-pool-id",
        default="boom_tns_new_labelled_early10_allavailable_v1",
    )
    parser.add_argument(
        "--parity-run-id",
        default="boom_tns_new_labelled_allavailable_parity_v1",
    )
    parser.add_argument("--source-audit-dir", type=Path)
    parser.add_argument("--source-pool", type=Path)
    parser.add_argument("--hidden-labels", type=Path)
    parser.add_argument("--target-pool", type=Path)
    parser.add_argument("--parity-output", type=Path)
    parser.add_argument("--max-alert", type=int, default=10)
    parser.add_argument("--cutout-batch-size", type=int, default=50)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--inference-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--parity-atol", type=float, default=1e-6)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Query missing cutouts and create the expanded pool.",
    )
    parser.add_argument(
        "--run-parity",
        action="store_true",
        help="Run baseline inference on both pools and compare latest logits.",
    )
    parser.add_argument(
        "--finalize-invalid-stages",
        action="store_true",
        help=(
            "Explicitly finalize after excluding stages whose cutouts fail "
            "the historical shape/finite-pixel contract. Rejections remain "
            "recorded in the ingestion report and rejected_stages.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_alert < 0:
        raise ValueError("max_alert must be non-negative.")

    if args.cutout_batch_size <= 0:
        raise ValueError("cutout_batch_size must be positive.")

    registry = DatasetRegistry.from_toml(args.registry)
    resolved = registry.resolve_profile(args.profile)
    source_audit_dir = args.source_audit_dir or (
        resolved.storage.data_root / "label_audits" / args.audit_id
    )
    source_pool = args.source_pool or (resolved.storage.data_root / "pools" / args.source_pool_id)
    hidden_labels_path = args.hidden_labels or (
        resolved.storage.persistent_root / "runs" / args.source_run_id / "hidden_labels.csv"
    )
    target_pool = args.target_pool or (resolved.storage.data_root / "pools" / args.target_pool_id)
    parity_output = args.parity_output or (
        resolved.storage.persistent_root / "runs" / args.parity_run_id
    )

    paths = {
        "alert_cache": source_audit_dir / "ztf_alert_rows.json",
        "aux_cache": source_audit_dir / "ztf_aux_histories.json",
        "source_manifest": source_pool / "candidate_pool.csv",
        "source_index": source_pool / "source_index.json",
        "hidden_labels": hidden_labels_path,
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    source_manifest = CandidatePoolManifest.from_csv(paths["source_manifest"])
    labels = load_hidden_labels(paths["hidden_labels"])
    source_index = load_source_index(paths["source_index"])
    alert_payload = json.loads(paths["alert_cache"].read_text(encoding="utf-8"))
    aux_payload = json.loads(paths["aux_cache"].read_text(encoding="utf-8"))
    adapter = ZTFAstroMiNNAlertAdapter(max_alert=args.max_alert)
    contexts, audit_rows = reconstruct_stage_contexts(
        object_ids=source_manifest.object_ids,
        source_index=source_index,
        alert_rows=alert_payload["rows"],
        aux_rows=aux_payload["rows"],
        adapter=adapter,
        max_alert=args.max_alert,
    )
    validate_contexts(
        source_manifest=source_manifest,
        contexts=contexts,
        labels=labels,
        source_index=source_index,
    )
    plan = build_plan(
        source_manifest=source_manifest,
        contexts=contexts,
        audit_rows=audit_rows,
        labels=labels,
        source_audit_dir=source_audit_dir,
        source_pool=source_pool,
        target_pool=target_pool,
        hidden_labels_path=hidden_labels_path,
        max_alert=args.max_alert,
        cache_paths=paths,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))

    if not args.materialize:
        print(
            "\nPRE-FLIGHT ONLY: add --materialize after the reconstruction audit has been reviewed."
        )

        if args.run_parity:
            raise ValueError("--run-parity requires --materialize.")

        return

    target_manifest_path = target_pool / "candidate_pool.csv"

    if not target_manifest_path.exists() and not os.environ.get("BABAMUL_API_TOKEN"):
        raise RuntimeError("BABAMUL_API_TOKEN is required to query earlier-stage cutouts.")

    target_manifest = materialize_pool(
        source_manifest=source_manifest,
        contexts=contexts,
        audit_rows=audit_rows,
        plan=plan,
        target_pool=target_pool,
        target_pool_id=args.target_pool_id,
        adapter=adapter,
        cutout_batch_size=args.cutout_batch_size,
        finalize_invalid_stages=args.finalize_invalid_stages,
    )
    print(
        "Expanded pool:",
        f"{len(target_manifest.object_ids):,} objects,",
        f"{len(target_manifest.records):,} samples",
    )

    if args.run_parity:
        run_latest_inference_parity(
            source_pool=source_pool,
            target_pool=target_pool,
            source_manifest=source_manifest,
            target_manifest=target_manifest,
            output_dir=parity_output,
            device=args.device,
            batch_size=args.inference_batch_size,
            num_workers=args.inference_workers,
            atol=args.parity_atol,
        )


def reconstruct_stage_contexts(
    *,
    object_ids: Sequence[str],
    source_index: Mapping[str, Mapping[str, object]],
    alert_rows: Sequence[Mapping[str, object]],
    aux_rows: Sequence[Mapping[str, object]],
    adapter: ZTFAstroMiNNAlertAdapter,
    max_alert: int,
) -> tuple[tuple[StageContext, ...], tuple[dict[str, object], ...]]:
    """Recover one latest candidate for every available alert number."""

    requested = set(object_ids)
    alerts_by_id: dict[str, list[Mapping[str, object]]] = {}

    for row in alert_rows:
        object_id = str(row.get("objectId") or "")

        if object_id in requested:
            alerts_by_id.setdefault(object_id, []).append(row)

    aux_by_id = {
        str(row["_id"]): row
        for row in aux_rows
        if row.get("_id") is not None and str(row["_id"]) in requested
    }
    contexts: list[StageContext] = []
    audits: list[dict[str, object]] = []

    for object_id in object_ids:
        acquisition = source_index.get(object_id)
        aux = aux_by_id.get(object_id)

        if acquisition is None:
            raise ValueError(f"Source index is missing object {object_id!r}.")

        if aux is None:
            raise ValueError(f"AUX cache is missing object {object_id!r}.")

        acquisition_alert = int(acquisition["alert"])
        acquisition_candid = int(acquisition["candid"])
        acquisition_jd = float(acquisition["jd"])
        by_alert: dict[int, StageContext] = {}
        invalid_messages: list[str] = []

        for row in alerts_by_id.get(object_id, ()):
            candidate = row.get("candidate")

            if not isinstance(candidate, Mapping):
                continue

            packet = {
                "objectId": object_id,
                "candidate": dict(candidate),
                "prv_candidates": aux.get("prv_candidates") or [],
            }

            try:
                alert = adapter.alert_number(packet)
                candid = int(candidate["candid"])
                jd = float(candidate["jd"])
            except (ZTFPacketError, KeyError, TypeError, ValueError) as exc:
                invalid_messages.append(str(exc))
                continue

            if alert > max_alert or alert > acquisition_alert:
                continue

            if jd > acquisition_jd + 1e-9:
                continue

            context = StageContext(
                object_id=object_id,
                alert=alert,
                candid=candid,
                jd=jd,
                packet=packet,
            )
            previous = by_alert.get(alert)

            if previous is None or context.jd > previous.jd:
                by_alert[alert] = context

        acquisition_context = by_alert.get(acquisition_alert)

        if acquisition_context is None:
            raise ValueError(f"Could not reconstruct acquisition stage for {object_id!r}.")

        if acquisition_context.candid != acquisition_candid:
            raise ValueError(
                f"Latest reconstructed candid changed for {object_id!r}: "
                f"{acquisition_context.candid} != {acquisition_candid}."
            )

        ordered = [by_alert[key] for key in sorted(by_alert)]
        contexts.extend(ordered)
        expected_alerts = set(range(acquisition_alert + 1))
        missing_alerts = sorted(expected_alerts.difference(by_alert))
        audits.append(
            {
                "object_id": object_id,
                "acquisition_alert": acquisition_alert,
                "acquisition_candid": acquisition_candid,
                "acquisition_jd": acquisition_jd,
                "cached_alert_rows": len(alerts_by_id.get(object_id, ())),
                "reconstructed_stages": len(ordered),
                "first_alert": ordered[0].alert,
                "last_alert": ordered[-1].alert,
                "missing_alerts": "|".join(map(str, missing_alerts)),
                "invalid_messages": " | ".join(dict.fromkeys(invalid_messages)),
            }
        )

    return tuple(contexts), tuple(audits)


def validate_contexts(
    *,
    source_manifest: CandidatePoolManifest,
    contexts: Sequence[StageContext],
    labels: Mapping[str, str],
    source_index: Mapping[str, Mapping[str, object]],
) -> None:
    object_ids = set(source_manifest.object_ids)

    if len(source_manifest.records) != len(object_ids):
        raise ValueError("The source pool must contain exactly one latest sample per object.")

    missing_source_files = source_manifest.missing_files()

    if missing_source_files:
        raise FileNotFoundError(
            f"The source pool references missing prepared samples: {missing_source_files[:3]}."
        )

    if object_ids != set(labels):
        raise ValueError("Source pool and hidden-label object sets do not agree.")

    if object_ids != set(source_index):
        raise ValueError("Source pool and source-index object sets do not agree.")

    keys = [(context.object_id, context.alert) for context in contexts]

    if len(keys) != len(set(keys)):
        raise ValueError("Reconstructed object/alert stages are not unique.")

    reconstructed_ids = {context.object_id for context in contexts}

    if reconstructed_ids != object_ids:
        raise ValueError("At least one source-pool object has no reconstructed stage.")


def build_plan(
    *,
    source_manifest: CandidatePoolManifest,
    contexts: Sequence[StageContext],
    audit_rows: Sequence[Mapping[str, object]],
    labels: Mapping[str, str],
    source_audit_dir: Path,
    source_pool: Path,
    target_pool: Path,
    hidden_labels_path: Path,
    max_alert: int,
    cache_paths: Mapping[str, Path],
) -> dict[str, object]:
    stages_per_object = Counter(context.object_id for context in contexts)
    samples_by_class = Counter(labels[context.object_id] for context in contexts)
    objects_by_class = Counter(labels.values())
    planned_alert_counts = Counter(context.alert for context in contexts)
    acquisition_alert_counts = Counter(int(row["acquisition_alert"]) for row in audit_rows)
    incomplete_histories = sum(bool(str(row["missing_alerts"])) for row in audit_rows)
    invalid_histories = sum(bool(str(row["invalid_messages"])) for row in audit_rows)
    noncontiguous_examples = [
        {
            "object_id": str(row["object_id"]),
            "acquisition_alert": int(row["acquisition_alert"]),
            "missing_alerts": str(row["missing_alerts"]),
        }
        for row in audit_rows
        if str(row["missing_alerts"])
    ][:20]
    invalid_examples = [
        {
            "object_id": str(row["object_id"]),
            "invalid_messages": str(row["invalid_messages"]),
        }
        for row in audit_rows
        if str(row["invalid_messages"])
    ][:20]
    counts = np.asarray(list(stages_per_object.values()), dtype=float)

    return {
        "source_pool_id": source_manifest.pool_id,
        "source_pool": str(source_pool),
        "target_pool": str(target_pool),
        "source_audit_dir": str(source_audit_dir),
        "hidden_labels": str(hidden_labels_path),
        "max_alert": max_alert,
        "history_policy": ("all_retrievable_complete_packets_at_or_before_acquisition"),
        "objects": len(source_manifest.object_ids),
        "source_samples": len(source_manifest.records),
        "planned_samples": len(contexts),
        "planned_samples_per_object": {
            "minimum": int(np.min(counts)),
            "median": float(np.median(counts)),
            "mean": float(np.mean(counts)),
            "maximum": int(np.max(counts)),
        },
        "objects_with_noncontiguous_cached_stages": incomplete_histories,
        "noncontiguous_stage_examples": noncontiguous_examples,
        "objects_with_invalid_cached_rows": invalid_histories,
        "invalid_cached_row_examples": invalid_examples,
        "acquisition_alert_counts": {
            str(alert): count for alert, count in sorted(acquisition_alert_counts.items())
        },
        "planned_alert_counts": {
            str(alert): count for alert, count in sorted(planned_alert_counts.items())
        },
        "object_label_counts": dict(sorted(objects_by_class.items())),
        "planned_sample_label_counts": dict(sorted(samples_by_class.items())),
        "input_sha256": {name: sha256_file(path) for name, path in cache_paths.items()},
    }


def materialize_pool(
    *,
    source_manifest: CandidatePoolManifest,
    contexts: Sequence[StageContext],
    audit_rows: Sequence[Mapping[str, object]],
    plan: Mapping[str, object],
    target_pool: Path,
    target_pool_id: str,
    adapter: ZTFAstroMiNNAlertAdapter,
    cutout_batch_size: int,
    finalize_invalid_stages: bool,
) -> CandidatePoolManifest:
    manifest_path = target_pool / "candidate_pool.csv"
    report_path = target_pool / "ingestion_report.json"
    index_path = target_pool / "source_index.json"
    audit_path = target_pool / "stage_reconstruction_audit.csv"
    rejected_path = target_pool / "rejected_stages.csv"
    identity_path = target_pool / "materialization_plan.json"
    samples_dir = target_pool / "samples"

    if manifest_path.exists():
        if not identity_path.exists():
            raise FileNotFoundError("Completed target pool is missing materialization_plan.json.")

        observed_identity = json.loads(identity_path.read_text(encoding="utf-8"))

        if observed_identity != dict(plan):
            raise ValueError("Existing expanded pool belongs to a different plan.")

        manifest = CandidatePoolManifest.from_csv(manifest_path)

        if manifest.pool_id != target_pool_id:
            raise ValueError("Existing target pool has a different pool ID.")

        expected = {(item.object_id, item.alert) for item in contexts}
        observed = {(record.object_id, record.alert) for record in manifest.records}

        if not report_path.exists():
            raise FileNotFoundError("Completed target pool is missing ingestion_report.json.")

        completed_report = json.loads(report_path.read_text(encoding="utf-8"))
        completion_policy = completed_report.get(
            "completion_policy",
            "strict_all_planned_stages",
        )
        rejected = {
            (str(row["object_id"]), int(row["alert"])) for row in completed_report.get("errors", [])
        }
        expected_observed = (
            expected.difference(rejected)
            if completion_policy == "exclude_invalid_cutout_stages"
            else expected
        )

        if expected_observed != observed:
            raise ValueError("Existing expanded manifest has a different stage set.")

        missing_files = manifest.missing_files()

        if missing_files:
            raise FileNotFoundError(
                f"Existing expanded pool references missing samples: {missing_files[:3]}."
            )

        print("Loaded completed expanded pool.")
        return manifest

    target_pool.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(plan)

    if identity_path.exists():
        observed_identity = json.loads(identity_path.read_text(encoding="utf-8"))

        if observed_identity != identity:
            raise ValueError("Partial target pool belongs to a different plan.")
    else:
        json_dump(identity, identity_path)

    write_dict_csv(audit_rows, audit_path)
    source_by_id = {record.object_id: record for record in source_manifest.records}
    context_by_candid = {context.candid: context for context in contexts}

    if len(context_by_candid) != len(contexts):
        raise ValueError("A candid was assigned to more than one stage.")

    records: dict[tuple[str, int], CandidatePoolRecord] = {}
    index_rows: dict[tuple[str, int], dict[str, object]] = {}
    pending_candids: list[int] = []
    copied_latest = 0
    resumed_samples = 0
    started = time.perf_counter()

    for context in contexts:
        key = (context.object_id, context.alert)
        sample_path = samples_dir / f"{context.object_id}_alert_{context.alert}.npy"

        if sample_path.exists():
            validate_sample(sample_path, context)
            resumed_samples += 1
        else:
            source_record = source_by_id[context.object_id]

            try:
                source_candid = int(source_record.source_reference.removeprefix("boom:"))
            except ValueError as exc:
                raise ValueError(
                    "Source record has an invalid BOOM reference: "
                    f"{source_record.source_reference!r}."
                ) from exc

            if source_record.alert == context.alert and source_candid == context.candid:
                copy_atomic(Path(source_record.filepath), sample_path)
                validate_sample(sample_path, context)
                copied_latest += 1
            else:
                pending_candids.append(context.candid)
                continue

        records[key] = record_from_context(context, sample_path)
        index_rows[key] = index_from_context(context)

    errors: list[dict[str, object]] = []

    if finalize_invalid_stages and report_path.exists():
        previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        previous_errors = previous_report.get("errors", [])
        previous_candids = {
            int(row["candid"]) for row in previous_errors if row.get("candid") is not None
        }
        invalid_only = all(
            row.get("reason") == "invalid_packet_or_cutout" for row in previous_errors
        )

        if previous_errors and invalid_only and previous_candids == set(pending_candids):
            errors = [dict(row) for row in previous_errors]
            pending_candids = []
            print(
                "Reusing audited invalid-stage failures:",
                len(errors),
            )

    client = BoomClient(query_batch_size=cutout_batch_size)

    for batch_number, candid_batch in enumerate(
        chunks(pending_candids, cutout_batch_size),
        start=1,
    ):
        cutout_rows = client.find_by_ids(
            catalog_name="ZTF_alerts_cutouts",
            ids=list(candid_batch),
            projection=None,
        )
        cutouts_by_id = {int(row["_id"]): row for row in cutout_rows if row.get("_id") is not None}

        for candid in candid_batch:
            context = context_by_candid[candid]
            cutouts = cutouts_by_id.get(candid)

            if cutouts is None:
                errors.append(
                    {
                        **index_from_context(context),
                        "reason": "missing_full_cutout",
                    }
                )
                continue

            complete_packet = {
                **context.packet,
                "cutoutScience": cutouts.get("cutoutScience"),
                "cutoutTemplate": cutouts.get("cutoutTemplate"),
                "cutoutDifference": cutouts.get("cutoutDifference"),
            }

            try:
                sample = adapter.prepare(complete_packet)
            except ZTFPacketError as exc:
                errors.append(
                    {
                        **index_from_context(context),
                        "reason": "invalid_packet_or_cutout",
                        "error": str(exc),
                    }
                )
                continue

            if sample.object_id != context.object_id or sample.alert != context.alert:
                raise RuntimeError(f"Prepared sample provenance changed for {context}.")

            sample_path = samples_dir / f"{context.object_id}_alert_{context.alert}.npy"
            save_sample_atomic(sample, sample_path)
            records[(context.object_id, context.alert)] = record_from_context(context, sample_path)
            index_rows[(context.object_id, context.alert)] = index_from_context(context)

        print(
            "Cutout batch",
            batch_number,
            "materialized:",
            f"{len(records):,}/{len(contexts):,}",
            "errors:",
            len(errors),
        )

    ordered_keys = [
        (context.object_id, context.alert)
        for context in contexts
        if (context.object_id, context.alert) in records
    ]
    report = {
        **identity,
        "completion_policy": (
            "exclude_invalid_cutout_stages"
            if errors and finalize_invalid_stages
            else "strict_all_planned_stages"
        ),
        "finalized_with_rejections": bool(errors and finalize_invalid_stages),
        "materialized_samples": len(records),
        "missing_samples": len(contexts) - len(records),
        "copied_latest_samples": copied_latest,
        "resumed_samples": resumed_samples,
        "queried_cutout_samples": (len(records) - copied_latest - resumed_samples),
        "sample_bytes": sum(Path(records[key].filepath).stat().st_size for key in ordered_keys),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "errors": errors,
    }
    json_dump(report, report_path)
    json_dump([index_rows[key] for key in ordered_keys], index_path)

    if errors:
        write_dict_csv(errors, rejected_path)

        if not finalize_invalid_stages:
            raise RuntimeError(
                f"{len(errors)} stages are missing or invalid. Partial "
                "samples were retained, but candidate_pool.csv was not "
                "written. Inspect ingestion_report.json."
            )

        unexpected_reasons = {
            str(row.get("reason"))
            for row in errors
            if row.get("reason") != "invalid_packet_or_cutout"
        }

        if unexpected_reasons:
            raise RuntimeError(
                "Explicit finalization only permits invalid cutout stages; "
                f"found reasons: {sorted(unexpected_reasons)}."
            )

    manifest = CandidatePoolManifest(
        pool_id=target_pool_id,
        records=[records[key] for key in ordered_keys],
    )

    if set(manifest.object_ids) != set(source_manifest.object_ids):
        raise RuntimeError("Expanded pool lost at least one source-pool object.")

    # Completion marker: always written last.
    manifest.to_csv(manifest_path)
    return manifest


def run_latest_inference_parity(
    *,
    source_pool: Path,
    target_pool: Path,
    source_manifest: CandidatePoolManifest,
    target_manifest: CandidatePoolManifest,
    output_dir: Path,
    device: str | None,
    batch_size: int,
    num_workers: int,
    atol: float,
) -> None:
    """Require expanded-pool latest predictions to equal source predictions."""

    import torch

    from al_pipeline.astronomy.astrominn.inference.run import run_inference

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    source = run_inference(
        model="astrominn_vit135",
        pool_path=source_pool,
        device=resolved_device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    target = run_inference(
        model="astrominn_vit135",
        pool_path=target_pool,
        device=resolved_device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    source_position = {object_id: index for index, object_id in enumerate(source.object_ids)}
    target_position = {object_id: index for index, object_id in enumerate(target.object_ids)}

    if set(source_position) != set(target_position):
        raise RuntimeError("Source and expanded inference object sets do not match.")

    source_records = {record.object_id: record for record in source_manifest.records}
    latest_target_records = {}

    for record in target_manifest.records:
        previous = latest_target_records.get(record.object_id)

        if previous is None or record.alert > previous.alert:
            latest_target_records[record.object_id] = record

    rows = []
    maximum_difference = 0.0
    mismatches = 0

    for object_id in source_manifest.object_ids:
        source_index = source_position[object_id]
        target_index = target_position[object_id]
        source_alert = source.alerts[source_index]
        target_alert = target.alerts[target_index]
        difference = float(
            np.max(
                np.abs(
                    source.predictions.values[source_index]
                    - target.predictions.values[target_index]
                )
            )
        )
        maximum_difference = max(maximum_difference, difference)
        source_record = source_records[object_id]
        target_record = latest_target_records[object_id]
        identical_sample = sha256_file(Path(source_record.filepath)) == sha256_file(
            Path(target_record.filepath)
        )
        passed = source_alert == target_alert and identical_sample and difference <= atol
        mismatches += int(not passed)
        rows.append(
            {
                "object_id": object_id,
                "source_alert": source_alert,
                "target_alert": target_alert,
                "identical_latest_sample": identical_sample,
                "max_abs_logit_difference": difference,
                "passed": passed,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(rows, output_dir / "latest_inference_parity.csv")
    summary = {
        "source_pool_id": source_manifest.pool_id,
        "target_pool_id": target_manifest.pool_id,
        "objects": len(rows),
        "source_samples": len(source_manifest.records),
        "target_samples": len(target_manifest.records),
        "device": resolved_device,
        "atol": atol,
        "maximum_absolute_logit_difference": maximum_difference,
        "mismatches": mismatches,
        "passed": mismatches == 0,
    }
    json_dump(summary, output_dir / "parity_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if mismatches:
        raise RuntimeError(f"Latest-alert parity failed for {mismatches} objects.")


def load_hidden_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if not {"object_id", "target_label"}.issubset(reader.fieldnames or ()):
            raise ValueError("Hidden labels require object_id and target_label columns.")

        labels = {str(row["object_id"]): str(row["target_label"]) for row in reader}

    if not labels:
        raise ValueError("Hidden-label table is empty.")

    return labels


def load_source_index(
    path: Path,
) -> dict[str, Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError("Source index must contain a list of rows.")

    return {str(row["object_id"]): row for row in payload}


def record_from_context(
    context: StageContext,
    sample_path: Path,
) -> CandidatePoolRecord:
    return CandidatePoolRecord(
        survey="ZTF",
        object_id=context.object_id,
        alert=context.alert,
        filepath=str(sample_path),
        source_reference=f"boom:{context.candid}",
    )


def index_from_context(context: StageContext) -> dict[str, object]:
    candidate = context.packet["candidate"]

    if not isinstance(candidate, Mapping):
        raise TypeError("Stage context candidate must be a mapping.")

    return {
        "object_id": context.object_id,
        "alert": context.alert,
        "candid": context.candid,
        "jd": context.jd,
        "ra": candidate.get("ra"),
        "dec": candidate.get("dec"),
        "source_reference": f"boom:{context.candid}",
    }


def validate_sample(path: Path, context: StageContext) -> None:
    try:
        sample = np.load(path, allow_pickle=True).item()
        valid = (
            str(sample.get("obj_id")) == context.object_id
            and int(sample.get("alerte")) == context.alert
            and np.asarray(sample["metadata"]).shape == (34,)
            and np.asarray(sample["image"]).shape == (3, 63, 63)
        )
    except Exception as exc:
        raise ValueError(f"Invalid prepared sample: {path}") from exc

    if not valid:
        raise ValueError(f"Prepared sample has inconsistent provenance or shape: {path}")


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    if temporary.exists():
        temporary.unlink()

    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)

    temporary.replace(destination)


def save_sample_atomic(sample: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("wb") as handle:
        np.save(
            handle,
            {
                "obj_id": sample.object_id,
                "alerte": sample.alert,
                "image": sample.image,
                "metadata": sample.metadata,
            },
            allow_pickle=True,
        )

    temporary.replace(path)


def chunks(values: Sequence[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def write_dict_csv(
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


if __name__ == "__main__":
    main()
