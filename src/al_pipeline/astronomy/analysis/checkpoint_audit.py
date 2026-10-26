"""Strict, read-only AstroMiNN artifact adapter for retrospective checkpoint audits."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASS_NAMES = ("AGN", "TDE", "SN I", "SN II", "CV")
SUITES = {
    "task4": "astrominn_closed_loop_warm_v1",
    "task6b": "astrominn_closed_loop_warm_f1best_085_v1",
    "task6c": "astrominn_closed_loop_equal_object_loss_085_v1",
}
SEEDS = (135, 246, 357)
POLICIES = ("entropy", "entropy_balanced_explore")


@dataclass(frozen=True)
class Snapshot:
    suite_key: str
    suite: str
    policy: str
    seed: int
    cycle: int
    object_ids: np.ndarray
    truth: np.ndarray
    pred: np.ndarray
    checkpoint: str
    metrics: dict


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_snapshot(*, persistent: Path, data: Path, suite_key: str,
                  policy: str, seed: int, cycle: int) -> Snapshot:
    if suite_key not in SUITES or policy not in POLICIES or seed not in SEEDS or cycle not in range(10):
        raise ValueError("Unsupported suite, policy, seed, or cycle")
    suite = SUITES[suite_key]
    run = data / "runs" / suite / policy / f"seed_{seed}" / f"cycle_{cycle:03d}" / "evaluation"
    npz = run / "validation_predictions.npz"
    evaluation_file = run / "evaluation_summary.json"
    if not npz.is_file() or not evaluation_file.is_file():
        raise FileNotFoundError(f"Missing saved validation evaluation: {npz} or {evaluation_file}")
    with np.load(npz, allow_pickle=False) as archive:
        if not {"object_ids", "object_targets", "object_values"}.issubset(archive.files):
            raise ValueError(f"Missing object-level prediction fields: {npz}")
        ids = np.asarray(archive["object_ids"], dtype=str)
        y = np.asarray(archive["object_targets"], dtype=np.int64)
        logits = np.asarray(archive["object_values"], dtype=np.float64)
    if (y.shape != (len(ids),) or logits.shape != (len(ids), 5)
            or len(set(ids.tolist())) != len(ids) or not np.isfinite(logits).all()
            or np.any(y < 0) or np.any(y >= 5) or len(set(y.tolist())) != 5):
        raise ValueError(f"Invalid validation-object predictions: {npz}")
    evaluation = read_json(evaluation_file)
    if tuple(evaluation.get("model_class_names", ())) != CLASS_NAMES:
        raise ValueError(f"Model class-order mismatch: {evaluation_file}")
    splits = [split for split in evaluation.get("splits", ()) if split.get("split") == "validation"]
    if len(splits) != 1:
        raise ValueError(f"Expected exactly one validation summary: {evaluation_file}")
    expected_checkpoint = str(evaluation.get("checkpoint_path", ""))
    if not expected_checkpoint:
        raise ValueError(f"Checkpoint provenance missing: {evaluation_file}")
    if not Path(expected_checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint from evaluation no longer exists: {expected_checkpoint}")
    if cycle:
        state_file = persistent / "runs" / suite / policy / f"seed_{seed}" / "run_state.json"
        summary_file = persistent / "runs" / suite / policy / f"seed_{seed}" / f"cycle_{cycle:03d}" / "cycle_summary.json"
        if not state_file.is_file() or not summary_file.is_file():
            raise FileNotFoundError(f"Missing completed-cycle metadata: {summary_file}")
        state = read_json(state_file)
        if int(state.get("last_completed_cycle", -1)) < cycle:
            raise ValueError(f"Cycle {cycle} not complete in state: {state_file}")
        summary = read_json(summary_file)
        if (int(summary.get("cycle", -1)) != cycle
                or str(summary.get("trained_checkpoint", "")) != expected_checkpoint):
            raise ValueError(f"Evaluation/checkpoint ancestry mismatch: {summary_file}")
    order = np.argsort(ids)
    return Snapshot(suite_key, suite, policy, seed, cycle, ids[order], y[order],
                    np.argmax(logits[order], axis=1).astype(np.int64),
                    expected_checkpoint, splits[0]["object_metrics"])


def ensure_alignment(incumbent: Snapshot, candidate: Snapshot) -> None:
    if not np.array_equal(incumbent.object_ids, candidate.object_ids):
        raise ValueError("Validation object IDs differ between checkpoints")
    if not np.array_equal(incumbent.truth, candidate.truth):
        raise ValueError("Validation truth labels differ between checkpoints")


def acquisition_diagnostics(*, persistent: Path, suite_key: str,
                            policy: str, seed: int, cycle: int) -> dict:
    """Label counts and alert multiplicity for the saved, actually acquired batch."""
    suite = SUITES[suite_key]
    run = persistent / "runs" / suite / policy / f"seed_{seed}"
    summary = read_json(run / f"cycle_{cycle:03d}" / "cycle_summary.json")
    labels = summary.get("revealed_labels")
    if not isinstance(labels, dict) or any(label not in CLASS_NAMES for label in labels.values()):
        raise ValueError(f"Missing or invalid revealed labels: {run}")
    if len(labels) != int(summary.get("acquisition_size", -1)):
        raise ValueError(f"Acquisition count mismatch: {run}")
    counts = {label: sum(y == label for y in labels.values()) for label in CLASS_NAMES}
    manifest = run / "manifests" / f"acquired_cycle_{cycle:03d}.csv"
    alerts_per_object = {}
    manifest_status = "missing"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"Invalid acquired manifest {manifest}")
            rows = list(reader)
            # Infer rather than assume the historical registry's object-id column.
            recognized = [c for c in ("object_id", "obj_id", "objectId", "objectid", "oid")
                          if c in reader.fieldnames]
            recognized += [c for c in reader.fieldnames if c not in recognized]
            for column in recognized:
                values = [row[column] for row in rows]
                if set(values) == set(labels) and all(values):
                    from collections import Counter
                    alerts_per_object = dict(Counter(values))
                    manifest_status = "verified"
                    break
            if not alerts_per_object:
                manifest_status = "unrecognized_object_id_column"
    multiplicities = list(alerts_per_object.values())
    return {"suite_key": suite_key, "policy": policy, "seed": seed, "cycle": cycle,
            "n_acquired": len(labels),
            **{f"acquired_{c.replace(' ', '_')}": n for c, n in counts.items()},
            "manifest_status": manifest_status,
            "mean_alerts_per_acquired_object": float(np.mean(multiplicities)) if multiplicities else None,
            "max_alerts_per_acquired_object": int(max(multiplicities)) if multiplicities else None,
            "n_manifest_rows": int(sum(multiplicities)) if multiplicities else None,
            "available_before_selection": summary.get("available_before_selection"),
            "available_after_selection": summary.get("available_after_selection")}
