#!/usr/bin/env python3
"""Exploratory, read-only scalar temperature-calibration audit for AstroMiNN ViT.

Uses existing historical validation evaluation logits and, optionally, the
candidate-pool inference from the *same* checkpoint. Does not train a neural
network, rescore the actual experiment, access hidden U labels, or touch its
resumable outputs. Temperature is fit on a deterministic stratified subset of
validation objects, then audited on the disjoint remainder. IMPORTANT: the
original ViT checkpoint may already have been selected on full validation;
this is exploratory calibration evidence, NOT a fully independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# Must agree with src/al_pipeline/astronomy/astrominn/training.py, in index order.
ASTROMINN_CLASS_ORDER = ("AGN", "TDE", "SN I", "SN II", "CV")


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_validation_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluator stores latest-alert object logits in 'object_values'."""
    with np.load(path, allow_pickle=False) as data:
        required = {"object_ids", "object_targets", "object_values"}
        if not required.issubset(data.files):
            raise ValueError(f"Invalid evaluator NPZ; required keys: {sorted(required)}")
        ids = np.asarray(data["object_ids"]).astype(str)
        labels = np.asarray(data["object_targets"])
        logits = np.asarray(data["object_values"], dtype=np.float64)
    check_logits(ids, logits, labels)
    return ids, labels.astype(int), logits


def check_logits(ids: np.ndarray, logits: np.ndarray, labels: np.ndarray | None = None) -> None:
    if ids.ndim != 1 or not len(ids) or len(set(map(str, ids))) != len(ids):
        raise ValueError("Require unique, nonempty object IDs (one row per object)")
    if logits.ndim != 2 or logits.shape != (len(ids), len(ASTROMINN_CLASS_ORDER)):
        raise ValueError("Expected one object logit row per ID and exactly five ordered classes")
    if not np.all(np.isfinite(logits)):
        raise ValueError("Logits must be finite")
    if labels is not None:
        if labels.ndim != 1 or len(labels) != len(ids) or labels.dtype.kind not in "iu":
            raise ValueError("Targets must be integer indices aligned with object IDs")
        if not np.all((labels >= 0) & (labels < logits.shape[1])):
            raise ValueError("Invalid target class index")


def split_stratified(ids: np.ndarray, y: np.ndarray, *, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, row-order-invariant 2-way stratified validation split."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("Calibration fraction must lie strictly between 0 and 1")
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    calibration: list[int] = []
    audit: list[int] = []
    for cls in sorted(set(map(int, y))):
        group = sorted((int(i) for i in np.where(y == cls)[0]), key=lambda i: str(ids[i]))
        if len(group) < 2:
            raise ValueError(f"Class {cls} has fewer than 2 validation objects; cannot stratify")
        # Seed independently for each class to avoid sensitive ordering effects.
        rng = np.random.default_rng(np.random.SeedSequence([seed, cls, 9221]))
        order = rng.permutation(np.array(group, dtype=int)).tolist()
        n_cal = min(len(group) - 1, max(1, int(round(len(group) * fraction))))
        calibration.extend(order[:n_cal])
        audit.extend(order[n_cal:])
    return np.array(sorted(calibration), dtype=int), np.array(sorted(audit), dtype=int)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy(probs: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs > 0, np.log(probs), 0.0)
    return -np.sum(probs * logp, axis=1)


def ranked_indices(values: np.ndarray) -> np.ndarray:
    # Stable ties retain input row order, identical to deterministic top-k convention.
    return np.argsort(-values, kind="stable")


def pool_rank_changes(ids: np.ndarray, logits: np.ndarray, temperature: float, budget: int) -> dict:
    if budget <= 0 or budget > len(ids):
        raise ValueError("Budget must be 1..pool_size")
    ordinary = entropy(softmax(logits))
    calibrated = entropy(softmax(logits / temperature))
    a = ranked_indices(ordinary)
    b = ranked_indices(calibrated)
    selected_a = [str(ids[i]) for i in a[:budget]]
    selected_b = [str(ids[i]) for i in b[:budget]]
    overlap = len(set(selected_a) & set(selected_b))
    pos_a = np.empty(len(a), dtype=int)
    pos_b = np.empty(len(b), dtype=int)
    pos_a[a] = np.arange(len(a))
    pos_b[b] = np.arange(len(b))
    return {
        "pool_objects": len(ids), "budget": budget,
        "top_k_overlap": overlap, "top_k_overlap_fraction": overlap / budget,
        "replaced_objects": budget - overlap,
        "mean_absolute_rank_shift": float(np.mean(np.abs(pos_a - pos_b))),
        "mean_uncalibrated_entropy": float(np.mean(ordinary)),
        "mean_calibrated_entropy": float(np.mean(calibrated)),
        "selected_uncalibrated_ids": selected_a,
        "selected_calibrated_ids": selected_b,
    }


def read_pool_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if not {"object_ids", "values", "output_type"}.issubset(data.files):
            raise ValueError("Invalid saved closed-loop inference NPZ")
        if str(np.asarray(data["output_type"]).item()) != "logits":
            raise ValueError("Pool inference must contain raw logits (not probabilities)")
        ids = np.asarray(data["object_ids"]).astype(str)
        logits = np.asarray(data["values"], dtype=float)
    check_logits(ids, logits)
    return ids, logits


def verify_evaluation_checkpoint(evaluation_dir: Path) -> dict:
    evaluation = json.loads((evaluation_dir / "evaluation_summary.json").read_text())
    if tuple(evaluation.get("model_class_names", ())) != ASTROMINN_CLASS_ORDER:
        raise ValueError("Unexpected model class order in evaluation metadata")
    checkpoint = Path(evaluation["checkpoint_path"])
    if not checkpoint.is_file() or file_hash(checkpoint) != evaluation["checkpoint_sha256"]:
        raise ValueError("Evaluated checkpoint missing or hash differs from evaluation metadata")
    return {"checkpoint_path": str(checkpoint), "checkpoint_sha256": evaluation["checkpoint_sha256"]}


def verify_checkpoint_pair(evaluation_dir: Path, cycle_summary: Path) -> dict:
    identity = verify_evaluation_checkpoint(evaluation_dir)
    cycle = json.loads(cycle_summary.read_text())
    checkpoint = Path(identity["checkpoint_path"])
    inference_checkpoint = Path(cycle["inference_checkpoint"])
    if checkpoint != inference_checkpoint:
        raise ValueError(f"Mismatch: evaluation of {checkpoint}, pool inference by {inference_checkpoint}")
    return identity


def clean_metrics(metrics: dict) -> dict:
    result = {}
    for k, v in metrics.items():
        f = float(v)
        result[k] = f if math.isfinite(f) else None
    return result


def analyze(args: argparse.Namespace) -> dict:
    # Imports are intentionally lazy so static preflight/tests do not need Torch.
    from al_pipeline.core.types import ClassificationPredictions
    from al_pipeline.core.prediction.calibration import TemperatureScaler
    from al_pipeline.core.evaluation.metrics import compute_classification_metrics

    eval_dir = args.evaluation_dir.resolve()
    pred_path = eval_dir / "validation_predictions.npz"
    identity = verify_evaluation_checkpoint(eval_dir)
    ids, truth, logits = read_validation_npz(pred_path)
    cal_ix, audit_ix = split_stratified(ids, truth, fraction=args.fraction, seed=args.seed)
    labels = tuple(range(len(ASTROMINN_CLASS_ORDER)))
    scaler = TemperatureScaler.fit(logits[cal_ix], truth[cal_ix], class_labels=labels, split="calibration")

    metric_names = ("log_loss", "brier", "ece", "accuracy", "f1_macro", "balanced_accuracy")
    def measure(ix: np.ndarray) -> dict:
        raw = ClassificationPredictions(logits[ix], "logits")
        scaled = scaler.transform(raw, class_labels=labels)
        if not np.array_equal(np.argmax(raw.as_probabilities(), axis=1), np.argmax(scaled.values, axis=1)):
            raise ValueError("Scalar positive temperature unexpectedly changed argmax")
        return {
            "objects": len(ix),
            "per_class_counts": {name: int(np.sum(truth[ix] == j)) for j, name in enumerate(ASTROMINN_CLASS_ORDER)},
            "uncalibrated": clean_metrics(compute_classification_metrics(truth[ix], raw, labels=labels, metrics=metric_names)),
            "calibrated": clean_metrics(compute_classification_metrics(truth[ix], scaled, labels=labels, metrics=metric_names)),
        }

    result = {
        "schema": "astrominn_temperature_diagnostic_v1",
        "purpose": "offline exploratory diagnostic; NO checkpoint or AL pipeline mutation",
        "class_order": ASTROMINN_CLASS_ORDER,
        "inputs": {"validation_predictions": str(pred_path), "validation_sha256": file_hash(pred_path), **identity},
        "split": {"seed": args.seed, "calibration_fraction": args.fraction, "total_validation_objects": len(ids),
                  "calibration_ids_sha256": hashlib.sha256("\n".join(sorted(map(str, ids[cal_ix]))).encode()).hexdigest(),
                  "audit_ids_sha256": hashlib.sha256("\n".join(sorted(map(str, ids[audit_ix]))).encode()).hexdigest()},
        "temperature": scaler.to_dict(),
        "calibration_fit": measure(cal_ix),
        "heldout_validation_audit": measure(audit_ix),
        "limitations": [
            "The ViT checkpoint may have been selected using the original full validation split; this split is not independently untouched.",
            "Macro-F1 is not expected to change under a positive scalar temperature.",
            "The pool comparison is counterfactual; original acquisition and AL results remain unchanged.",
            "No test labels, hidden pool labels, or training data were used to fit temperature.",
        ],
    }
    if args.pool_inference is not None:
        if args.cycle_summary is None:
            raise ValueError("--cycle-summary is mandatory with --pool-inference for checkpoint provenance")
        result["inputs"].update(verify_checkpoint_pair(eval_dir, args.cycle_summary))
        result["inputs"].update({"pool_inference": str(args.pool_inference.resolve()),
                                 "pool_sha256": file_hash(args.pool_inference),
                                 "cycle_summary": str(args.cycle_summary.resolve()),
                                 "cycle_summary_sha256": file_hash(args.cycle_summary)})
        pool_ids, pool_logits = read_pool_npz(args.pool_inference)
        result["pool_entropy_comparison"] = pool_rank_changes(pool_ids, pool_logits, scaler.temperature, args.budget)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", type=Path, required=True,
                   help="Directory containing validation_predictions.npz and evaluation_summary.json")
    p.add_argument("--pool-inference", type=Path,
                   help="Optional matching AL cycle_###/inference.npz (must be raw logits)")
    p.add_argument("--cycle-summary", type=Path,
                   help="Corresponding persistent cycle_###/cycle_summary.json to verify inference checkpoint")
    p.add_argument("--budget", type=int, default=473)
    p.add_argument("--seed", type=int, default=135,
                   help="Seed for deterministic stratified calibration/audit split, not training seed")
    p.add_argument("--fraction", type=float, default=0.5,
                   help="Fraction of validation objects per class used to fit T")
    p.add_argument("--output", type=Path, required=True, help="Separate diagnostic JSON, never experiment output")
    args = p.parse_args()
    if args.pool_inference is None and args.cycle_summary is not None:
        p.error("--cycle-summary requires --pool-inference")
    if args.output.resolve().is_relative_to(args.evaluation_dir.resolve()):
        p.error("Output must be outside the immutable evaluation directory")
    result = analyze(args)
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != data:
        raise ValueError("Existing diagnostic differs; use a new --output path")
    if not args.output.exists():
        args.output.write_text(data)
    audit = result["heldout_validation_audit"]
    print(json.dumps({
        "temperature": result["temperature"]["temperature"],
        "validation_objects_fit": result["calibration_fit"]["objects"],
        "validation_objects_audit": audit["objects"],
        "audit_uncalibrated": audit["uncalibrated"],
        "audit_calibrated": audit["calibrated"],
        "pool_selection_changes": result.get("pool_entropy_comparison", {}).get("replaced_objects"),
        "saved_report": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
