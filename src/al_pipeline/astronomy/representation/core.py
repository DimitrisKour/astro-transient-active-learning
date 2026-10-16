"""Task 5A: object-level representation diagnostics, independent of AstroMiNN IO.

Only L0 is permitted for fitting a probe. Validation and test are report-only.
The feature extractor and potential fine-tuning strategies are interchangeable
at the representation boundary; no hidden U0 labels are required or accepted.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPRESENTATIONS = ("vit_cls", "lc_expert", "spatial_expert", "coord_expert", "expert_concat")
SPLITS = ("train", "validation", "test")


def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_object(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".tmp_", delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        finally:
            f.flush()
            os.fsync(f.fileno())
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".tmp_", suffix=".npz", delete=False) as f:
        temp = Path(f.name)
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def latest_rows(manifest: Path, object_column: str, label_column: str,
                label_map: Mapping[str, int], *, alert_column: str = "alert") -> list[dict[str, Any]]:
    """Exactly one latest alert per object; refuse duplicates and label conflicts.

    Latest-alert selection is equivalent to the production aggregation, which
    raises if two rows have the same maximum alert for an object.
    """
    latest: dict[str, dict[str, Any]] = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {object_column, label_column, alert_column}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(f"Missing columns {sorted(required)} in {manifest}")
        for index, row in enumerate(reader):
            obj = str(row[object_column]).strip()
            label = str(row[label_column]).strip()
            if not obj or label not in label_map:
                raise ValueError(f"Invalid object/label on row {index} in {manifest}")
            try:
                alert = int(row[alert_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid alert at row {index} of {manifest}") from exc
            if alert < 0:
                raise ValueError(f"Negative alert in {manifest}")
            item = {"object_id": obj, "target": int(label_map[label]), "alert": alert, "row": index}
            old = latest.get(obj)
            if old is not None:
                if item["target"] != old["target"]:
                    raise ValueError(f"Conflicting labels for {obj} in {manifest}")
                if alert == old["alert"]:
                    raise ValueError(f"Duplicate alert {alert} for {obj} in {manifest}")
            if old is None or alert > old["alert"]:
                latest[obj] = item
    if not latest:
        raise ValueError(f"Empty manifest: {manifest}")
    return sorted(latest.values(), key=lambda r: r["object_id"])


def stratified_subset(rows: list[dict[str, Any]], cap_per_class: int | None) -> list[dict[str, Any]]:
    """For smoke mode only; deterministic balanced selection using object IDs."""
    if cap_per_class is None:
        return rows
    if cap_per_class < 2:
        raise ValueError("cap_per_class must be >=2")
    by_class: dict[int, list[dict[str, Any]]] = {}
    for item in rows:
        by_class.setdefault(item["target"], []).append(item)
    selected = []
    for target, group in sorted(by_class.items()):
        group.sort(key=lambda r: hashlib.sha256(r["object_id"].encode()).digest())
        selected += group[:cap_per_class]
    return sorted(selected, key=lambda r: r["object_id"])


def shard_for(object_id: str, n_shards: int) -> int:
    if n_shards < 1:
        raise ValueError("n_shards must be positive")
    return int.from_bytes(hashlib.sha256(object_id.encode()).digest()[:8], "big") % n_shards


def validate_disjoint(groups: Mapping[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for name, rows in groups.items():
        for row in rows:
            obj = row["object_id"]
            if obj in seen:
                raise ValueError(f"Object leakage: {obj} in {seen[obj]} and {name}")
            seen[obj] = name


def describe_rows(rows: list[dict[str, Any]]) -> dict[str, object]:
    return {"objects": len(rows), "class_counts": dict(sorted(Counter(r["target"] for r in rows).items()))}


def _probe_matrix(x: np.ndarray, *, split: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError(f"Invalid {split} feature matrix: {x.shape}")
    return np.asarray(x, dtype=np.float64)


def fit_and_evaluate(train_x: np.ndarray, train_y: np.ndarray,
                     other: Mapping[str, tuple[np.ndarray, np.ndarray]],
                     *, names: tuple[str, ...], c: float = 1.0) -> dict[str, Any]:
    """Frozen hyperparameter, balanced logistic linear probe, fit only on L0.

    Validation/test are evaluated but never used for fitting or tuning.
    No test-derived selection is performed: all requested representations are
    reported, including failures and weaker probes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, log_loss, precision_recall_fscore_support)
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if c <= 0:
        raise ValueError("Regularization C must be positive")
    x = _probe_matrix(train_x, split="L0")
    y = np.asarray(train_y, dtype=np.int64)
    classes = np.arange(len(names), dtype=np.int64)
    if len(y) != len(x) or set(classes) - set(y):
        raise ValueError("L0 probe requires support for every target class")
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(
        C=c, max_iter=2000, class_weight="balanced", random_state=135))
    pipeline.fit(x, y)
    scaler = pipeline.named_steps["standardscaler"]
    reference_x = scaler.transform(x)
    output: dict[str, Any] = {"configuration": {
        "probe": "standard_scaler_plus_balanced_logistic_regression",
        "C": c, "fit_data": "historical_L0_only", "class_names": list(names),
        "aggregation": "single_latest_alert_per_object", "test_for_selection": False}, "splits": {}}
    for name, (features, true_labels) in {"train": (x, y), **other}.items():
        if name not in SPLITS:
            raise ValueError(f"Unknown split {name!r}")
        mat = _probe_matrix(features, split=name)
        target = np.asarray(true_labels, dtype=np.int64)
        if mat.shape[1] != x.shape[1] or len(target) != len(mat):
            raise ValueError(f"Incompatible {name} probe input")
        if set(target) - set(classes):
            raise ValueError(f"Unknown class in {name}")
        pred = pipeline.predict(mat)
        prob = pipeline.predict_proba(mat)
        prec, recall, f1, support = precision_recall_fscore_support(
            target, pred, labels=classes, zero_division=0)
        # k=1 neighborhood consistency relative to TRAIN objects only.
        # Train excludes self-neighbor; non-training splits never fit kNN.
        query = scaler.transform(mat)
        ref = reference_x
        k = 2 if name == "train" else 1
        if k > len(ref):
            raise ValueError("Not enough L0 objects for nearest-neighbor diagnostic")
        nearest = NearestNeighbors(n_neighbors=k).fit(ref)
        indices = nearest.kneighbors(query, return_distance=False)
        if name == "train":
            # Self-distance ties can make own index non-first; exclude explicitly.
            # Using all L0 neighbors is cheap for L0 and deterministic.
            distances = ((query[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
            np.fill_diagonal(distances, np.inf)
            neighbor_y = y[np.argmin(distances, axis=1)]
        else:
            neighbor_y = y[indices[:, 0]]
        output["splits"][name] = {
            "objects": len(target), "accuracy": float(accuracy_score(target, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
            "macro_f1": float(f1_score(target, pred, labels=classes, average="macro", zero_division=0)),
            "log_loss": float(log_loss(target, prob, labels=classes)),
            "nearest_train_object_label_agreement": float(np.mean(target == neighbor_y)),
            "confusion": confusion_matrix(target, pred, labels=classes).tolist(),
            "per_class": {class_name: {
                "precision": float(prec[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i, class_name in enumerate(names)},
        }
    return output
