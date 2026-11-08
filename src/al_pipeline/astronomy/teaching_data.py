"""Lightweight, CPU-only datasets for the AppleCiDEr teaching example.

This module deliberately imports neither PyTorch nor archived model sources.
The generated fallback is *synthetic* and is never presented as ZTF data.
Real exports use exactly the same NPZ schema and pass strict validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

CLASS_NAMES = ("AGN", "TDE", "SN I", "SN II", "CV")
REQUIRED_KEYS = frozenset({
    "schema_version", "features", "labels", "object_ids", "splits",
    "alert_counts", "preview_indices", "preview_images", "feature_source", "provenance",
})


def validate_teaching_data(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate the same no-pickle, object-level contract for real/demo data."""
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"Teaching dataset missing keys: {sorted(missing)}")
    if int(np.asarray(data["schema_version"]).item()) != 1:
        raise ValueError("Unsupported teaching dataset schema version")
    x = np.asarray(data["features"], dtype=np.float32)
    y = np.asarray(data["labels"], dtype=np.int32)
    ids = np.asarray(data["object_ids"]).astype(str)
    splits = np.asarray(data["splits"]).astype(str)
    alerts = np.asarray(data["alert_counts"], dtype=np.int32)
    ix = np.asarray(data["preview_indices"], dtype=np.int32)
    thumbs = np.asarray(data["preview_images"], dtype=np.float32)
    provenance = str(np.asarray(data["provenance"]).item())
    feature_source = str(np.asarray(data["feature_source"]).item())
    n = len(y)
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] < 2:
        raise ValueError("features must have shape (n_objects, n_features>=2)")
    if y.ndim != 1 or ids.shape != (n,) or splits.shape != (n,) or alerts.shape != (n,):
        raise ValueError("Object-level arrays must have the same one-dimensional length")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite features")
    if not set(np.unique(y)).issubset(set(range(len(CLASS_NAMES)))):
        raise ValueError("Unknown target class index")
    if len(set(ids)) != n or not all(ids):
        raise ValueError("Object IDs must be unique and nonempty across all splits")
    if not set(splits).issubset({"train", "validation", "test"}):
        raise ValueError("Unknown split names")
    if not all(np.any(splits == part) for part in ("train", "validation", "test")):
        raise ValueError("Missing at least one train/validation/test split")
    if np.any(alerts < 1):
        raise ValueError("Every object needs at least one alert")
    if ix.ndim != 1 or thumbs.ndim != 4 or thumbs.shape[0] != len(ix) or thumbs.shape[1] != 3:
        raise ValueError("preview_images must have shape (n_previews, 3, height, width)")
    if thumbs.shape[2] < 4 or thumbs.shape[3] < 4 or not np.isfinite(thumbs).all():
        raise ValueError("Invalid preview image values or dimensions")
    if np.any(ix < 0) or np.any(ix >= n) or len(set(ix.tolist())) != len(ix):
        raise ValueError("Preview indices must be unique valid row indices")
    if not provenance or not feature_source:
        raise ValueError("Data provenance and feature source must be documented")
    if set(np.unique(y[splits == "train"])) != set(range(len(CLASS_NAMES))):
        raise ValueError("All five classes are required for the train split")
    if set(np.unique(y[splits == "validation"])) != set(range(len(CLASS_NAMES))):
        raise ValueError("All five classes are required for the validation split")
    return {
        "schema_version": np.array(1), "features": x, "labels": y,
        "object_ids": ids, "splits": splits, "alert_counts": alerts,
        "preview_indices": ix, "preview_images": thumbs,
        "feature_source": np.array(feature_source), "provenance": np.array(provenance),
    }


def load_teaching_data(path: str | Path) -> dict[str, np.ndarray]:
    """Load a compact benchmark without executing pickle payloads."""
    with np.load(path, allow_pickle=False) as container:
        payload = {name: container[name] for name in container.files}
    return validate_teaching_data(payload)


def make_synthetic_demo(*, seed: int = 2026) -> dict[str, np.ndarray]:
    """Generate explicitly artificial five-class features for testing/preview.

    Class labels, alert counts, and image stamps here are simulated; they
    cannot be used to evaluate any real astronomical model or acquisition policy.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(CLASS_NAMES)
    # An imbalanced but controlled benchmark; no real population proportions.
    class_counts = [190, 40, 490, 185, 95]
    d = 52
    means = rng.normal(size=(n_classes, d)).astype(np.float32)
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    means *= 2.8
    # Bring two supernova types closer together to produce informative errors.
    means[3] = 0.78 * means[2] + 0.22 * means[3]
    parts = []
    for c, count in enumerate(class_counts):
        x = means[c] + rng.normal(scale=1.25, size=(count, d))
        x[:, 0] += 0.4 * (c == 1)
        parts.append((x.astype(np.float32), np.full(count, c, np.int32)))
    x = np.concatenate([a for a, _ in parts])
    y = np.concatenate([b for _, b in parts])
    ids = np.asarray([f"synthetic-{i:05d}" for i in range(len(y))], dtype=str)
    perm = rng.permutation(len(y))
    x, y, ids = x[perm], y[perm], ids[perm]
    alerts = rng.integers(1, 11, size=len(y), dtype=np.int32)
    # Stratified, fixed split. The split is known for the artificial fixture.
    from sklearn.model_selection import train_test_split
    train, other = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed)
    val, test = train_test_split(other, test_size=0.5, stratify=y[other], random_state=seed + 1)
    split = np.full(len(y), "train", dtype="U10")
    split[val] = "validation"
    split[test] = "test"
    # Fake Gaussian stamps used ONLY to exercise the gallery code.
    chosen = []
    for cls in range(n_classes):
        chosen.extend(rng.choice(np.flatnonzero((y == cls) & (split == "validation")), size=3, replace=False))
    chosen = np.asarray(chosen, dtype=np.int32)
    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    thumbs = []
    for idx in chosen:
        sigma = (0.09 + 0.04 * y[idx])
        cx, cy = rng.uniform(-0.15, 0.15, size=2)
        gauss = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        image = np.stack([
            gauss + rng.normal(scale=.10, size=gauss.shape),
            .75 * gauss + rng.normal(scale=.10, size=gauss.shape),
            .5 * gauss + rng.normal(scale=.10, size=gauss.shape),
        ])
        thumbs.append(image.astype(np.float32))
    return validate_teaching_data({
        "schema_version": np.array(1), "features": x, "labels": y,
        "object_ids": ids, "splits": split, "alert_counts": alerts,
        "preview_indices": chosen, "preview_images": np.stack(thumbs),
        "feature_source": np.array("synthetic latent features"),
        "provenance": np.array("SYNTHETIC DEMO ONLY: no ZTF or AstroMiNN data"),
    })