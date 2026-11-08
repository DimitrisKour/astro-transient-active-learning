#!/usr/bin/env python3
"""Prepare a *small*, real, CPU-loadable AppleCiDEr school dataset.

Run this one-time export where the historical manifest and referenced .npy
samples are accessible (e.g., your research storage). The teaching notebook never reads those sources and never needs PyTorch or
model checkpoints. Historical .npy files may contain pickled PyTorch objects,
so the one-time preparation environment must provide torch to deserialize them.

Example (replace with your verified manifest/column paths):
  python scripts/examples/export_astro_school_features.py \
      --train-manifest /path/to/train.csv \
      --validation-manifest /path/to/validation.csv \
      --test-manifest /path/to/test.csv \
      --data-root /path/to/data_root \
      --object-id-column object_id --file-column filepath --label-column label \
      --output examples/data/astro_real_v1.npz

Output is **not** automatically publishable. Review label/data rights,
source provenance, class balance, and CPU pilot accuracy before committing.
The deterministic image/metadata representation is NOT an AstroMiNN embedding.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from al_pipeline.astronomy.teaching_data import CLASS_NAMES, validate_teaching_data

LABEL_TO_INDEX = {
    "AGN": 0, "Nuclear": 0, "TDE": 1, "Tidal Disruption Event": 1,
    "SN I": 2, "SN Ia": 2, "SN Ib": 2, "SN Ic": 2,
    "SN II": 3, "SN IIP": 3, "SN IIp": 3, "SN IIn": 3,
    "CV": 4, "Cataclysmic": 4,
}


def read_object_manifest(path, *, oid_column, file_column, label_column):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {oid_column, file_column, label_column, "alert"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        objects = {}
        for row in reader:
            oid = row[oid_column].strip()
            raw_label = row[label_column].strip()
            if not oid or raw_label not in LABEL_TO_INDEX:
                raise ValueError(f"Bad object ID or unsupported label {oid!r}: {raw_label!r}")
            label = LABEL_TO_INDEX[raw_label]
            alert = int(row["alert"])
            if alert < 0:
                raise ValueError(f"Negative alert for {oid}")
            prev = objects.get(oid)
            if prev is not None and prev[0] != label:
                raise ValueError(f"Conflicting class labels for object {oid}")
            if prev is None:
                objects[oid] = (label, alert, row[file_column], 1)
            elif alert >= prev[1]:
                if alert == prev[1]:
                    raise ValueError(f"Duplicate latest alert for {oid}: {alert}")
                objects[oid] = (label, alert, row[file_column], prev[3] + 1)
            else:
                objects[oid] = (prev[0], prev[1], prev[2], prev[3] + 1)
    return objects


def select_objects(objects, *, max_objects, minimum_per_class, rng):
    """Cap each split while retaining rare-class support (sampled benchmark)."""
    by_class = defaultdict(list)
    for oid, (label, *_other) in objects.items():
        by_class[label].append(oid)
    if len(by_class) != len(CLASS_NAMES):
        raise ValueError("Historical split does not contain all five target classes")
    selected = []
    for label in range(len(CLASS_NAMES)):
        entries = sorted(by_class[label])
        minimum = min(minimum_per_class, len(entries))
        selected.extend(rng.choice(entries, size=minimum, replace=False).tolist())
    if max_objects < len(selected):
        raise ValueError("Requested cap is too small to cover required per-class minimums")
    remaining = sorted(set(objects) - set(selected))
    more = min(max_objects - len(selected), len(remaining))
    selected.extend(rng.choice(remaining, size=more, replace=False).tolist())
    return sorted(selected)


def derive_image_features(sample):
    """CPU-only, deterministic per-object 8x8 image bins + 34 metadata values."""
    img = np.asarray(sample["image"], dtype=np.float32)
    meta = np.asarray(sample["metadata"], dtype=np.float32)
    if img.shape != (3, 63, 63) or meta.shape != (34,):
        raise ValueError(f"Unsupported sample shapes: {img.shape}, {meta.shape}")
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    sections = []
    for ch in img:
        center = np.median(ch)
        scaled = np.clip((ch - center) / (ch.std() + 1e-6), -5.0, 5.0)
        rows = np.array_split(scaled, 8, axis=0)
        blocks = np.stack([np.asarray([piece.mean() for piece in np.array_split(row, 8, axis=1)]) for row in rows])
        sections.extend(blocks.ravel())
    meta = np.nan_to_num(meta, nan=0.0, posinf=0.0, neginf=0.0)
    meta = np.clip(meta, -1000.0, 1000.0)
    return np.concatenate([np.asarray(sections, np.float32), meta]).astype(np.float32), img


def resolve_path(value, *, data_root):
    path = Path(value)
    if path.is_absolute():
        return path
    if data_root is None:
        raise ValueError("Relative sample paths require --data-root")
    return Path(data_root) / path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(args):
    rng = np.random.default_rng(args.seed)
    locations = {
        "train": Path(args.train_manifest),
        "validation": Path(args.validation_manifest),
        "test": Path(args.test_manifest),
    }
    selected = {}
    existing = set()
    for name, path in locations.items():
        objects = read_object_manifest(path, oid_column=args.object_id_column,
                                       file_column=args.file_column,
                                       label_column=args.label_column)
        if existing.intersection(objects):
            raise ValueError(f"Cross-split object-ID leakage in {name}")
        existing.update(objects)
        cap = {"train": args.train_objects, "validation": args.validation_objects,
               "test": args.test_objects}[name]
        selected[name] = {oid: objects[oid] for oid in select_objects(
            objects, max_objects=cap, minimum_per_class=(12 if name == "train" else 4), rng=rng)}
        if name == "train" and any(sum(item[0] == cls for item in selected[name].values()) < 4
                                   for cls in range(len(CLASS_NAMES))):
            raise ValueError("At least four training objects per class are needed for L0")
    features, labels, ids, splits, alerts = [], [], [], [], []
    previews, preview_indices = [], []
    for name, objects in selected.items():
        previews_per_class = Counter()
        for oid, (label, last_alert, file_path, n_alerts) in objects.items():
            path = resolve_path(file_path, data_root=args.data_root)
            if not path.is_file():
                raise FileNotFoundError(path)
            # Historical AstroMiNN files store a trusted Python mapping in an
            # .npy file. allow_pickle=True is ONLY used on trusted research input.
            try:
                sample = np.load(path, allow_pickle=True).item()
            except ModuleNotFoundError as exc:
                if exc.name != "torch":
                    raise
                raise RuntimeError(
                    "This trusted historical .npy sample contains pickled PyTorch "
                    "objects. The one-time instructor export requires an environment "
                    "with torch installed (such as the research environment), even "
                    "though the exported .npz and the student notebook are CPU-only "
                    "and do not require PyTorch. Rerun the exporter using that "
                    "environment; do not install torch in the teaching environment."
                ) from exc
            if str(sample.get("obj_id")) != oid:
                raise ValueError(f"Manifest/sample object mismatch: {oid}, {path}")
            sample_alert = sample.get("alerte", sample.get("alert", last_alert))
            if int(sample_alert) != last_alert:
                raise ValueError(f"Manifest/sample alert mismatch for {oid}")
            vec, img = derive_image_features(sample)
            index = len(ids)
            features.append(vec)
            labels.append(label)
            ids.append(oid)
            splits.append(name)
            alerts.append(n_alerts)
            if name == "validation" and previews_per_class[label] < 3:
                # Pixel stamp previews only; no original large samples saved.
                thumbs = img[:, ::2, ::2]
                previews.append(thumbs)
                preview_indices.append(index)
                previews_per_class[label] += 1
    provenance = (
        "Real historical ZTF-derived AstroMiNN manifest samples; "
        "image bins + metadata, one latest-alert sample per object; "
        "not neural-network embeddings"
    )
    dataset = validate_teaching_data({
        "schema_version": np.array(1),
        "features": np.asarray(features, np.float32),
        "labels": np.asarray(labels, np.int32),
        "object_ids": np.asarray(ids, dtype=str),
        "splits": np.asarray(splits, dtype=str),
        "alert_counts": np.asarray(alerts, np.int32),
        "preview_indices": np.asarray(preview_indices, np.int32),
        "preview_images": np.asarray(previews, np.float32),
        "feature_source": np.array("image_8x8_channels_plus_34_metadata"),
        "provenance": np.array(provenance),
    })
    output = Path(args.output).expanduser()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    metadata = {
        "schema_version": 1, "dataset": "astro_school_cpu_object_level",
        "provenance": provenance, "feature_source": str(dataset["feature_source"]),
        "split_counts": {name: Counter(CLASS_NAMES[int(i)] for i in np.asarray(labels)[np.asarray(splits) == name])
                         for name in ("train", "validation", "test")},
        "manifest_sha256": {name: sha256(path) for name, path in locations.items()},
        "seed": args.seed, "features_shape": list(dataset["features"].shape),
        "output_bytes": output.stat().st_size,
        "methodological_note": (
            "All features derived per object without hidden labels; logistic-regression"
            " scaler is fit only on revealed training examples at each cycle."
        ),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Exported {output}; check redistribution rights before publication")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "test"):
        p.add_argument(f"--{name}-manifest", type=Path, required=True)
        p.add_argument(f"--{name}-objects", type=int,
                       default={"train": 700, "validation": 200, "test": 200}[name])
    p.add_argument("--object-id-column", default="object_id")
    p.add_argument("--file-column", default="filepath")
    p.add_argument("--label-column", default="label")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=135)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if any(getattr(args, f"{name}_objects") < 10 for name in ("train", "validation", "test")):
        p.error("Each split cap must be at least ten objects")
    export(args)


if __name__ == "__main__":
    main()