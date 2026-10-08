"""Reconstruct the historical AstroMiNN ViT train/val/test manifests.

This reproduces the object-level split used by the January 22, 2026
AstroMiNN vit_tower_135 checkpoint:

- data root: fixed_data
- use_all_samples: True
- maximum alert: 10
- loader seed: 125
- train/validation/test object split: 50% / 25% / 25%

The historical dataloader's exact hard-coded `real_classes` representation
is preserved intentionally for split reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


# IMPORTANT:
# This intentionally matches dataloader3.py exactly, including spelling.
# Do not "correct" these names here unless deliberately defining a new split.
REAL_CLASSES = (
    "AGN",
    "TDE",
    "SN II",
    "SN IIp",
    "SN Ia",
    "SN IIn",
    "SN Ib",
    "SN Ic",
    "Cataclysmic",
)

MANIFEST_FIELDS = (
    "obj_id",
    "filepath",
    "target",
    "alert",
    "survey",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--loader-seed",
        type=int,
        default=125,
    )
    parser.add_argument(
        "--max-alert",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--survey",
        type=str,
        default="ZTF",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def split_target(label: str) -> np.ndarray:
    """Reproduce dataloader3.py's historical real_target vector."""
    target = np.zeros(
        len(REAL_CLASSES),
        dtype=np.uint8,
    )

    for index, category in enumerate(REAL_CLASSES):
        if label == category:
            target[index] = 1

    return target


def write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_paths = sorted(
        data_root.glob("*.npy"),
        key=lambda path: path.name,
    )

    if not file_paths:
        raise ValueError(f"No .npy files found under {data_root}")

    rows: list[dict[str, str]] = []

    objects: list[str] = []
    object_split_targets: list[np.ndarray] = []
    object_labels: dict[str, str] = {}

    obj_id_to_indices: dict[str, list[int]] = defaultdict(list)

    skipped_after_alert = 0

    print(f"Found {len(file_paths):,} .npy files.")
    print("Reading samples...")

    for file_number, path in enumerate(
        file_paths,
        start=1,
    ):
        sample = np.load(
            path,
            allow_pickle=True,
        ).item()

        try:
            obj_id = str(sample["obj_id"])
            original_class = str(sample["target"])
            alert = int(sample["alerte"])
        except KeyError as exc:
            raise ValueError(f"{path} is missing required field {exc.args[0]!r}.") from exc

        # Exact behavior of AstroDataset when:
        # use_all_samples=True and alert=10.
        if args.max_alert > 0 and alert > args.max_alert:
            skipped_after_alert += 1
            continue

        if not obj_id:
            raise ValueError(f"Empty obj_id in {path}")

        previous_label = object_labels.get(obj_id)

        if previous_label is not None and previous_label != original_class:
            raise ValueError(
                f"Object {obj_id!r} has inconsistent labels: "
                f"{previous_label!r} and {original_class!r}."
            )

        if obj_id not in object_labels:
            object_labels[obj_id] = original_class
            objects.append(obj_id)
            object_split_targets.append(split_target(original_class))

        row = {
            "obj_id": obj_id,
            "filepath": str(path),
            "target": original_class,
            "alert": str(alert),
            "survey": args.survey,
        }

        obj_id_to_indices[obj_id].append(len(rows))
        rows.append(row)

        if file_number % 10_000 == 0:
            print(f"  inspected {file_number:,} / {len(file_paths):,}")

    print(f"Retained {len(rows):,} samples from {len(objects):,} objects.")
    print(f"Skipped {skipped_after_alert:,} samples with alert > {args.max_alert}.")

    y = np.asarray(object_split_targets)

    # First historical split:
    # 50% train, 50% validation+test.
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=args.loader_seed,
    )

    train_obj_indexes, val_test_obj_indexes = next(
        splitter.split(
            np.zeros(len(objects)),
            y,
        )
    )

    val_test_classes = np.asarray([object_split_targets[index] for index in val_test_obj_indexes])

    # Second historical split:
    # remaining 50% -> 25% validation, 25% test.
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=args.loader_seed,
    )

    val_indexes_relative, test_indexes_relative = next(
        splitter.split(
            np.zeros(len(val_test_obj_indexes)),
            val_test_classes,
        )
    )

    train_obj_ids = [objects[index] for index in train_obj_indexes]

    val_test_obj_ids = [objects[index] for index in val_test_obj_indexes]

    val_obj_ids = [val_test_obj_ids[index] for index in val_indexes_relative]

    test_obj_ids = [val_test_obj_ids[index] for index in test_indexes_relative]

    train_objects = set(train_obj_ids)
    validation_objects = set(val_obj_ids)
    test_objects = set(test_obj_ids)

    if train_objects & validation_objects:
        raise RuntimeError("Train/validation object overlap detected.")

    if train_objects & test_objects:
        raise RuntimeError("Train/test object overlap detected.")

    if validation_objects & test_objects:
        raise RuntimeError("Validation/test object overlap detected.")

    if (train_objects | validation_objects | test_objects) != set(objects):
        raise RuntimeError("Object split does not cover the complete dataset.")

    def rows_for_objects(
        object_ids: list[str],
    ) -> list[dict[str, str]]:
        indices: list[int] = []

        for obj_id in object_ids:
            indices.extend(obj_id_to_indices[obj_id])

        return [rows[index] for index in indices]

    train_rows = rows_for_objects(train_obj_ids)
    validation_rows = rows_for_objects(val_obj_ids)
    test_rows = rows_for_objects(test_obj_ids)

    if len(train_rows) + len(validation_rows) + len(test_rows) != len(rows):
        raise RuntimeError("Manifest rows do not cover all retained samples.")

    train_path = output_dir / "train_manifest.csv"
    validation_path = output_dir / "validation_manifest.csv"
    test_path = output_dir / "test_manifest.csv"

    write_manifest(
        train_path,
        train_rows,
        overwrite=args.overwrite,
    )
    write_manifest(
        validation_path,
        validation_rows,
        overwrite=args.overwrite,
    )
    write_manifest(
        test_path,
        test_rows,
        overwrite=args.overwrite,
    )

    labels = Counter(object_labels.values())

    zero_vector_labels = sorted(
        {label for label in object_labels.values() if split_target(label).sum() == 0}
    )

    metadata = {
        "data_root": str(data_root),
        "loader_seed": args.loader_seed,
        "max_alert": args.max_alert,
        "use_all_samples": True,
        "survey": args.survey,
        "split": {
            "train_fraction": 0.5,
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "unit": "object",
        },
        "counts": {
            "retained_samples": len(rows),
            "objects": len(objects),
            "train_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "test_samples": len(test_rows),
            "train_objects": len(train_objects),
            "validation_objects": len(validation_objects),
            "test_objects": len(test_objects),
        },
        "object_label_counts": dict(sorted(labels.items())),
        "historical_split_real_classes": list(REAL_CLASSES),
        "labels_with_zero_historical_split_vector": (zero_vector_labels),
    }

    metadata_path = output_dir / "split_metadata.json"

    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {metadata_path}")

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Manifest generation complete.")
    print(f"Train:      {len(train_rows):,} samples / {len(train_objects):,} objects")
    print(f"Validation: {len(validation_rows):,} samples / {len(validation_objects):,} objects")
    print(f"Test:       {len(test_rows):,} samples / {len(test_objects):,} objects")
    print()
    print("Original object labels:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count:,}")

    if zero_vector_labels:
        print()
        print("Historical split labels represented by an all-zero real_target vector:")
        for label in zero_vector_labels:
            print(f"  {label}")

    print()
    print(f"Wrote manifests to: {output_dir}")


if __name__ == "__main__":
    main()
