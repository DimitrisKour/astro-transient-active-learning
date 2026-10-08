"""Tests for the historical AstroMiNN ViT training adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from torch import nn

from al_pipeline.astronomy.data.registry import DatasetSpec
from al_pipeline.astronomy.astrominn.training import (
    AstroMiNNViTManifestDataset,
    AstroMiNNViTTrainer,
    AstroMiNNViTTrainingConfig,
    MODEL_CLASS_NAMES,
    _class_weights,
    _new_object_training_weights,
)
from al_pipeline.core.training.base import TrainingRequest


LABELS = (
    "AGN",
    "Tidal Disruption Event",
    "SN Ia",
    "SN II",
    "Cataclysmic",
)


def _write_artifact_config(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "astroMiNN_config.json").write_text(
        json.dumps(
            {
                "classes": {
                    "Nuclear": ["AGN"],
                    "Tidal Disruption Event": ["Tidal Disruption Event"],
                    "SN I": ["SN Ia", "SN Ib", "SN Ic"],
                    "SN II": ["SN II", "SN IIn", "SN IIP"],
                    "Cataclysmic": ["Cataclysmic"],
                },
                "patch_size": [45, 5],
                "image_norm": "median",
            }
        ),
        encoding="utf-8",
    )


def _write_manifest_and_samples(
    root: Path,
    name: str,
) -> Path:
    sample_dir = root / f"{name}_samples"
    sample_dir.mkdir()
    rows = []

    for index, label in enumerate(LABELS):
        object_id = f"{name}_{index}"
        sample_path = sample_dir / f"{object_id}.npy"
        generator = np.random.default_rng(index)
        np.save(
            sample_path,
            {
                "obj_id": object_id,
                "alerte": index,
                "metadata": generator.normal(size=34).astype(np.float32),
                "image": generator.normal(size=(3, 63, 63)).astype(np.float32),
                "target": label,
            },
            allow_pickle=True,
        )
        rows.append(
            {
                "object_id": object_id,
                "filepath": str(sample_path),
                "label": label,
                "alert": str(index),
            }
        )

    manifest = root / f"{name}.csv"

    with manifest.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "object_id",
                "filepath",
                "label",
                "alert",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        name="astrominn-test",
        survey="ZTF",
        modality="image_metadata",
        selection_unit="object",
        object_id_column="object_id",
        sample_file_column="filepath",
        label_column="label",
    )


class _ToyAstroMiNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lc_tower = nn.Sequential(
            nn.BatchNorm1d(34),
            nn.Linear(34, 4),
        )
        self.spatial_tower = nn.Linear(34, 4)
        self.coord_tower = nn.Linear(34, 4)
        self.vit_tower = nn.Linear(
            34,
            len(MODEL_CLASS_NAMES),
        )

    def forward(self, metadata, image=None):
        return {"logits": self.vit_tower(metadata)}


def test_manifest_dataset_preserves_one_hot_float_targets(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_config(artifact_dir)
    manifest = _write_manifest_and_samples(
        tmp_path,
        "train",
    )
    dataset = AstroMiNNViTManifestDataset(
        manifest_path=manifest,
        dataset=_dataset_spec(),
        artifact_dir=artifact_dir,
    )

    item = dataset[0]

    assert tuple(item["metadata"].shape) == (34,)
    assert tuple(item["image"].shape) == (3, 45, 45)
    assert item["target"].dtype == torch.float32
    assert tuple(item["target"].shape) == (5,)
    assert item["target"].sum().item() == 1.0
    assert item["sample_weight"].item() == 1.0


def test_equal_object_weighting_preserves_historical_rows(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical.csv"
    cumulative = tmp_path / "cumulative.csv"
    fieldnames = ["object_id", "filepath", "label", "alert"]
    historical_rows = [
        {"object_id": "old_a", "filepath": "a0", "label": "A", "alert": 0},
        {"object_id": "old_a", "filepath": "a1", "label": "A", "alert": 1},
        {"object_id": "old_b", "filepath": "b0", "label": "B", "alert": 0},
    ]
    new_rows = [
        {"object_id": "new_a", "filepath": "n0", "label": "A", "alert": 0},
        {"object_id": "new_a", "filepath": "n1", "label": "A", "alert": 1},
        {"object_id": "new_a", "filepath": "n2", "label": "A", "alert": 2},
        {"object_id": "new_b", "filepath": "m0", "label": "B", "alert": 0},
    ]

    for path, rows in (
        (historical, historical_rows),
        (cumulative, [*historical_rows, *new_rows]),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    weights, summary = _new_object_training_weights(
        train_manifest=cumulative,
        historical_manifest=historical,
        object_id_column="object_id",
        mode="equal_object",
    )

    assert set(weights) == {"new_a", "new_b"}
    assert weights["new_a"] == pytest.approx(0.5)
    assert weights["new_b"] == pytest.approx(1.5)
    assert summary["historical_mean_samples_per_object"] == 1.5
    assert summary["new_object_total_weight"] == 1.5


def test_legacy_policy_exposes_recovered_quirks() -> None:
    policy = AstroMiNNViTTrainingConfig.legacy_seed135()

    assert policy.augment_validation
    assert policy.legacy_frozen_expert_train_mode
    assert policy.legacy_seed_sequence


def test_manifest_class_weights_use_only_current_training_rows(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "current_labeled.csv"
    rows = []
    for class_index, label in enumerate(LABELS, start=1):
        rows.extend(
            {
                "object_id": f"object_{class_index}_{row_index}",
                "filepath": "unused.npy",
                "label": label,
                "alert": 0,
            }
            for row_index in range(class_index)
        )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["object_id", "filepath", "label", "alert"],
        )
        writer.writeheader()
        writer.writerows(rows)

    weights, summary = _class_weights(
        policy=AstroMiNNViTTrainingConfig(class_weight_source="training_manifest"),
        device=torch.device("cpu"),
        train_manifest=manifest,
        label_column="label",
    )

    assert weights.mean().item() == pytest.approx(1.0)
    assert summary["source"] == "training_manifest"
    assert list(summary["sample_counts"].values()) == [1, 2, 3, 4, 5]
    assert weights[0] > weights[-1]


def test_manifest_class_weights_require_every_labeled_class(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "incomplete.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label"])
        writer.writeheader()
        writer.writerow({"label": "AGN"})

    with pytest.raises(ValueError, match="requires every class"):
        _class_weights(
            policy=AstroMiNNViTTrainingConfig(class_weight_source="training_manifest"),
            device=torch.device("cpu"),
            train_manifest=manifest,
            label_column="label",
        )


def test_trainer_writes_reusable_cycle_artifacts(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    _write_artifact_config(artifact_dir)
    train_manifest = _write_manifest_and_samples(
        tmp_path,
        "train",
    )
    validation_manifest = _write_manifest_and_samples(
        tmp_path,
        "validation",
    )

    trainer = AstroMiNNViTTrainer(
        dataset=_dataset_spec(),
        artifact_dir=artifact_dir,
        initialization="random_vit",
        device="cpu",
        policy=AstroMiNNViTTrainingConfig(
            batch_size=5,
            validation_batch_size=5,
            max_epochs=2,
            patience=1,
            train_num_workers=0,
            validation_num_workers=0,
            augment_training=False,
            augment_validation=False,
        ),
        model_factory=lambda load, checkpoint: _ToyAstroMiNN(),
    )
    request = TrainingRequest(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        output_dir=tmp_path / "run",
        run_id="toy",
        cycle=0,
        strategy="baseline",
        seed=135,
    )

    result = trainer.train(request)

    assert result.checkpoint_path.exists()
    assert result.history_path.exists()
    assert result.summary_path.exists()
    assert result.epochs_completed >= 1

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["training_samples"] == 5
    assert summary["validation_samples"] == 5
    assert summary["trainable_parameters"] < summary["total_parameters"]

    recovered = trainer.completed_result(request)

    assert recovered is not None
    assert recovered.checkpoint_path == result.checkpoint_path
    assert recovered.best_epoch == result.best_epoch
