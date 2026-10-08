"""Fixed-split evaluator for the historical AstroMiNN ViT."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from al_pipeline.astronomy.data.registry import DatasetSpec
from al_pipeline.core.evaluation.base import ModelEvaluationResult, SplitEvaluation
from al_pipeline.core.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
)
from al_pipeline.astronomy.astrominn.inference.aggregation import aggregate_object_predictions
from al_pipeline.astronomy.astrominn.inference.astrominn import AstroMiNNPredictor
from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_LEGACY_CODE_DIR,
    load_astrominn_vit_model,
)
from al_pipeline.astronomy.astrominn.training import AstroMiNNViTManifestDataset, MODEL_CLASS_NAMES


class AstroMiNNViTEvaluator:
    """Evaluate checkpoints on immutable sample manifests.

    Metrics are recorded both at sample level and at object level using the
    latest available alert. Active-learning learning curves should use the
    object-level metrics because acquisition also operates per object.
    """

    def __init__(
        self,
        *,
        dataset: DatasetSpec,
        manifests: Mapping[str, str | Path] | None = None,
        device: str | torch.device = "cpu",
        batch_size: int = 2_048,
        num_workers: int = 0,
        legacy_code_dir: str | Path = DEFAULT_LEGACY_CODE_DIR,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")

        if manifests is None:
            manifests = {
                name: path
                for name, path in (
                    ("validation", dataset.validation_manifest),
                    ("test", dataset.test_manifest),
                )
                if path is not None
            }

        if not manifests:
            raise ValueError("At least one fixed evaluation manifest is required.")

        self.dataset = dataset
        self.manifests = {str(name): Path(path) for name, path in manifests.items()}
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.legacy_code_dir = Path(legacy_code_dir)
        self.artifact_dir = Path(artifact_dir)

    def evaluate(
        self,
        *,
        checkpoint_path: Path,
        output_dir: Path,
    ) -> ModelEvaluationResult:
        """Run evaluation and write metrics, confusions, and predictions."""

        checkpoint_path = Path(checkpoint_path)
        output_dir = Path(output_dir)

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        for manifest in self.manifests.values():
            if not manifest.exists():
                raise FileNotFoundError(manifest)

        existing = self.completed_result(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )

        if existing is not None:
            return existing

        output_dir.mkdir(parents=True, exist_ok=True)
        model = load_astrominn_vit_model(
            legacy_code_dir=self.legacy_code_dir,
            artifact_dir=self.artifact_dir,
            vit_checkpoint=checkpoint_path,
        )
        predictor = AstroMiNNPredictor(
            model,
            device=self.device,
        )
        split_results = []

        for split, manifest in self.manifests.items():
            split_results.append(
                self._evaluate_split(
                    split=split,
                    manifest=manifest,
                    predictor=predictor,
                    output_dir=output_dir,
                )
            )

        summary_path = output_dir / "evaluation_summary.json"
        payload = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "model_class_names": list(MODEL_CLASS_NAMES),
            "splits": [asdict(result) for result in split_results],
        }
        _json_dump_atomic(payload, summary_path)

        return ModelEvaluationResult(
            checkpoint_path=checkpoint_path,
            summary_path=summary_path,
            splits=tuple(split_results),
        )

    def completed_result(
        self,
        *,
        checkpoint_path: Path,
        output_dir: Path,
    ) -> ModelEvaluationResult | None:
        """Load evaluation only when it belongs to the same checkpoint."""

        checkpoint_path = Path(checkpoint_path)
        summary_path = Path(output_dir) / "evaluation_summary.json"

        if not summary_path.exists():
            return None

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        observed_hash = payload.get("checkpoint_sha256")
        expected_hash = _sha256_file(checkpoint_path)

        if observed_hash != expected_hash:
            raise ValueError("Existing evaluation belongs to a different checkpoint.")

        if tuple(payload.get("model_class_names", ())) != MODEL_CLASS_NAMES:
            raise ValueError("Existing evaluation uses a different model class order.")

        splits = tuple(SplitEvaluation(**item) for item in payload["splits"])

        return ModelEvaluationResult(
            checkpoint_path=checkpoint_path,
            summary_path=summary_path,
            splits=splits,
        )

    def _evaluate_split(
        self,
        *,
        split: str,
        manifest: Path,
        predictor: AstroMiNNPredictor,
        output_dir: Path,
    ) -> SplitEvaluation:
        dataset = AstroMiNNViTManifestDataset(
            manifest_path=manifest,
            dataset=self.dataset,
            artifact_dir=self.artifact_dir,
            augment=False,
            cache_in_memory=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        inference = predictor.predict(loader)

        if inference.target_indices is None:
            raise RuntimeError(f"Evaluation split {split!r} did not provide targets.")

        sample_targets = np.asarray(
            inference.target_indices,
            dtype=np.int64,
        )
        sample_metrics = compute_classification_metrics(
            sample_targets,
            inference.predictions,
        )
        sample_per_class = compute_per_class_metrics(
            sample_targets,
            inference.predictions,
        )
        sample_confusion = compute_confusion_matrix(
            sample_targets,
            inference.predictions,
        )

        object_predictions = aggregate_object_predictions(
            object_ids=inference.object_ids,
            alerts=inference.alerts,
            predictions=inference.predictions,
            method="latest",
        )
        target_by_object: dict[str, int] = {}

        for object_id, target in zip(
            inference.object_ids,
            sample_targets,
            strict=True,
        ):
            previous = target_by_object.setdefault(
                object_id,
                int(target),
            )

            if previous != int(target):
                raise ValueError(
                    f"Object {object_id!r} has inconsistent labels in evaluation split {split!r}."
                )

        object_targets = np.asarray(
            [target_by_object[object_id] for object_id in object_predictions.object_ids],
            dtype=np.int64,
        )
        object_metrics = compute_classification_metrics(
            object_targets,
            object_predictions.predictions,
        )
        object_per_class = compute_per_class_metrics(
            object_targets,
            object_predictions.predictions,
        )
        object_confusion = compute_confusion_matrix(
            object_targets,
            object_predictions.predictions,
        )

        _write_confusion(
            output_dir / f"{split}_sample_confusion.csv",
            sample_confusion,
        )
        _write_confusion(
            output_dir / f"{split}_object_confusion.csv",
            object_confusion,
        )
        _json_dump_atomic(
            {
                "sample": _named_per_class(sample_per_class),
                "object": _named_per_class(object_per_class),
            },
            output_dir / f"{split}_per_class_metrics.json",
        )
        _save_predictions(
            output_dir / f"{split}_predictions.npz",
            sample_object_ids=inference.object_ids,
            sample_alerts=inference.alerts,
            sample_targets=sample_targets,
            sample_values=inference.predictions.values,
            object_ids=object_predictions.object_ids,
            object_targets=object_targets,
            object_values=object_predictions.predictions.values,
        )

        return SplitEvaluation(
            split=split,
            sample_count=len(sample_targets),
            object_count=len(object_targets),
            sample_metrics=sample_metrics,
            object_metrics=object_metrics,
        )


def _named_per_class(
    metrics: Mapping[object, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {MODEL_CLASS_NAMES[int(index)]: dict(values) for index, values in metrics.items()}


def _write_confusion(path: Path, matrix: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *MODEL_CLASS_NAMES])

        for class_name, row in zip(
            MODEL_CLASS_NAMES,
            matrix,
            strict=True,
        ):
            writer.writerow([class_name, *map(int, row)])

    temporary.replace(path)


def _save_predictions(
    path: Path,
    *,
    sample_object_ids: tuple[str, ...],
    sample_alerts: tuple[int | None, ...],
    sample_targets: np.ndarray,
    sample_values: np.ndarray,
    object_ids: tuple[str, ...],
    object_targets: np.ndarray,
    object_values: np.ndarray,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_object_ids=np.asarray(sample_object_ids, dtype=str),
            sample_alerts=np.asarray(
                [-1 if alert is None else alert for alert in sample_alerts],
                dtype=np.int64,
            ),
            sample_targets=sample_targets,
            sample_values=sample_values,
            object_ids=np.asarray(object_ids, dtype=str),
            object_targets=object_targets,
            object_values=object_values,
        )

    temporary.replace(path)


def _json_dump_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


__all__ = ["AstroMiNNViTEvaluator"]
