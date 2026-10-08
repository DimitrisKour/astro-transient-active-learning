"""Historical five-class AstroMiNN ViT adapter.

This module isolates the model-specific details required to reproduce the
frozen seed-135 AstroMiNN ViT experiment. Generic active-learning code should
not depend on anything defined here.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
from al_pipeline.astronomy.astrominn.inference.aggregation import aggregate_object_predictions
from al_pipeline.astronomy.astrominn.inference.astrominn import (
    AstroMiNNInferenceResult,
    AstroMiNNPredictor,
)


DEFAULT_LEGACY_CODE_DIR = Path("/projects/bcrv/ffontinelenunes/things_for_hailey")

DEFAULT_ARTIFACT_DIR = Path(
    "/projects/bcrv/dkourtesis/applecider_adaptive/historical/astrominn_vit135_20260122"
)

IMAGE_EPS = 1e-8


class AstroMiNNViTPoolPredictor:
    """Checkpoint-aware predictor used by active-learning experiments."""

    def __init__(
        self,
        *,
        pool_path: str | Path,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
        num_workers: int = 0,
        legacy_code_dir: str | Path = DEFAULT_LEGACY_CODE_DIR,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")

        self.pool_path = Path(pool_path)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.legacy_code_dir = Path(legacy_code_dir)
        self.artifact_dir = Path(artifact_dir)

    def predict(
        self,
        *,
        object_ids: Iterable[str],
        checkpoint_path: Path,
    ) -> AstroMiNNInferenceResult:
        """Score the requested unlabeled objects with one checkpoint."""

        return predict_astrominn_vit_pool(
            pool_path=self.pool_path,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            legacy_code_dir=self.legacy_code_dir,
            artifact_dir=self.artifact_dir,
            object_ids=object_ids,
            vit_checkpoint=checkpoint_path,
        )


def load_astrominn_vit_config(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Load the frozen ViT experiment configuration."""

    artifact_dir = Path(artifact_dir)

    with (artifact_dir / "astroMiNN_config.json").open() as handle:
        return json.load(handle)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    """Load a historical PyTorch state dictionary."""

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if isinstance(state, dict) and state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    return state


def load_astrominn_vit_model(
    *,
    legacy_code_dir: str | Path = DEFAULT_LEGACY_CODE_DIR,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    load_vit_checkpoint: bool = True,
    vit_checkpoint: str | Path | None = None,
) -> torch.nn.Module:
    """Reconstruct the five-class seed-135 AstroMiNN ViT.

    Parameters
    ----------
    load_vit_checkpoint
        If true, load the historical ViT classifier weights. Set this to
        false only when reproducing the original ViT training from its
        random initialization. The three pretrained expert towers are always
        loaded.

    vit_checkpoint
        Optional replacement ViT checkpoint. Adaptive-learning cycles use
        this to initialize cycle ``t`` from the best checkpoint produced by
        cycle ``t - 1``. Checkpoints are expected to contain either a bare
        ViT state dictionary or a ``state_dict`` wrapper.
    """

    legacy_code_dir = Path(legacy_code_dir)
    artifact_dir = Path(artifact_dir)

    if str(legacy_code_dir) not in sys.path:
        sys.path.insert(0, str(legacy_code_dir))

    # Historical code imports Towers.py and ViT.py from this directory.
    from astroMiNN_V2 import astroMiNN

    config = copy.deepcopy(load_astrominn_vit_config(artifact_dir))

    # Prevent the historical constructor from loading relative checkpoint
    # paths. We explicitly load the frozen, hashed checkpoints below.
    #
    # "vit_tower" also chooses the ViT branch in astroMiNN.forward().
    config["tags"] = [
        "vit_tower",
        "lc_tower",
        "spatial_tower",
        "coord_tower",
    ]

    model = astroMiNN(
        config=config,
        num_classes=5,
    )

    model.lc_tower.load_state_dict(
        _load_state(artifact_dir / "best_lc_tower_135.pth"),
        strict=True,
    )

    model.spatial_tower.load_state_dict(
        _load_state(artifact_dir / "best_spatial_tower_135.pth"),
        strict=True,
    )

    # The ViT consumes the 104-dimensional spatial hidden representation,
    # rather than the spatial tower's three-class output.
    model.spatial_tower.buzzcut()

    model.coord_tower.load_state_dict(
        _load_state(artifact_dir / "best_coord_tower_135.pth"),
        strict=True,
    )

    if load_vit_checkpoint:
        checkpoint_path = (
            Path(vit_checkpoint)
            if vit_checkpoint is not None
            else artifact_dir / "vit_tower_135.pth"
        )

        vit_state = _load_state(checkpoint_path)

        if vit_state and all(key.startswith("vit_tower.") for key in vit_state):
            vit_state = {key.removeprefix("vit_tower."): value for key, value in vit_state.items()}

        model.vit_tower.load_state_dict(
            vit_state,
            strict=True,
        )

    model.eval()

    return model


def preprocess_astrominn_vit(
    *,
    metadata: np.ndarray,
    image: np.ndarray,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact ViT-era model-input preprocessing.

    The canonical adaptive-learning record remains unchanged:

        metadata: (34,)
        image:    (3, 63, 63)

    This function performs only the transformation required at the
    AstroMiNN-ViT model boundary.
    """

    metadata_tensor = torch.as_tensor(
        metadata,
        dtype=torch.float32,
    ).clone()

    if metadata_tensor.shape != (34,):
        raise ValueError(
            "AstroMiNN ViT expects 34 metadata features before "
            f"model preprocessing; got {tuple(metadata_tensor.shape)}."
        )

    metadata_tensor = torch.nan_to_num(
        metadata_tensor,
        nan=-999.0,
    )

    image_tensor = torch.as_tensor(
        image,
        dtype=torch.float32,
    ).clone()

    if image_tensor.shape != (3, 63, 63):
        raise ValueError(
            "AstroMiNN ViT expects canonical images with shape "
            f"(3, 63, 63); got {tuple(image_tensor.shape)}."
        )

    crop_size = int(config["patch_size"][0])

    lower = int((63 - crop_size) / 2)
    upper = 63 - lower

    image_tensor = image_tensor[
        :,
        lower:upper,
        lower:upper,
    ]

    image_norm = config["image_norm"]

    if image_norm == "median":
        for channel in range(3):
            channel_median = torch.median(image_tensor[channel].reshape(-1))

            image_tensor[channel] -= channel_median

            image_tensor[channel] /= image_tensor[channel].std() + IMAGE_EPS

    elif image_norm == "L2":
        image_tensor /= torch.norm(
            image_tensor,
            p=2,
        )

    elif image_norm:
        raise ValueError(f"Unsupported historical image_norm: {image_norm!r}.")

    if not torch.isfinite(metadata_tensor).all():
        raise ValueError("AstroMiNN ViT metadata contains non-finite values.")

    if not torch.isfinite(image_tensor).all():
        raise ValueError("AstroMiNN ViT image contains non-finite values.")

    return metadata_tensor, image_tensor


class AstroMiNNViTPoolDataset(Dataset):
    """Expose a canonical adaptive candidate pool to the historical ViT."""

    def __init__(
        self,
        pool_path: str | Path,
        *,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        object_ids: Iterable[str] | None = None,
    ) -> None:
        self.pool_path = Path(pool_path)
        self.samples_dir = self.pool_path / "samples"

        # These checks make the prepared pool boundary explicit.
        for required in (
            self.pool_path / "candidate_pool.csv",
            self.pool_path / "ingestion_report.json",
            self.samples_dir,
        ):
            if not required.exists():
                raise FileNotFoundError(f"Prepared candidate pool is incomplete: {required}")

        manifest = CandidatePoolManifest.from_csv(self.pool_path / "candidate_pool.csv")
        requested_ids = (
            None if object_ids is None else tuple(str(object_id) for object_id in object_ids)
        )

        if requested_ids is not None:
            if len(requested_ids) != len(set(requested_ids)):
                raise ValueError("object_ids must not contain duplicates.")

            unknown_ids = set(requested_ids).difference(manifest.object_ids)

            if unknown_ids:
                raise ValueError(
                    "Requested inference objects are not present in the "
                    f"candidate pool: {sorted(unknown_ids)}."
                )

            requested_set = set(requested_ids)
        else:
            requested_set = None

        self.sample_paths = tuple(
            Path(record.filepath)
            for record in manifest.records
            if requested_set is None or record.object_id in requested_set
        )

        if not self.sample_paths:
            raise ValueError(f"No candidate samples found in {self.samples_dir}.")

        self.config = load_astrominn_vit_config(artifact_dir)

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.sample_paths[index]

        sample = np.load(
            path,
            allow_pickle=True,
        ).item()

        if "obj_id" not in sample:
            raise KeyError(f"{path.name} does not contain 'obj_id'.")

        metadata, image = preprocess_astrominn_vit(
            metadata=np.asarray(sample["metadata"]),
            image=np.asarray(sample["image"]),
            config=self.config,
        )

        if "alerte" in sample:
            alert = int(sample["alerte"])
        elif "alert" in sample:
            alert = int(sample["alert"])
        else:
            raise KeyError(f"{path.name} does not contain an alert number.")

        return {
            "obj_id": str(sample["obj_id"]),
            "alert": alert,
            "metadata": metadata,
            "image": image,
        }


def predict_astrominn_vit_pool(
    *,
    pool_path: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
    legacy_code_dir: str | Path = DEFAULT_LEGACY_CODE_DIR,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    object_ids: Iterable[str] | None = None,
    vit_checkpoint: str | Path | None = None,
) -> AstroMiNNInferenceResult:
    """Load a ViT checkpoint and run inference over a prepared pool.

    ``object_ids`` restricts inference to the current unlabeled set without
    creating a second physical pool. ``vit_checkpoint`` lets later adaptive
    cycles score that set with the model produced by the previous cycle.
    Omitting it preserves the historical frozen-checkpoint behavior.
    """

    dataset = AstroMiNNViTPoolDataset(
        pool_path,
        artifact_dir=artifact_dir,
        object_ids=object_ids,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = load_astrominn_vit_model(
        legacy_code_dir=legacy_code_dir,
        artifact_dir=artifact_dir,
        vit_checkpoint=vit_checkpoint,
    )

    predictor = AstroMiNNPredictor(
        model=model,
        device=device,
        # Capture the normalized CLS representation passed directly to the
        # final classifier. A transient hook keeps the historical model and
        # checkpoint format unchanged.
        embedding_layer=model.vit_tower.fc_out,
    )

    result = predictor.predict(dataloader)

    # Active learning operates at object level. A real alert stream can
    # contain more than one historical-compatible alert for the same object.
    # In that case, keep the prediction from the latest available alert.
    #
    # TODO: Make the aggregation policy configurable if future experiments
    # need alternatives such as mean-probability aggregation.
    if len(result.object_ids) != len(set(result.object_ids)):
        aggregated = aggregate_object_predictions(
            object_ids=result.object_ids,
            alerts=result.alerts,
            predictions=result.predictions,
            method="latest",
        )

        if aggregated.source_indices is None:
            raise RuntimeError("Latest-alert aggregation did not expose source indices.")

        source_indices = np.asarray(
            aggregated.source_indices,
            dtype=np.int64,
        )

        latest_alert_by_object: dict[str, int] = {}

        for object_id, alert in zip(
            result.object_ids,
            result.alerts,
        ):
            if alert is None:
                raise ValueError(f"Object {object_id!r} is missing an alert number.")

            if object_id not in latest_alert_by_object or alert > latest_alert_by_object[object_id]:
                latest_alert_by_object[object_id] = alert

        result = AstroMiNNInferenceResult(
            object_ids=aggregated.object_ids,
            alerts=tuple(latest_alert_by_object[object_id] for object_id in aggregated.object_ids),
            predictions=aggregated.predictions,
            target_indices=None,
            embeddings=(None if result.embeddings is None else result.embeddings[source_indices]),
        )

    return result
