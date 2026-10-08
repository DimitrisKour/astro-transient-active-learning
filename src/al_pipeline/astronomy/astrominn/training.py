"""Reproducible training adapter for the historical seed-135 AstroMiNN ViT.

The implementation preserves the scientifically relevant behavior recovered
from the 2026-01-22 W&B run while keeping it behind the generic training
contract. Historical manifests are consumed directly; the original raw-file
split is never regenerated during adaptive-learning cycles.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from al_pipeline.astronomy.data.registry import DatasetSpec
from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_LEGACY_CODE_DIR,
    load_astrominn_vit_config,
    load_astrominn_vit_model,
    preprocess_astrominn_vit,
)
from al_pipeline.core.training.base import TrainingRequest, TrainingResult


MODEL_CLASS_NAMES = (
    "AGN",
    "TDE",
    "SN I",
    "SN II",
    "CV",
)

CONFIG_CLASS_NAMES = (
    "Nuclear",
    "Tidal Disruption Event",
    "SN I",
    "SN II",
    "Cataclysmic",
)

# Both the historical native labels and the resolver's canonical labels are
# accepted. This lets a cumulative L_t manifest retain historical provenance
# while appending newly resolved AppleCiDEr labels.
LABEL_TO_INDEX = {
    "AGN": 0,
    "Nuclear": 0,
    "TDE": 1,
    "Tidal Disruption Event": 1,
    "SN I": 2,
    "SN Ia": 2,
    "SN Ib": 2,
    "SN Ic": 2,
    "SN II": 3,
    "SN IIn": 3,
    "SN IIp": 3,
    "SN IIP": 3,
    "CV": 4,
    "Cataclysmic": 4,
}

# Sample-level counts printed by the exact W&B run. The legacy trainer used
# all max-alert-compatible train/validation/test samples when constructing
# its class weights, rather than training rows alone.
LEGACY_CLASS_COUNT_REFERENCE = (
    5_678,
    239,
    40_228,
    8_968,
    2_375,
)

InitializationMode = Literal[
    "pretrained",
    "random_vit",
]

NewObjectWeighting = Literal[
    "per_sample",
    "equal_object",
]

ClassWeightSource = Literal[
    "legacy_reference",
    "training_manifest",
]

ModelFactory = Callable[
    [bool, Path | None],
    nn.Module,
]


@dataclass(frozen=True, slots=True)
class AstroMiNNViTTrainingConfig:
    """Optimization policy for one AstroMiNN ViT training request."""

    batch_size: int = 512
    validation_batch_size: int = 2_048
    max_epochs: int = 1_000
    patience: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    beta1: float = 0.99
    beta2: float = 0.99
    optimizer_epsilon: float = 5e-10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.4
    minimum_learning_rate: float = 5e-10
    gradient_clip_norm: float = 9.0
    class_weight_exponent: float = 0.85
    class_weight_numerator: float = 5_000.0
    class_count_reference: tuple[int, ...] = LEGACY_CLASS_COUNT_REFERENCE
    loader_seed: int = 125
    train_num_workers: int = 6
    validation_num_workers: int = 3
    augment_training: bool = True
    augment_validation: bool = False
    cache_samples_in_memory: bool = True
    legacy_frozen_expert_train_mode: bool = False
    legacy_seed_sequence: bool = False
    deterministic: bool = True
    overwrite: bool = False
    new_object_weighting: NewObjectWeighting = "per_sample"
    class_weight_source: ClassWeightSource = "legacy_reference"

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "batch_size",
            "validation_batch_size",
            "max_epochs",
            "patience",
            "scheduler_patience",
        )

        for name in positive_integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

        for name in (
            "train_num_workers",
            "validation_num_workers",
            "loader_seed",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")

        if len(self.class_count_reference) != len(MODEL_CLASS_NAMES):
            raise ValueError("class_count_reference must contain one count per class.")

        if any(count <= 0 for count in self.class_count_reference):
            raise ValueError("All class-count references must be positive.")

        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor must be between zero and one.")

        if self.new_object_weighting not in {
            "per_sample",
            "equal_object",
        }:
            raise ValueError("new_object_weighting must be 'per_sample' or 'equal_object'.")

        if self.class_weight_source not in {
            "legacy_reference",
            "training_manifest",
        }:
            raise ValueError(
                "class_weight_source must be 'legacy_reference' or 'training_manifest'."
            )

    @classmethod
    def legacy_seed135(
        cls,
        **overrides: Any,
    ) -> AstroMiNNViTTrainingConfig:
        """Return settings that reproduce the 2026-01-22 procedure.

        The original code augmented validation samples and left the frozen
        expert modules in training mode. These behaviors are retained only
        by this explicit compatibility constructor.
        """

        values: dict[str, Any] = {
            "augment_validation": True,
            "legacy_frozen_expert_train_mode": True,
            "legacy_seed_sequence": True,
        }
        values.update(overrides)
        return cls(**values)


class AstroMiNNViTManifestDataset(Dataset):
    """Load model-ready AstroMiNN samples from a run manifest."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        dataset: DatasetSpec,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        augment: bool = False,
        cache_in_memory: bool = True,
        object_sample_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset = dataset
        self.artifact_dir = Path(artifact_dir)
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        self.object_sample_weights = {
            str(object_id): float(weight)
            for object_id, weight in (object_sample_weights or {}).items()
        }

        invalid_weights = {
            object_id: weight
            for object_id, weight in self.object_sample_weights.items()
            if not np.isfinite(weight) or weight <= 0
        }

        if invalid_weights:
            raise ValueError(
                f"Object sample weights must be positive and finite: {invalid_weights}."
            )

        if dataset.object_id_column is None:
            raise ValueError("DatasetSpec requires object_id_column.")

        if dataset.sample_file_column is None:
            raise ValueError("DatasetSpec requires sample_file_column.")

        if dataset.label_column is None:
            raise ValueError("DatasetSpec requires label_column.")

        self.object_id_column = dataset.object_id_column
        self.sample_file_column = dataset.sample_file_column
        self.label_column = dataset.label_column
        self.rows = _read_manifest(self.manifest_path)
        self.model_config = load_astrominn_vit_config(self.artifact_dir)
        _validate_model_class_order(self.model_config)
        self.augmentation = _legacy_image_augmentation() if augment else None

        required = {
            self.object_id_column,
            self.sample_file_column,
            self.label_column,
        }
        missing = required.difference(self.rows[0])

        if missing:
            raise ValueError(
                f"Manifest {self.manifest_path} is missing columns: {sorted(missing)}."
            )

        self._cached_items = (
            tuple(self._load_manifest_row(row) for row in self.rows) if cache_in_memory else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = (
            self._cached_items[index]
            if self._cached_items is not None
            else self._load_manifest_row(self.rows[index])
        )
        image = item["image"].clone()

        if self.augmentation is not None:
            image = self.augmentation(image)

        return {
            **item,
            "image": image,
        }

    def _load_manifest_row(
        self,
        row: Mapping[str, str],
    ) -> dict[str, Any]:
        object_id = str(row[self.object_id_column])
        label = str(row[self.label_column])

        try:
            target_index = LABEL_TO_INDEX[label]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported AstroMiNN training label {label!r} in {self.manifest_path}."
            ) from exc

        sample_path = self._resolve_sample_path(row[self.sample_file_column])
        sample = np.load(
            sample_path,
            allow_pickle=True,
        ).item()

        if not isinstance(sample, Mapping):
            raise TypeError(f"AstroMiNN sample is not a mapping: {sample_path}")

        sample_object_id = str(sample.get("obj_id", ""))

        if sample_object_id != object_id:
            raise ValueError(
                f"Sample object ID {sample_object_id!r} does not match "
                f"manifest object ID {object_id!r}: {sample_path}"
            )

        metadata, image = preprocess_astrominn_vit(
            metadata=np.asarray(sample["metadata"]),
            image=np.asarray(sample["image"]),
            config=self.model_config,
        )

        target = torch.zeros(
            len(MODEL_CLASS_NAMES),
            dtype=torch.float32,
        )
        target[target_index] = 1.0

        alert_value = sample.get(
            "alerte",
            sample.get("alert", row.get("alert")),
        )

        if alert_value is None:
            raise ValueError(f"Sample has no alert provenance: {sample_path}")

        if "alert" in row and row["alert"] != "":
            if int(row["alert"]) != int(alert_value):
                raise ValueError(
                    f"Sample alert {alert_value} does not match manifest "
                    f"alert {row['alert']} for {object_id}."
                )

        sample_label = sample.get("target")

        if sample_label is not None:
            sample_index = LABEL_TO_INDEX.get(str(sample_label))

            if sample_index is not None and sample_index != target_index:
                raise ValueError(
                    f"Sample label {sample_label!r} conflicts with manifest "
                    f"label {label!r} for {object_id}."
                )

        return {
            "obj_id": object_id,
            "alert": int(alert_value),
            "metadata": metadata,
            "image": image,
            # Preserve the legacy one-hot float target. CrossEntropyLoss has
            # different mean-reduction normalization for float targets than
            # for integer class indices, so this is scientifically relevant.
            "target": target,
            "target_index": target_index,
            "sample_weight": torch.tensor(
                self.object_sample_weights.get(object_id, 1.0),
                dtype=torch.float32,
            ),
        }

    def _resolve_sample_path(self, value: object) -> Path:
        path = Path(str(value))

        if not path.is_absolute():
            if self.dataset.data_root is None:
                raise ValueError(
                    f"Cannot resolve relative sample path {path}: DatasetSpec has no data_root."
                )

            path = self.dataset.data_root / path

        if not path.exists():
            raise FileNotFoundError(path)

        return path


class AstroMiNNViTTrainer:
    """Train or fine-tune only the seed-135 ViT classification tower."""

    def __init__(
        self,
        *,
        dataset: DatasetSpec,
        policy: AstroMiNNViTTrainingConfig | None = None,
        device: str | torch.device = "cpu",
        initialization: InitializationMode = "pretrained",
        legacy_code_dir: str | Path = DEFAULT_LEGACY_CODE_DIR,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if initialization not in {"pretrained", "random_vit"}:
            raise ValueError("initialization must be 'pretrained' or 'random_vit'.")

        self.dataset = dataset
        self.policy = policy or AstroMiNNViTTrainingConfig()
        self.device = torch.device(device)
        self.initialization = initialization
        self.legacy_code_dir = Path(legacy_code_dir)
        self.artifact_dir = Path(artifact_dir)
        self.model_factory = model_factory

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Execute one reproducible training or fine-tuning request."""

        self._validate_request(request)
        request.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = request.output_dir / "best_vit_tower.pth"
        last_state_path = request.output_dir / "last_training_state.pth"
        history_path = request.output_dir / "training_history.json"
        summary_path = request.output_dir / "training_summary.json"

        if summary_path.exists() and not self.policy.overwrite:
            raise FileExistsError(f"Training output already exists: {summary_path}")

        self._seed_process(request.seed)

        object_sample_weights, weighting_summary = _new_object_training_weights(
            train_manifest=request.train_manifest,
            historical_manifest=self.dataset.train_manifest,
            object_id_column=self.dataset.object_id_column,
            mode=self.policy.new_object_weighting,
        )
        training_dataset = AstroMiNNViTManifestDataset(
            manifest_path=request.train_manifest,
            dataset=self.dataset,
            artifact_dir=self.artifact_dir,
            augment=self.policy.augment_training,
            cache_in_memory=(self.policy.cache_samples_in_memory),
            object_sample_weights=object_sample_weights,
        )
        validation_dataset = AstroMiNNViTManifestDataset(
            manifest_path=request.validation_manifest,
            dataset=self.dataset,
            artifact_dir=self.artifact_dir,
            augment=self.policy.augment_validation,
            cache_in_memory=(self.policy.cache_samples_in_memory),
        )

        generator: torch.Generator | None = None

        if not self.policy.legacy_seed_sequence:
            generator = torch.Generator()
            generator.manual_seed(self.policy.loader_seed)

        train_loader = DataLoader(
            training_dataset,
            batch_size=self.policy.batch_size,
            shuffle=True,
            num_workers=self.policy.train_num_workers,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.policy.validation_batch_size,
            shuffle=False,
            num_workers=self.policy.validation_num_workers,
            pin_memory=self.device.type == "cuda",
        )

        model = self._build_model(request)
        model.to(self.device)
        _freeze_expert_towers(model)

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError("AstroMiNN has no trainable parameters.")

        optimizer = AdamW(
            trainable_parameters,
            lr=self.policy.learning_rate,
            weight_decay=self.policy.weight_decay,
            betas=(
                self.policy.beta1,
                self.policy.beta2,
            ),
            eps=self.policy.optimizer_epsilon,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            min_lr=self.policy.minimum_learning_rate,
            patience=self.policy.scheduler_patience,
            factor=self.policy.scheduler_factor,
        )
        class_weights, class_weighting_summary = _class_weights(
            policy=self.policy,
            device=self.device,
            train_manifest=request.train_manifest,
            label_column=self.dataset.label_column,
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",
        )

        history: list[dict[str, Any]] = []
        best_validation_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        start = time.perf_counter()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch_index in range(self.policy.max_epochs):
            epoch = epoch_index + 1
            _set_training_mode(
                model,
                legacy_frozen_expert_train_mode=(self.policy.legacy_frozen_expert_train_mode),
            )
            training_loss = _train_epoch(
                loader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                trainable_parameters=trainable_parameters,
                device=self.device,
                gradient_clip_norm=(self.policy.gradient_clip_norm),
            )

            # The historical trainer performed separate validation passes for
            # AUPRC and loss. This matters in compatibility mode because its
            # validation samples were randomly augmented on every access.
            macro_auprc, class_auprcs = _evaluate_auprc(
                loader=validation_loader,
                model=model,
                device=self.device,
            )
            validation_loss = _evaluate_loss(
                loader=validation_loader,
                model=model,
                criterion=criterion,
                device=self.device,
            )
            scheduler.step(validation_loss)

            improved = validation_loss < best_validation_loss

            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                _torch_save_atomic(
                    model.vit_tower.state_dict(),
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            epoch_record = {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_macro_auprc": macro_auprc,
                "validation_class_auprc": {
                    class_name: score
                    for class_name, score in zip(
                        MODEL_CLASS_NAMES,
                        class_auprcs,
                    )
                },
                "learning_rate": optimizer.param_groups[0]["lr"],
                "best_validation_loss": best_validation_loss,
                "improved": improved,
            }
            history.append(epoch_record)
            _json_dump_atomic(
                history,
                history_path,
            )
            _torch_save_atomic(
                {
                    "epoch": epoch,
                    "vit_tower_state_dict": (model.vit_tower.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "epochs_without_improvement": (epochs_without_improvement),
                },
                last_state_path,
            )

            if epochs_without_improvement >= self.policy.patience:
                break

        elapsed_seconds = time.perf_counter() - start
        peak_cuda_memory_bytes = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else None
        )

        summary = {
            "run_id": request.run_id,
            "cycle": request.cycle,
            "strategy": request.strategy,
            "seed": request.seed,
            "initialization": self.initialization,
            "initial_checkpoint": (
                str(request.initial_checkpoint) if request.initial_checkpoint is not None else None
            ),
            "train_manifest": str(request.train_manifest),
            "train_manifest_sha256": _sha256_file(request.train_manifest),
            "validation_manifest": str(request.validation_manifest),
            "validation_manifest_sha256": _sha256_file(request.validation_manifest),
            "training_samples": len(training_dataset),
            "validation_samples": len(validation_dataset),
            "model_class_names": list(MODEL_CLASS_NAMES),
            "policy": asdict(self.policy),
            "new_object_weighting": weighting_summary,
            "class_weighting": class_weighting_summary,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "best_validation_loss": best_validation_loss,
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(history_path),
            "last_training_state_path": str(last_state_path),
        }
        _json_dump_atomic(
            summary,
            summary_path,
        )

        return TrainingResult(
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            summary_path=summary_path,
            best_epoch=best_epoch,
            epochs_completed=len(history),
            best_validation_loss=best_validation_loss,
            elapsed_seconds=elapsed_seconds,
            peak_cuda_memory_bytes=peak_cuda_memory_bytes,
        )

    def completed_result(
        self,
        request: TrainingRequest,
    ) -> TrainingResult | None:
        """Recover a completed cycle after an orchestrator restart.

        The stored experiment identity and manifest hashes are checked before
        artifacts are reused. This prevents a run directory from silently
        accepting results produced from different labeled data.
        """

        summary_path = request.output_dir / "training_summary.json"

        if not summary_path.exists():
            return None

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        expected_identity = {
            "run_id": request.run_id,
            "cycle": request.cycle,
            "strategy": request.strategy,
            "seed": request.seed,
            "train_manifest_sha256": _sha256_file(request.train_manifest),
            "validation_manifest_sha256": _sha256_file(request.validation_manifest),
            "initial_checkpoint": (
                str(request.initial_checkpoint) if request.initial_checkpoint is not None else None
            ),
            "initialization": self.initialization,
            "policy": json.loads(json.dumps(asdict(self.policy))),
        }

        mismatches = {
            key: (summary.get(key), value)
            for key, value in expected_identity.items()
            if summary.get(key) != value
        }

        if mismatches:
            raise ValueError(
                f"Existing training output does not match the requested cycle: {mismatches}."
            )

        result = TrainingResult(
            checkpoint_path=Path(summary["checkpoint_path"]),
            history_path=Path(summary["history_path"]),
            summary_path=summary_path,
            best_epoch=int(summary["best_epoch"]),
            epochs_completed=int(summary["epochs_completed"]),
            best_validation_loss=float(summary["best_validation_loss"]),
            elapsed_seconds=float(summary["elapsed_seconds"]),
            peak_cuda_memory_bytes=(
                None
                if summary.get("peak_cuda_memory_bytes") is None
                else int(summary["peak_cuda_memory_bytes"])
            ),
        )

        missing = [
            path
            for path in (
                result.checkpoint_path,
                result.history_path,
                result.summary_path,
            )
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Completed training summary references missing artifacts: "
                + ", ".join(str(path) for path in missing)
            )

        return result

    def _build_model(
        self,
        request: TrainingRequest,
    ) -> nn.Module:
        checkpoint = request.initial_checkpoint
        load_vit_checkpoint = checkpoint is not None or self.initialization == "pretrained"

        if self.model_factory is not None:
            return self.model_factory(
                load_vit_checkpoint,
                checkpoint,
            )

        return load_astrominn_vit_model(
            legacy_code_dir=self.legacy_code_dir,
            artifact_dir=self.artifact_dir,
            load_vit_checkpoint=load_vit_checkpoint,
            vit_checkpoint=checkpoint,
        )

    def _seed_process(self, experiment_seed: int) -> None:
        seed = self.policy.loader_seed if self.policy.legacy_seed_sequence else experiment_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if self.policy.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _validate_request(self, request: TrainingRequest) -> None:
        for path in (
            request.train_manifest,
            request.validation_manifest,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        if request.initial_checkpoint is not None:
            if not request.initial_checkpoint.exists():
                raise FileNotFoundError(request.initial_checkpoint)

        if self.initialization == "random_vit" and request.initial_checkpoint is not None:
            raise ValueError(
                "random_vit initialization cannot be combined with an initial checkpoint."
            )


def _legacy_image_augmentation() -> transforms.Compose:
    """Return the exact image transform used by the legacy trainer.

    The two rotations inside ``RandomApply`` are intentionally retained in
    sequence. Together they produce a full 360-degree rotation; changing this
    would no longer reproduce the historical procedure.
    """

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.Lambda(
                        lambda image: torch.rot90(
                            image,
                            k=1,
                            dims=(-2, -1),
                        )
                    ),
                    transforms.Lambda(
                        lambda image: torch.rot90(
                            image,
                            k=3,
                            dims=(-2, -1),
                        )
                    ),
                ],
                p=0.75,
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=5,
                        sigma=(0.0005, 0.0005),
                    )
                ],
                p=1.0,
            ),
        ]
    )


def _validate_model_class_order(
    config: Mapping[str, Any],
) -> None:
    configured = tuple(config["classes"].keys())

    if configured != CONFIG_CLASS_NAMES:
        raise ValueError(
            f"AstroMiNN model class order changed: {configured!r} != {CONFIG_CLASS_NAMES!r}."
        )


def _freeze_expert_towers(model: nn.Module) -> None:
    for module_name in (
        "lc_tower",
        "spatial_tower",
        "coord_tower",
    ):
        module = getattr(model, module_name)

        for parameter in module.parameters():
            parameter.requires_grad = False

    for parameter in model.vit_tower.parameters():
        parameter.requires_grad = True


def _set_training_mode(
    model: nn.Module,
    *,
    legacy_frozen_expert_train_mode: bool,
) -> None:
    model.train()

    if legacy_frozen_expert_train_mode:
        return

    # Frozen parameters alone do not freeze BatchNorm buffers or dropout.
    # Explicit evaluation mode prevents the historical hidden state drift.
    model.lc_tower.eval()
    model.spatial_tower.eval()
    model.coord_tower.eval()
    model.vit_tower.train()


def _legacy_class_weights(
    policy: AstroMiNNViTTrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            policy.class_weight_numerator / (count**policy.class_weight_exponent)
            for count in policy.class_count_reference
        ],
        dtype=torch.float32,
        device=device,
    )


def _class_weights(
    *,
    policy: AstroMiNNViTTrainingConfig,
    device: torch.device,
    train_manifest: Path,
    label_column: str | None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return class weights without exposing hidden-pool label counts."""

    if policy.class_weight_source == "legacy_reference":
        weights = _legacy_class_weights(policy, device)
        return weights, {
            "source": "legacy_reference",
            "sample_counts": {
                class_name: int(count)
                for class_name, count in zip(
                    MODEL_CLASS_NAMES,
                    policy.class_count_reference,
                    strict=True,
                )
            },
            "weights": {
                class_name: float(value)
                for class_name, value in zip(
                    MODEL_CLASS_NAMES,
                    weights.detach().cpu().tolist(),
                    strict=True,
                )
            },
        }

    if label_column is None:
        raise ValueError("DatasetSpec requires label_column for manifest class weights.")

    rows = _read_manifest(train_manifest)
    counts = [0] * len(MODEL_CLASS_NAMES)
    for row in rows:
        label = str(row[label_column])
        try:
            counts[LABEL_TO_INDEX[label]] += 1
        except KeyError as exc:
            raise ValueError(
                f"Unsupported AstroMiNN training label {label!r} in {train_manifest}."
            ) from exc

    missing = [MODEL_CLASS_NAMES[index] for index, count in enumerate(counts) if count == 0]
    if missing:
        raise ValueError(
            f"Manifest-derived class weighting requires every class in L_t; missing {missing}."
        )

    raw = torch.tensor(
        [count ** (-policy.class_weight_exponent) for count in counts],
        dtype=torch.float32,
        device=device,
    )
    weights = raw / raw.mean()
    return weights, {
        "source": "training_manifest",
        "normalization": "unit_mean",
        "sample_counts": {
            class_name: int(count)
            for class_name, count in zip(
                MODEL_CLASS_NAMES,
                counts,
                strict=True,
            )
        },
        "weights": {
            class_name: float(value)
            for class_name, value in zip(
                MODEL_CLASS_NAMES,
                weights.detach().cpu().tolist(),
                strict=True,
            )
        },
    }


def _new_object_training_weights(
    *,
    train_manifest: Path,
    historical_manifest: Path | None,
    object_id_column: str | None,
    mode: NewObjectWeighting,
) -> tuple[dict[str, float], dict[str, object]]:
    """Weight new samples so each acquired object has equal total weight.

    Historical rows retain weight one. In ``equal_object`` mode, the total
    weight of every newly acquired object equals the historical mean number
    of samples per object. This preserves the established historical loss
    while preventing variable BOOM archive coverage from changing an
    acquired object's total contribution.
    """

    if mode == "per_sample":
        return {}, {
            "mode": mode,
            "applied": False,
        }

    if mode != "equal_object":
        raise ValueError(f"Unsupported new-object weighting mode: {mode!r}.")

    if historical_manifest is None:
        raise ValueError("equal_object weighting requires the historical train manifest.")

    if object_id_column is None:
        raise ValueError("equal_object weighting requires dataset.object_id_column.")

    historical_rows = _read_manifest(historical_manifest)
    training_rows = _read_manifest(train_manifest)
    historical_ids = {str(row[object_id_column]) for row in historical_rows}

    if not historical_ids:
        raise ValueError("Historical train manifest contains no objects.")

    historical_mean = len(historical_rows) / len(historical_ids)
    new_counts = Counter(
        str(row[object_id_column])
        for row in training_rows
        if str(row[object_id_column]) not in historical_ids
    )
    weights = {object_id: historical_mean / count for object_id, count in new_counts.items()}

    if weights:
        minimum = min(weights.values())
        maximum = max(weights.values())
    else:
        minimum = None
        maximum = None

    return weights, {
        "mode": mode,
        "applied": bool(weights),
        "historical_samples": len(historical_rows),
        "historical_objects": len(historical_ids),
        "historical_mean_samples_per_object": historical_mean,
        "new_objects": len(new_counts),
        "new_samples": sum(new_counts.values()),
        "new_object_total_weight": historical_mean,
        "minimum_new_sample_weight": minimum,
        "maximum_new_sample_weight": maximum,
    }


def _train_epoch(
    *,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: list[nn.Parameter],
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    total_loss = 0.0

    for batch in loader:
        metadata = batch["metadata"].to(
            device,
            non_blocking=True,
        )
        image = batch["image"].to(
            device,
            non_blocking=True,
        )
        target = batch["target"].to(
            device,
            non_blocking=True,
        )
        sample_weight = batch["sample_weight"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)
        output = model(metadata, image=image)
        logits = _extract_logits(output)
        per_sample_loss = criterion(logits, target)

        if per_sample_loss.ndim != 1:
            raise RuntimeError("AstroMiNN training loss must contain one value per sample.")

        if per_sample_loss.shape != sample_weight.shape:
            raise RuntimeError("Training loss and sample weights have different shapes.")

        loss = torch.sum(per_sample_loss * sample_weight) / torch.sum(sample_weight)

        if not torch.isfinite(loss):
            raise RuntimeError("AstroMiNN produced a non-finite loss.")

        loss.backward()
        nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=gradient_clip_norm,
        )
        optimizer.step()
        total_loss += float(loss.detach().cpu())

    if len(loader) == 0:
        raise ValueError("Cannot train from an empty dataloader.")

    return total_loss / len(loader)


def _evaluate_loss(
    *,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.inference_mode():
        for batch in loader:
            metadata = batch["metadata"].to(
                device,
                non_blocking=True,
            )
            image = batch["image"].to(
                device,
                non_blocking=True,
            )
            target = batch["target"].to(
                device,
                non_blocking=True,
            )
            logits = _extract_logits(model(metadata, image=image))
            per_sample_loss = criterion(logits, target)

            if per_sample_loss.ndim != 1:
                raise RuntimeError("AstroMiNN validation loss must contain one value per sample.")

            total_loss += float(per_sample_loss.mean().detach().cpu())

    if len(loader) == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    return total_loss / len(loader)


def _evaluate_auprc(
    *,
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
) -> tuple[float, list[float]]:
    model.eval()
    target_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            metadata = batch["metadata"].to(
                device,
                non_blocking=True,
            )
            image = batch["image"].to(
                device,
                non_blocking=True,
            )
            logits = _extract_logits(model(metadata, image=image))
            probabilities = torch.softmax(
                logits,
                dim=1,
            )
            target_batches.append(batch["target"].cpu())
            probability_batches.append(probabilities.cpu())

    if not target_batches:
        raise ValueError("Cannot evaluate an empty dataloader.")

    targets = torch.cat(target_batches).numpy()
    probabilities = torch.cat(probability_batches).numpy()
    class_auprcs: list[float] = []

    for class_index in range(len(MODEL_CLASS_NAMES)):
        precision, recall, _ = precision_recall_curve(
            targets[:, class_index],
            probabilities[:, class_index],
        )
        class_auprcs.append(float(auc(recall, precision)))

    return float(np.mean(class_auprcs)), class_auprcs


def _extract_logits(output: object) -> torch.Tensor:
    if not isinstance(output, Mapping):
        raise TypeError("AstroMiNN output must be a mapping.")

    if "logits" not in output:
        raise KeyError("AstroMiNN output has no 'logits' entry.")

    logits = output["logits"]

    if not isinstance(logits, torch.Tensor):
        raise TypeError("AstroMiNN logits must be a tensor.")

    if logits.ndim != 2 or logits.shape[1] != len(MODEL_CLASS_NAMES):
        raise ValueError("AstroMiNN logits must have shape (batch, 5).")

    return logits


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")

        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")

    return rows


def _json_dump_atomic(
    payload: object,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _torch_save_atomic(
    payload: object,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


__all__ = [
    "AstroMiNNViTManifestDataset",
    "AstroMiNNViTTrainer",
    "AstroMiNNViTTrainingConfig",
    "CONFIG_CLASS_NAMES",
    "LABEL_TO_INDEX",
    "LEGACY_CLASS_COUNT_REFERENCE",
    "MODEL_CLASS_NAMES",
]
