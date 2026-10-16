"""Model-specific feature extraction for Task 5A AstroMiNN diagnostics.

Feature hooks do not change the published expert towers or ViT weights.
The 'vit_cls' representation is the input to vit_tower.fc_out, matching the
existing active-learning embedding capture. Expert captures use tower outputs
and are validated on a real CPU smoke run before full-scale extraction.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from al_pipeline.astronomy.representation.core import REPRESENTATIONS


class FeatureRecorder:
    """Context-managed, checked forward hooks on AstroMiNN feature boundaries."""

    def __init__(self, model: nn.Module, representations: Sequence[str]):
        self.model = model
        self.representations = tuple(representations)
        if not self.representations or len(set(self.representations)) != len(self.representations):
            raise ValueError("Choose distinct representation names")
        if set(self.representations) - set(REPRESENTATIONS):
            raise ValueError(f"Unknown representations: {self.representations}")
        self._handles: list[Any] = []
        self._captured: dict[str, list[torch.Tensor]] = {}

    def __enter__(self) -> "FeatureRecorder":
        if "vit_cls" in self.representations:
            fc = self.model.vit_tower.fc_out
            self._handles.append(fc.register_forward_pre_hook(self._pre_hook("vit_cls")))
        if any(name != "vit_cls" for name in self.representations):
            for name, module in (("lc_expert", "lc_tower"),
                                 ("spatial_expert", "spatial_tower"),
                                 ("coord_expert", "coord_tower")):
                if name in self.representations or "expert_concat" in self.representations:
                    self._handles.append(getattr(self.model, module).register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()

    def _record(self, key: str, value: object) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"Representation {key!r} must be a rank-2 tensor; got {type(value)} "
                             f"shape={getattr(value, 'shape', None)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite {key} embedding")
        self._captured.setdefault(key, []).append(value.detach().cpu())

    def _pre_hook(self, key: str):
        def callback(_module: nn.Module, inputs: tuple[object, ...]) -> None:
            if len(inputs) != 1:
                raise ValueError(f"Expected exactly one input to {key}")
            self._record(key, inputs[0])
        return callback

    def _hook(self, key: str):
        def callback(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            self._record(key, output)
        return callback

    def pop_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        requested = set(self.representations)
        needed = requested - {"expert_concat"}
        if "expert_concat" in requested:
            needed.update(("lc_expert", "spatial_expert", "coord_expert"))
        result = {}
        for key in needed:
            matches = self._captured.get(key, [])
            if len(matches) != 1 or matches[0].shape[0] != batch_size:
                raise RuntimeError(f"{key}: expected one capture of batch {batch_size}, "
                                   f"observed {[tuple(x.shape) for x in matches]}. "
                                   "Inspect historical expert forward outputs before using this representation.")
            result[key] = matches[0].numpy().astype(np.float32, copy=False)
        if "expert_concat" in requested:
            result["expert_concat"] = np.concatenate(
                [result[key] for key in ("lc_expert", "spatial_expert", "coord_expert")], axis=1)
        self._captured.clear()
        return {key: result[key] for key in self.representations}


def extract_features(model: nn.Module, dataloader: DataLoader, *,
                     representations: Sequence[str], device: str = "cpu") -> dict[str, np.ndarray]:
    """Extract one row per dataloader sample, preserving IDs and target indices."""
    if device != "cpu":
        raise ValueError("Task 5A extraction is deliberately CPU-only")
    model = model.to(device)
    model.eval()
    features: dict[str, list[np.ndarray]] = {key: [] for key in representations}
    ids: list[str] = []
    labels: list[int] = []
    alerts: list[int] = []
    with FeatureRecorder(model, representations) as recorder:
        with torch.inference_mode():
            for batch in dataloader:
                metadata = batch["metadata"].to(device)
                image = batch["image"].to(device)
                prediction = model(metadata, image=image)
                if not isinstance(prediction, Mapping) or "logits" not in prediction:
                    raise TypeError("AstroMiNN model must return {'logits': ...}")
                n = len(batch["obj_id"])
                if prediction["logits"].shape[0] != n:
                    raise ValueError("AstroMiNN output count differs from batch")
                captured = recorder.pop_batch(n)
                for key, value in captured.items():
                    features[key].append(value)
                ids.extend(str(x) for x in batch["obj_id"])
                for field, output in (("target_index", labels), ("alert", alerts)):
                    values = batch[field]
                    output.extend(int(v) for v in values)
    if not ids:
        raise ValueError("Empty feature extraction dataloader")
    if len(set(ids)) != len(ids):
        raise ValueError("Task5 requires exactly one alert per object, got duplicates")
    return {"object_ids": np.asarray(ids, dtype=str),
            "labels": np.asarray(labels, dtype=np.int64),
            "alerts": np.asarray(alerts, dtype=np.int64),
            **{key: np.concatenate(value, axis=0) for key, value in features.items()}}
