"""Model-dispatched adaptive inference."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from al_pipeline.astronomy.astrominn.inference.astrominn import AstroMiNNInferenceResult
from al_pipeline.astronomy.astrominn.inference.astrominn_vit import predict_astrominn_vit_pool


def run_inference(
    *,
    model: str,
    pool_path: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
    object_ids: Iterable[str] | None = None,
    checkpoint_path: str | Path | None = None,
) -> AstroMiNNInferenceResult:
    """Run a named model over a prepared adaptive candidate pool."""

    if model == "astrominn_vit135":
        return predict_astrominn_vit_pool(
            pool_path=pool_path,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            object_ids=object_ids,
            vit_checkpoint=checkpoint_path,
        )

    # TODO:
    # Once the canonical current AstroMiNN checkpoint is identified,
    # expose it here as "astrominn". Keep the historical ViT name explicit
    # so experiments remain reproducible.

    raise ValueError(f"Unknown adaptive inference model: {model!r}.")
