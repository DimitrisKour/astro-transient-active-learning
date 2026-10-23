"""Opt-in ViT adaptation arms: frozen expert towers throughout.

The ordinary AstroMiNN trainer saves only model.vit_tower.state_dict().
We deliberately do not unfreeze expert towers under that checkpoint contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

MODES = ("vit_full", "vit_head_only")


def make_vit_factory(mode: str, *, artifact_dir: str | Path, legacy_code_dir: str | Path) -> Callable:
    """Create an existing AstroMiNN trainer-compatible model factory.

    vit_full: existing full ViT trainable behavior.
    vit_head_only: freeze all ViT parameters except fc_out; CLS embeddings fixed.
    In both modes the stock trainer additionally freezes all three experts.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown ViT representation adaptation: {mode!r}")
    artifact_dir = Path(artifact_dir)
    legacy_code_dir = Path(legacy_code_dir)

    def factory(load_vit_checkpoint: bool, checkpoint: Path | None):
        from al_pipeline.astronomy.astrominn.inference.astrominn_vit import load_astrominn_vit_model
        model = load_astrominn_vit_model(
            artifact_dir=artifact_dir,
            legacy_code_dir=legacy_code_dir,
            load_vit_checkpoint=load_vit_checkpoint,
            vit_checkpoint=checkpoint,
        )
        model._applecider_vit_head_only = (mode == "vit_head_only")
        if mode == "vit_head_only":
            for parameter in model.vit_tower.parameters():
                parameter.requires_grad_(False)
            head = model.vit_tower.fc_out
            if not hasattr(head, "parameters"):
                raise TypeError("AstroMiNN vit_tower.fc_out is not a trainable module")
            head_parameters = list(head.parameters())
            if not head_parameters:
                raise ValueError("AstroMiNN final layer has no parameters")
            for parameter in head_parameters:
                parameter.requires_grad_(True)
        return model

    return factory
