"""Model factory for new AstroMiNN image+metadata, no TEMPO or fusion.

Weights and architecture of the upstream ``fusion_model_basics`` checkout are
not copied or modified. Checkpointing always stores the complete model state.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from .contracts import CLASS_NAMES, class_order_config, sha256


@dataclass(frozen=True)
class NewImageModelSpec:
    variant: Literal['patch_images', 'btsbot_images'] = 'patch_images'
    image_initialization: Literal['pretrained', 'randomized'] = 'pretrained'
    hf_snapshot: str | None = None
    hf_weights_sha256: str | None = None
    freeze_image: bool = False
    source_revision: str = 'UNVERIFIED'

    def __post_init__(self):
        if self.variant not in ('patch_images', 'btsbot_images'):
            raise ValueError('Only image + metadata AstroMiNN variants are supported')
        if self.variant == 'btsbot_images' and not self.hf_snapshot:
            raise ValueError('BTSbot requires a local validated Hugging Face snapshot')
        if self.variant == 'patch_images' and self.image_initialization == 'pretrained':
            raise ValueError('patch_images has no image-pretraining weights; use randomized')
        if self.source_revision == 'UNVERIFIED':
            raise ValueError('Provide the exact fusion_model_basics git SHA')


def build_model(spec: NewImageModelSpec, *, seed: int, device: str | torch.device = 'cpu'):
    import fusion_model_basics
    source = getattr(fusion_model_basics, '__file__', None)
    if source is not None:
        actual = subprocess.check_output(['git', '-C', str(Path(source).parent),
                                          'rev-parse', 'HEAD'], text=True).strip()
        if actual != spec.source_revision:
            raise ValueError(f'Imported upstream revision {actual} != requested {spec.source_revision}')
    from fusion_model_basics.config import build_astrominn_config
    from fusion_model_basics.astrominn_encoder import AstroMiNNNPZClassifier

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    cfg = build_astrominn_config(
        spec.variant, coord_checkpoint=None,
        hf_image_model_dir=spec.hf_snapshot, device=str(device), dir='.',
        classes=class_order_config(),
    )
    model = AstroMiNNNPZClassifier(cfg)
    if tuple(model.class_names) != tuple(CLASS_NAMES):
        raise ValueError(f'Wrong class order: {model.class_names}')
    if spec.variant == 'btsbot_images':
        image_encoder = model.vit_tower.image_encoder
        backbone = image_encoder.backbone
        # Upstream loads the pretrained timm backbone with strict=False.
        # Count actual matching parameter elements to prevent silent fallback
        # to a nearly random backbone after an upstream dependency change.
        snapshot = Path(spec.hf_snapshot)
        weights_file = snapshot/'pytorch_model.bin'
        if not weights_file.is_file():
            raise FileNotFoundError(weights_file)
        imported = torch.load(weights_file, map_location='cpu', weights_only=True)
        converted = image_encoder._convert_state_dict(imported)
        expected = backbone.state_dict()
        matched = sum(t.numel() for k, t in expected.items()
                      if k in converted and converted[k].shape == t.shape
                      and torch.is_floating_point(t))
        total = sum(t.numel() for t in expected.values() if torch.is_floating_point(t))
        if total == 0 or matched / total < 0.95:
            raise RuntimeError(f'BTSbot pretrained backbone tensor coverage {matched}/{total} '
                               'is below 95%; do not trust this image pretraining')
        if spec.hf_weights_sha256 and sha256(weights_file) != spec.hf_weights_sha256:
            raise ValueError('HF pretrained snapshot changed since experiment configuration')
        if spec.image_initialization == 'randomized':
            # Do NOT call reset_parameters on the loaded backbone: timm
            # ConvNeXt may retain non-module parameters such as layer scales.
            # Reconstruct the exact SAME ConvNeXt architecture with entirely
            # new random weights, including every Parameter.
            import timm
            image_encoder = model.vit_tower.image_encoder
            name = image_encoder._resolve_timm_name(
                image_encoder.hf_train_config.get('model_kind', '')
            )
            fresh = timm.create_model(name, pretrained=False, num_classes=0)
            if fresh.num_features != image_encoder.output_dim:
                raise RuntimeError('Randomized ConvNeXt feature width changed')
            image_encoder.backbone = fresh
            backbone = fresh
        if spec.freeze_image:
            for p in backbone.parameters():
                p.requires_grad_(False)
    if spec.freeze_image and spec.variant == 'patch_images':
        raise ValueError('Cannot freeze the random patch-image encoder')
    # No pretrained coordinate checkpoint in the new standalone setup.
    # Upstream freezes coord by default; undo this so random coordinates can learn.
    for p in model.coord_tower.parameters():
        p.requires_grad_(True)
    model.to(device)
    return model, cfg


def load_checkpoint(path: Path, *, spec: NewImageModelSpec, device: str | torch.device,
                    seed: int):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('model_spec') != asdict(spec):
        raise ValueError('Checkpoint model architecture/pretraining/source SHA differs')
    if tuple(payload.get('class_names', ())) != tuple(CLASS_NAMES):
        raise ValueError('Checkpoint five-class order differs')
    model, cfg = build_model(spec, seed=seed, device=device)
    model.load_state_dict(payload['model_state'], strict=True)
    return model, cfg, payload
