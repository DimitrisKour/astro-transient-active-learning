"""MC-dropout BALD inference over latest-alert AstroMiNN pool objects.

A single fixed mask per active ViT dropout module defines one coherent
approximate parameter draw across every object in the pool. Masks are redrawn
between posterior samples; BatchNorm and other modules stay in eval mode.
This is an approximation to the Bayesian posterior, NOT exact Bayesian inference.
"""
from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import types
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (
    AstroMiNNViTPoolDataset, load_astrominn_vit_model,
    DEFAULT_LEGACY_CODE_DIR,
)
from al_pipeline.core.scoring.bald import BALDInput, BALDScorer

SUPPORTED = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


def active_vit_dropout(model):
    if not hasattr(model, 'vit_tower'):
        raise ValueError('AstroMiNN model has no vit_tower; cannot activate MC dropout')
    layers = [(name, module) for name, module in model.vit_tower.named_modules()
              if isinstance(module, SUPPORTED) and 0 < float(module.p) < 1]
    if not layers:
        raise RuntimeError('No active ViT nn.Dropout modules: MC dropout BALD is not valid on this checkpoint')
    return layers


@contextmanager
def fixed_vit_dropout_sample(model: nn.Module, seed: int):
    """Override dropout forwards without setting BatchNorm/train mode.

    Each module's mask is fixed across the pool within a sampled model. This
    treats dropout masks as the variational model draw and uses shared masks
    for every object's conditional prediction. No global RNG is mutated.
    """
    layers = active_vit_dropout(model)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    originals = []
    sampled_masks = {}
    try:
        for name, layer in layers:
            originals.append((layer, layer.forward))

            def forward(self, x, _name=name):
                if x.ndim < 2:
                    raise ValueError('Fixed MC dropout expects batch-first activations')
                p = float(self.p)
                if isinstance(self, (nn.Dropout2d, nn.Dropout3d, nn.Dropout1d)):
                    # nn.DropoutNd: one mask per channel, independent of spatial dims.
                    mask_shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
                else:
                    mask_shape = (1, *x.shape[1:])
                old = sampled_masks.get(_name)
                if old is None:
                    mask = (torch.rand(mask_shape, generator=generator) >= p).to(dtype=torch.float32)/(1-p)
                    sampled_masks[_name] = mask
                else:
                    mask = old
                    if tuple(mask.shape) != tuple(mask_shape):
                        raise RuntimeError(f'Dropout tensor shape changed across pool batches: {_name}')
                return x * mask.to(device=x.device, dtype=x.dtype)

            layer.forward = types.MethodType(forward, layer)
        yield tuple(name for name, _ in layers)
    finally:
        for layer, previous in reversed(originals):
            layer.forward = previous


def latest_pool_paths(pool_path: Path, object_ids: tuple[str, ...]):
    """Use exactly the same latest-alert rule as Task4, without extra alerts.

    Ties at the latest alert are not silently disambiguated; the pilot fails
    closed so the user can establish equivalence to the historical aggregator.
    """
    manifest = CandidatePoolManifest.from_csv(Path(pool_path)/'candidate_pool.csv')
    chosen = {}
    for record in manifest.records:
        key = str(record.object_id)
        alert = int(record.alert)
        if key not in chosen or alert > chosen[key][0]:
            chosen[key] = (alert, Path(record.filepath))
        elif alert == chosen[key][0]:
            raise ValueError(f'Ambiguous latest alert for {key!r}: separate samples have same alert')
    if len(set(object_ids)) != len(object_ids):
        raise ValueError('Duplicate object IDs')
    if set(object_ids) - set(chosen):
        raise ValueError('Requested object IDs missing from pool')
    return tuple(chosen[k][1] for k in object_ids), tuple(chosen[k][0] for k in object_ids)


@dataclass(frozen=True)
class MCResult:
    object_ids: tuple[str, ...]
    probabilities: np.ndarray
    dropout_layers: tuple[str, ...]
    scores: np.ndarray


def sample_pool(*, pool_path: Path, object_ids: tuple[str, ...], checkpoint_path: Path,
                artifact_dir: Path, device='cuda', samples: int = 20, batch_size: int = 64,
                seed: int = 135, cycle: int = 1, num_workers: int = 0,
                legacy_code_dir=DEFAULT_LEGACY_CODE_DIR) -> MCResult:
    if samples < 2 or samples > 128: raise ValueError('Invalid number of MC samples')
    paths, alerts = latest_pool_paths(Path(pool_path), tuple(object_ids))
    data = AstroMiNNViTPoolDataset(pool_path, artifact_dir=artifact_dir, object_ids=object_ids)
    # Only latest alert contributes to Task4's object-level acquisition scores.
    data.sample_paths = paths
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = load_astrominn_vit_model(
        legacy_code_dir=legacy_code_dir, artifact_dir=artifact_dir,
        vit_checkpoint=checkpoint_path,
    )
    device = torch.device(device)
    model.to(device)
    model.eval()
    layers = active_vit_dropout(model)
    if any(m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)):
        raise RuntimeError('BatchNorm must remain in eval mode')
    draws = []
    with torch.no_grad():
        for t in range(samples):
            draw_seed = int(np.random.SeedSequence([seed, cycle, t, 120120]).generate_state(1)[0])
            batches = []
            with fixed_vit_dropout_sample(model, draw_seed):
                row = 0
                for batch in loader:
                    names = tuple(str(v) for v in batch['obj_id'])
                    expected = object_ids[row:row + len(names)]
                    if names != expected: raise ValueError('Stochastic inference object order changed')
                    row += len(names)
                    outputs = model(batch['metadata'].to(device), image=batch['image'].to(device))
                    if not isinstance(outputs, dict) or 'logits' not in outputs:
                        raise ValueError('AstroMiNN must return logits')
                    logits = outputs['logits']
                    if logits.ndim != 2 or logits.shape[0] != len(names) or not torch.isfinite(logits).all():
                        raise ValueError('Invalid logits during MC dropout sampling')
                    batches.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                if row != len(object_ids): raise ValueError('Incomplete MC dropout pass')
            draws.append(np.concatenate(batches).astype(np.float32))
    probability_tensor = np.stack(draws, axis=1) # (objects, posterior_draws, classes)
    scores = BALDScorer().score(BALDInput(probability_tensor))
    if float(scores.max()) <= 1e-9 or np.max(np.var(probability_tensor.astype(np.float64),axis=1)) <= 1e-12:
        raise RuntimeError('MC dropout generated effectively identical predictions: cannot claim meaningful BALD')
    if any(m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)):
        raise RuntimeError('BatchNorm unexpectedly entered training mode')
    return MCResult(tuple(object_ids), probability_tensor, tuple(name for name, _ in layers), scores)
