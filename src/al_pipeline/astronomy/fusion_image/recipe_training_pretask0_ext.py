"""Resident-RAM full-history AstroMiNN recipe comparison; no acquisition/test access.

This module deliberately leaves NewImageTrainer's AL adapter unchanged. Its
checkpoint payloads follow the same NewImageModelSpec / model_state contract.
"""
from __future__ import annotations

import copy
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from al_pipeline.astronomy.fusion_image.backends import object_scores
from al_pipeline.astronomy.fusion_image.fast_cache import FastHistoricalDataset
from al_pipeline.astronomy.fusion_image.resident_cache import attach_resident_cache
from al_pipeline.astronomy.fusion_image.contracts import (
    CLASS_NAMES, label_index, load_rows, object_labels, sha256, write_json_atomic,
)
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model
from al_pipeline.core.evaluation.metrics import compute_classification_metrics, compute_per_class_metrics
from al_pipeline.core.types import ClassificationPredictions


@dataclass(frozen=True)
class Recipe:
    name: str
    seed: int = 135
    max_epochs: int = 80
    patience: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    class_weight_exponent: float = 0.85
    lr_schedule: Literal['constant', 'warmup_cosine'] = 'constant'
    warmup_epochs: int = 3
    warmup_factor: float = 0.1
    cosine_min_factor: float = 0.05
    cosine_decay_epochs: int | None = None
    effective_count_basis: Literal['samples', 'objects'] = 'samples'
    image_augmentation: bool = False
    ema: bool = False
    ema_decay: float = 0.999
    class_weighting: Literal['inverse_frequency', 'effective_number'] = 'inverse_frequency'
    effective_beta: float = 0.999

    def __post_init__(self):
        if not self.name or self.seed < 0 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError('Invalid recipe identity or epoch count')
        if self.batch_size < 1 or self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.class_weight_exponent <= 2:
            raise ValueError('Invalid optimizer settings')
        if self.lr_schedule not in ('constant', 'warmup_cosine'):
            raise ValueError('Unsupported LR schedule')
        if not 0 < self.warmup_factor <= 1 or not 0 <= self.cosine_min_factor <= 1:
            raise ValueError('Invalid schedule factors')
        if self.lr_schedule == 'warmup_cosine' and not 1 < self.warmup_epochs < self.max_epochs:
            raise ValueError('Warmup must be shorter than training horizon')
        if self.cosine_decay_epochs is not None and not self.warmup_epochs < self.cosine_decay_epochs <= self.max_epochs:
            raise ValueError('Cosine decay horizon must be after warmup and at most max_epochs')
        if self.effective_count_basis not in ('samples', 'objects'):
            raise ValueError('Unsupported effective count basis')
        if not 0 < self.ema_decay < 1 or not 0 < self.effective_beta < 1:
            raise ValueError('EMA decay and effective beta must be within (0,1)')


def lr_multiplier(recipe: Recipe, epoch: int) -> float:
    """Epoch LR; first epoch is warmup start, warmup ends at base LR."""
    if not 1 <= epoch <= recipe.max_epochs:
        raise ValueError('Epoch outside schedule')
    if recipe.lr_schedule == 'constant':
        return 1.0
    if epoch <= recipe.warmup_epochs:
        return recipe.warmup_factor + (1.0-recipe.warmup_factor) * (epoch-1)/(recipe.warmup_epochs-1)
    horizon = recipe.cosine_decay_epochs or recipe.max_epochs
    progress = min(1.0, (epoch-recipe.warmup_epochs)/(horizon-recipe.warmup_epochs))
    return recipe.cosine_min_factor + (1.0-recipe.cosine_min_factor) * 0.5*(1+math.cos(math.pi*progress))


def augment_images(images: torch.Tensor) -> torch.Tensor:
    """One identical D4 transform for all three channels of each cutout.

    These operations preserve image content and shape, happen *after* cached
    preprocessing, and are never applied to validation or cache files.
    """
    if images.ndim != 4 or images.shape[1:] != (3,63,63):
        raise ValueError('Expected (batch,3,63,63) cutouts')
    # Batch-level random draws only affect training; each sample has own transform.
    # Group by transform to avoid a costly stack of copies on the GPU.
    transforms = torch.randint(0, 8, (images.shape[0],), device=images.device)
    out = images.clone()
    for k in range(8):
        positions = torch.nonzero(transforms == k, as_tuple=True)[0]
        if not positions.numel():
            continue
        block = images.index_select(0, positions)
        block = torch.rot90(block, k % 4, dims=(-2,-1))
        if k >= 4:
            block = torch.flip(block, dims=(-1,))
        out.index_copy_(0, positions, block)
    return out


def weight_vector(rows: list[dict], *, recipe: Recipe, id_col: str, label_col: str):
    if recipe.class_weighting == 'inverse_frequency':
        counts = Counter(label_index(row[label_col]) for row in rows)
        values = np.asarray([counts.get(i,0) for i in range(len(CLASS_NAMES))], dtype=float)
        if np.any(values <= 0):
            raise ValueError('Missing class from full training data')
        w = values ** (-recipe.class_weight_exponent)
    else:
        if recipe.effective_count_basis == 'objects':
            objs = object_labels(rows, id_col=id_col, label_col=label_col)
            counts = Counter(objs.values())
        else:
            counts = Counter(label_index(row[label_col]) for row in rows)
        values = np.asarray([counts.get(i,0) for i in range(len(CLASS_NAMES))], dtype=float)
        if np.any(values <= 0):
            raise ValueError('Missing object class from full training data')
        w = (1-recipe.effective_beta) / (-np.expm1(values * np.log(recipe.effective_beta)))
    w = w / w.mean()
    return w


@torch.inference_mode()
def evaluate(model: torch.nn.Module, dl: DataLoader, weights: torch.Tensor, device):
    model.eval()
    ids, alerts, ys, scores = [], [], [], []
    loss_sum, loss_batches = 0.0, 0
    for batch in dl:
        logits = model(batch['metadata'].to(device), batch['image'].to(device))['logits']
        target = batch['target_index'].to(device)
        if logits.shape != (len(target), len(CLASS_NAMES)) or not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite logits or incorrect class dimension')
        weighted = F.cross_entropy(logits.float(), target, weight=weights, reduction='none')
        loss_sum += float(weighted.mean())
        loss_batches += 1
        ids.extend(map(str, batch['obj_id']))
        alerts.extend(map(int, batch['alert'].tolist()))
        ys.extend(map(int, target.cpu().tolist()))
        scores.append(logits.detach().cpu().float().numpy())
    scores_np = np.concatenate(scores)
    object_y, object_pred = object_scores(ids, alerts, scores_np, ys)
    sample_pred = ClassificationPredictions(values=scores_np, output_type='logits')
    return {
        'loss': loss_sum/loss_batches,
        'object_metrics': compute_classification_metrics(object_y, object_pred),
        'sample_metrics': compute_classification_metrics(np.asarray(ys), sample_pred),
        'object_per_class': compute_per_class_metrics(object_y, object_pred),
        'object_count': len(object_y), 'sample_count': len(ys),
    }


class ParameterEMA:
    """EMA trainable parameters; copy current BN/buffer state before evaluation."""
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay = decay
        self.steps = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.steps += 1
        shadow = dict(self.model.named_parameters())
        for name, p in model.named_parameters():
            if p.is_floating_point():
                shadow[name].lerp_(p.detach(), 1-self.decay)
            else:
                shadow[name].copy_(p.detach())
        for ema_b, src_b in zip(self.model.buffers(), model.buffers(), strict=True):
            ema_b.copy_(src_b.detach())


def train_recipe(*, recipe: Recipe, spec: NewImageModelSpec, runtime_config: dict,
                 cache_root: Path, train_manifest: Path, validation_manifest: Path,
                 id_col: str, label_col: str, output_dir: Path, identity: dict):
    if not torch.cuda.is_available():
        raise RuntimeError('Scientific fits require CUDA')
    device = torch.device('cuda')
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Existing fit data: {root}; preserve partial results for audit')
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root/'fit_identity.json', identity)
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    all_started = time.perf_counter()
    model_start = time.perf_counter()
    model, model_cfg = build_model(spec, seed=recipe.seed, device=device)
    model_init_seconds = time.perf_counter() - model_start
    print(f'[fast pre-task0] model_init_seconds={model_init_seconds:.3f}',flush=True)
    # Verify exact preprocessing identity comes from the newly built model.
    from .cache_helpers import preprocessing_identity
    if preprocessing_identity(model_cfg) != preprocessing_identity(runtime_config):
        raise ValueError('Cached and model preprocessing do not agree')
    manifest_start = time.perf_counter()
    train_rows = load_rows(train_manifest)
    val_rows = load_rows(validation_manifest)
    manifest_load_seconds = time.perf_counter() - manifest_start
    cache_start = time.perf_counter()
    train_ds = FastHistoricalDataset(
        rows=train_rows, cache_root=cache_root, split='train',
        object_column=id_col, label_column=label_col)
    val_ds = FastHistoricalDataset(
        rows=val_rows, cache_root=cache_root, split='validation',
        object_column=id_col, label_column=label_col)
    # Sequential one-time HDD read into shared RAM. The prior mmap-only dataset
    # incurred 7-46 seconds per randomly selected batch on Lustre.
    resident_info = attach_resident_cache(train_ds, val_ds, cache_root=cache_root)
    train_dl = DataLoader(train_ds, batch_size=recipe.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(recipe.seed), num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=recipe.batch_size, shuffle=False, num_workers=0)
    dataset_init_seconds = time.perf_counter() - cache_start
    print(f'[fast pre-task0] manifest_load_seconds={manifest_load_seconds:.3f} dataset_init_seconds={dataset_init_seconds:.3f}',flush=True)
    w = weight_vector(train_rows, recipe=recipe, id_col=id_col, label_col=label_col)
    weights = torch.as_tensor(w, device=device, dtype=torch.float32)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
    ema = ParameterEMA(model, recipe.ema_decay) if recipe.ema else None
    checkpoint_source = 'ema' if recipe.ema else 'raw'
    train_started = time.perf_counter()
    setup_timing = {
        'model_init_seconds':model_init_seconds,
        'manifest_load_seconds':manifest_load_seconds,
        'dataset_init_seconds':dataset_init_seconds,
        'resident_load_seconds':resident_info['total_seconds'],
        'resident_cache_bytes':resident_info['resident_bytes'],
        'data_access_mode':resident_info['strategy'],
        'other_setup_seconds':train_started-all_started-model_init_seconds-manifest_load_seconds-dataset_init_seconds,
        'total_setup_seconds':train_started-all_started,
    }
    write_json_atomic(root/'setup_timing.json', setup_timing)
    print('[fast pre-task0] setup='+str(setup_timing),flush=True)
    best_loss, best_epoch, stale = math.inf, 0, 0
    best_f1, best_f1_epoch = -math.inf, 0
    history = []
    for epoch in range(1, recipe.max_epochs+1):
        started = time.perf_counter()
        rate = recipe.learning_rate * lr_multiplier(recipe, epoch)
        for group in opt.param_groups:
            group['lr'] = rate
        model.train()
        loss_total, batches = 0.0, 0
        train_t0 = time.perf_counter()
        for batch in train_dl:
            x = batch['metadata'].to(device)
            im = batch['image'].to(device)
            if recipe.image_augmentation:
                im = augment_images(im)
            y = batch['target_index'].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x, im)['logits']
                loss = F.cross_entropy(logits.float(), y, weight=weights,
                                       reduction='none').mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite train loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)
            loss_total += float(loss.detach())
            batches += 1
        training_seconds = time.perf_counter() - train_t0
        eval_t0 = time.perf_counter()
        result = evaluate(ema.model if ema else model, val_dl, weights, device)
        validation_seconds = time.perf_counter()-eval_t0
        val_loss = float(result['loss'])
        val_f1 = float(result['object_metrics']['f1_macro'])
        record = {
            'epoch': epoch, 'learning_rate':rate, 'train_loss':loss_total/batches,
            'validation_loss':val_loss, 'validation_object_macro_f1':val_f1,
            'training_seconds':training_seconds,
            'validation_seconds':validation_seconds,
            'epoch_seconds':time.perf_counter()-started,
            'evaluated_weights':checkpoint_source,
        }
        history.append(record)
        write_json_atomic(root/'training_history.json', history)
        print(f'[pre-task0] {recipe.name} epoch {epoch}/{recipe.max_epochs} '
              f'lr={rate:.3e} train={record["train_loss"]:.5f} '
              f'val={val_loss:.5f} macroF1={val_f1:.5f} '
              f'elapsed={record["epoch_seconds"]:.1f}s', flush=True)
        if val_f1 > best_f1:
            best_f1, best_f1_epoch = val_f1, epoch
            f1_model = ema.model if ema else model
            tmp_f1 = root/'best_object_f1.pth.tmp'
            torch.save({
                'model_state':f1_model.state_dict(),
                'model_spec':asdict(spec), 'class_names':list(CLASS_NAMES),
                'seed':recipe.seed, 'epoch':epoch, 'validation_loss':val_loss,
                'validation_macro_f1':val_f1, 'checkpoint_metric':'object_macro_f1',
                'checkpoint_weights':checkpoint_source,
            }, tmp_f1)
            tmp_f1.replace(root/'best_object_f1.pth')
        if val_loss < best_loss:
            best_epoch, best_loss, stale = epoch, val_loss, 0
            model_to_save = ema.model if ema else model
            tmp = root/'best_vit_tower.pth.tmp'
            torch.save({
                'model_state':model_to_save.state_dict(),
                'model_spec':asdict(spec), 'class_names':list(CLASS_NAMES),
                'seed':recipe.seed, 'epoch':epoch, 'validation_loss':val_loss,
                'validation_macro_f1':val_f1, 'checkpoint_metric':'loss',
                'checkpoint_weights':checkpoint_source,
            }, tmp)
            tmp.replace(root/'best_vit_tower.pth')
        else:
            stale += 1
        if stale >= recipe.patience:
            break
    train_elapsed = time.perf_counter() - train_started
    # Final evaluation is from the selected saved checkpoint, NEVER last epoch.
    from .model import load_checkpoint
    final_eval_start = time.perf_counter()
    selected, _, _ = load_checkpoint(root/'best_vit_tower.pth', spec=spec,
                                      seed=recipe.seed, device=device)
    result = evaluate(selected, val_dl, weights, device)
    final_evaluation_seconds = time.perf_counter() - final_eval_start
    summary = {
        'setup_timing':setup_timing, 'final_evaluation_seconds':final_evaluation_seconds,
        'identity':identity, 'run_name':recipe.name, 'status':'completed',
        'best_epoch':best_epoch, 'best_validation_loss':best_loss,
        'epochs_completed':len(history), 'fit_seconds':train_elapsed,
        'end_to_end_seconds':time.perf_counter()-all_started,
        'checkpoint':str(root/'best_vit_tower.pth'),
        'checkpoint_weights':checkpoint_source,
        'training_rows':len(train_rows), 'validation_rows':len(val_rows),
        'class_weights':dict(zip(CLASS_NAMES, map(float,w))),
        'class_weight_count_basis': recipe.effective_count_basis if recipe.class_weighting == 'effective_number' else 'samples',
        'validation_object_metrics':result['object_metrics'],
        'validation_sample_metrics':result['sample_metrics'],
        'validation_objects':result['object_count'],
        'validation_per_class':str(root/'validation_per_class_metrics.json'),
        'best_macro_f1_during_fit':best_f1,
        'best_macro_f1_epoch':best_f1_epoch,
        'best_f1_checkpoint':str(root/'best_object_f1.pth'),
    }
    write_json_atomic(root/'validation_per_class_metrics.json', result['object_per_class'])
    write_json_atomic(root/'training_summary.json', summary)
    return summary
