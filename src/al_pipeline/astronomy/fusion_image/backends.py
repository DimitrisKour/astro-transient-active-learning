"""New image+metadata model implementing the existing AL backend contracts.

A first independent training check and the later AL loop use the same code.
This intentionally uses five-class weighted cross entropy as in Experiment B,
not the unrelated TEMPO+AstroMiNN hierarchical evidential fusion loss.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from al_pipeline.astronomy.astrominn.inference.aggregation import aggregate_object_predictions
from al_pipeline.astronomy.astrominn.types import ClassificationInferenceResult
from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
from al_pipeline.core.evaluation.base import ModelEvaluationResult, SplitEvaluation
from al_pipeline.core.evaluation.metrics import compute_classification_metrics, compute_per_class_metrics
from al_pipeline.core.training.base import TrainingRequest, TrainingResult
from al_pipeline.core.types import ClassificationPredictions

from .contracts import CLASS_NAMES, label_index, load_rows, sha256, write_json_atomic
from .data import HistoricalNPZDataset
from .model import NewImageModelSpec, build_model, load_checkpoint


@dataclass(frozen=True)
class ImageTrainingPolicy:
    max_epochs: int = 50
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    class_weight_exponent: float = 0.85
    weighting: Literal['per_sample', 'equal_object'] = 'per_sample'
    checkpoint_metric: Literal['loss', 'object_macro_f1'] = 'loss'
    max_training_batches: int | None = None  # pilot only, never scientific training

    def __post_init__(self):
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError('Training epochs/patience/batch_size must be >=1')
        if not 0 <= self.class_weight_exponent <= 2:
            raise ValueError('Invalid class weight exponent')
        if self.checkpoint_metric not in {'loss', 'object_macro_f1'}:
            raise ValueError('Invalid checkpoint rule')
        if self.max_training_batches is not None and self.max_training_batches < 1:
            raise ValueError('max_training_batches must be positive')


def loader(rows, *, index, runtime_config, id_col, label_col, batch_size,
           shuffle=False, seed: int | None = None):
    cache_root = os.environ.get('FUSION_IMAGE_CACHE_ROOT')
    if cache_root:
        cache_dir = Path(cache_root)
        cache_manifest = json.loads((cache_dir / 'cache_manifest.json').read_text())
        if cache_manifest.get('cache_version') == 2:
            from .cache_v2 import ShardedHistoricalNPZDataset
            dataset_type = ShardedHistoricalNPZDataset
        else:
            from .cache import CachedHistoricalNPZDataset
            dataset_type = CachedHistoricalNPZDataset
        ds = dataset_type(
            rows=rows, cache_root=cache_dir, runtime_config=runtime_config,
            object_column=id_col, label_column=label_col,
            preload=os.environ.get('FUSION_IMAGE_CACHE_PRELOAD', '1') == '1')
    else:
        ds = HistoricalNPZDataset(rows=rows, index=index, runtime_config=runtime_config,
                                  object_column=id_col, label_column=label_col)
    # Isolate training-row ordering from architecture-specific RNG consumption:
    # the pretrained/randomized ConvNeXt pair must see the same minibatches.
    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


def _predict_batches(model, dl, device: torch.device):
    model.eval()
    ids: list[str] = []
    alerts: list[int] = []
    ys: list[int] = []
    scores = []
    with torch.inference_mode():
        for batch in dl:
            output = model(batch['metadata'].to(device), batch['image'].to(device))
            logits = output['logits']
            if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
                raise RuntimeError('New model must output 5 logits in historical class order')
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite new model logits')
            scores.append(logits.cpu().float().numpy())
            ids.extend(batch['obj_id'])
            alerts.extend(int(v) for v in batch['alert'].tolist())
            if 'target_index' in batch:
                ys.extend(int(v) for v in batch['target_index'].tolist())
    if not scores:
        raise ValueError('Empty prediction loader')
    return ids, alerts, np.concatenate(scores), ys


def object_scores(ids, alerts, logits, ys):
    pred = ClassificationPredictions(values=logits, output_type='logits')
    agg = aggregate_object_predictions(object_ids=tuple(ids), alerts=tuple(alerts), predictions=pred)
    label_by_obj = {}
    for oid, y in zip(ids, ys):
        if oid in label_by_obj and label_by_obj[oid] != y:
            raise ValueError(f'Conflicting label within object {oid}')
        label_by_obj[oid] = y
    objects = np.asarray([label_by_obj[o] for o in agg.object_ids])
    return objects, agg.predictions


def metrics_for_loader(model, dl, device):
    ids, alerts, logits, ys = _predict_batches(model, dl, device)
    if not ys:
        raise ValueError('Evaluation requires labels')
    obj_y, obj_pred = object_scores(ids, alerts, logits, ys)
    sample_pred = ClassificationPredictions(values=logits, output_type='logits')
    return (compute_classification_metrics(np.asarray(ys), sample_pred),
            compute_classification_metrics(obj_y, obj_pred),
            compute_per_class_metrics(obj_y, obj_pred), len(ids), len(obj_y))


def weighted_loss_from_logits(logits, ys, weights, batch_size: int) -> float:
    """Same mean of batch-mean weighted CE as the original trainer.

    Input logits are the unrounded float32 array from the one validation pass.
    This uses CPU to avoid storing GPU logits for a full historical split.
    """
    if batch_size < 1:
        raise ValueError('Invalid validation batch size')
    logits_cpu = torch.as_tensor(logits, dtype=torch.float32, device='cpu')
    labels_cpu = torch.as_tensor(ys, dtype=torch.long, device='cpu')
    weights_cpu = weights.detach().to(device='cpu', dtype=torch.float32)
    loss_sum, batches = 0.0, 0
    for start in range(0, len(ys), batch_size):
        stop = min(start + batch_size, len(ys))
        ce = torch.nn.functional.cross_entropy(logits_cpu[start:stop],
            labels_cpu[start:stop], weight=weights_cpu, reduction='none').mean()
        loss_sum += float(ce)
        batches += 1
    return loss_sum / max(batches, 1)


def metrics_and_loss_for_loader(model, dl, device, weights):
    ids, alerts, logits, ys = _predict_batches(model, dl, device)
    if not ys:
        raise ValueError('Validation requires labels')
    obj_y, obj_pred = object_scores(ids, alerts, logits, ys)
    sample_pred = ClassificationPredictions(values=logits, output_type='logits')
    metrics = (compute_classification_metrics(np.asarray(ys), sample_pred),
               compute_classification_metrics(obj_y, obj_pred),
               compute_per_class_metrics(obj_y, obj_pred), len(ids), len(obj_y))
    return (*metrics, weighted_loss_from_logits(logits, ys, weights, dl.batch_size))


class NewImageTrainer:
    def __init__(self, *, spec: NewImageModelSpec, policy: ImageTrainingPolicy,
                 index: Mapping[str, Path], runtime_config: dict, id_col: str,
                 label_col: str, device: str | torch.device = 'cpu',
                 initialization: Literal['random', 'checkpoint'] = 'checkpoint'):
        self.spec = spec
        self.policy = policy
        self.index = index
        self.config = runtime_config
        self.id_col = id_col
        self.label_col = label_col
        self.device = torch.device(device)
        self.initialization = initialization

    def completed_result(self, request: TrainingRequest) -> TrainingResult | None:
        p = request.output_dir / 'training_summary.json'
        if not p.exists():
            return None
        obj = json.loads(p.read_text())
        initial = str(request.initial_checkpoint) if request.initial_checkpoint else None
        expected = {
            'run_id': request.run_id, 'seed': request.seed, 'cycle': request.cycle,
            'strategy': request.strategy, 'initial_checkpoint': initial,
            'manifest_sha256': sha256(request.train_manifest),
            'validation_sha256': sha256(request.validation_manifest),
            'policy': asdict(self.policy), 'model_spec': asdict(self.spec),
        }
        if any(obj.get(k) != v for k, v in expected.items()):
            raise ValueError('Completed training identity mismatch; refusing to reuse')
        for name in ('best_vit_tower.pth', 'training_history.json'):
            if not (request.output_dir / name).is_file():
                raise FileNotFoundError(request.output_dir / name)
        return TrainingResult(checkpoint_path=request.output_dir/'best_vit_tower.pth',
                              history_path=request.output_dir/'training_history.json',
                              summary_path=p, best_epoch=int(obj['best_epoch']),
                              epochs_completed=int(obj['epochs_completed']),
                              best_validation_loss=float(obj['best_validation_loss']),
                              elapsed_seconds=float(obj['elapsed_seconds']))

    def train(self, request: TrainingRequest) -> TrainingResult:
        if self.completed_result(request) is not None:
            raise FileExistsError('Training already completed for this run')
        if request.output_dir.exists() and list(request.output_dir.iterdir()):
            raise FileExistsError(f'Incomplete output exists; inspect before rerun: {request.output_dir}')
        request.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(request.seed)
        if self.initialization == 'random':
            if request.initial_checkpoint is not None:
                raise ValueError('Random restart received checkpoint unexpectedly')
            model, cfg = build_model(self.spec, seed=request.seed, device=self.device)
        else:
            if request.initial_checkpoint is None:
                raise ValueError('Checkpoint initialization requires checkpoint')
            model, cfg, _ = load_checkpoint(request.initial_checkpoint,
                                             spec=self.spec, device=self.device, seed=request.seed)
        train_rows = load_rows(request.train_manifest)
        val_rows = load_rows(request.validation_manifest)
        train = loader(train_rows, index=self.index, runtime_config=cfg, id_col=self.id_col,
                       label_col=self.label_col, batch_size=self.policy.batch_size,
                       shuffle=True, seed=request.seed)
        val = loader(val_rows, index=self.index, runtime_config=cfg, id_col=self.id_col,
                     label_col=self.label_col, batch_size=self.policy.batch_size)
        counts = Counter(label_index(r[self.label_col]) for r in train_rows)
        # Use only revealed L_t rows: never consult hidden-pool class counts.
        cls = np.asarray([counts.get(i, 0) for i in range(5)], dtype=float)
        if np.any(cls == 0):
            raise ValueError('All five classes must be present in the current labeled set')
        w = (cls ** -self.policy.class_weight_exponent)
        w = w / w.mean()
        weights = torch.as_tensor(w, device=self.device, dtype=torch.float32)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=self.policy.learning_rate, weight_decay=self.policy.weight_decay)
        best_loss, best_f1 = float('inf'), float('-inf')
        best_epoch, stale, loss_stale = 0, 0, 0
        history = []
        object_counts = Counter(str(r[self.id_col]) for r in train_rows)
        peak = None
        start = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        for epoch in range(1, self.policy.max_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            sum_loss, batches = 0.0, 0
            for batch in train:
                optimizer.zero_grad(set_to_none=True)
                x = batch['metadata'].to(self.device)
                im = batch['image'].to(self.device)
                target = batch['target_index'].to(self.device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=self.device.type == 'cuda'):
                    output = model(x, im)['logits']
                    weighted_loss = torch.nn.functional.cross_entropy(output, target,
                                                                   weight=weights, reduction='none')
                    if self.policy.weighting == 'equal_object':
                        s = torch.tensor([1.0 / object_counts[o] for o in batch['obj_id']],
                                         dtype=torch.float32, device=self.device)
                        loss = (weighted_loss * s).sum() / s.sum()
                    else:
                        loss = weighted_loss.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss; aborting rather than corrupting model')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                sum_loss += float(loss.detach())
                batches += 1
                if self.policy.max_training_batches is not None and batches >= self.policy.max_training_batches:
                    break
            train_seconds = time.perf_counter() - epoch_start
            val_start = time.perf_counter()
            # Reuse the validation logits for BOTH the object-level probe and
            # weighted validation loss. Previously this iterated the entire
            # compressed-NPZ validation dataset twice per epoch; this preserves
            # the same per-batch-mean validation objective and evaluation rules.
            sm, om, pc, _, _, val_loss = metrics_and_loss_for_loader(
                model, val, self.device, weights)
            validation_seconds = time.perf_counter() - val_start
            val_f1 = float(om['f1_macro'])
            record = {'epoch': epoch, 'train_loss': sum_loss/max(batches, 1),
                      'validation_loss': val_loss, 'validation_object_macro_f1': val_f1,
                      'training_seconds': train_seconds,
                      'validation_seconds': validation_seconds,
                      'epoch_seconds': time.perf_counter() - epoch_start,
                      'elapsed_seconds': time.perf_counter() - start}
            history.append(record)
            write_json_atomic(request.output_dir/'training_history.json', history)
            print(f'[fusion image] {request.run_id} epoch {epoch}/{self.policy.max_epochs} '
                  f'train_loss={record["train_loss"]:.5f} val_loss={val_loss:.5f} '
                  f'val_object_macro_f1={val_f1:.5f} elapsed_s={time.perf_counter()-start:.1f}',
                  flush=True)
            improved_loss = val_loss < best_loss
            loss_stale = 0 if improved_loss else loss_stale + 1
            best_loss = min(best_loss, val_loss)
            if self.policy.checkpoint_metric == 'loss':
                improved = improved_loss
            else:
                improved = val_f1 > best_f1
            best_f1 = max(best_f1, val_f1)
            if improved:
                best_epoch, stale = epoch, 0
                payload = {'model_state': model.state_dict(), 'model_spec': asdict(self.spec),
                           'class_names': list(CLASS_NAMES), 'seed': request.seed,
                           'epoch': epoch, 'validation_loss': val_loss,
                           'validation_macro_f1': val_f1,
                           'checkpoint_metric': self.policy.checkpoint_metric}
                tmp_ckpt = request.output_dir/'best_model.pt.tmp'
                torch.save(payload, tmp_ckpt)
                tmp_ckpt.replace(request.output_dir/'best_vit_tower.pth')
            else:
                stale += 1
            if loss_stale >= self.policy.patience:
                break
        elapsed = time.perf_counter() - start
        if self.device.type == 'cuda':
            peak = int(torch.cuda.max_memory_allocated(self.device))
        summary = {
            'run_id': request.run_id, 'seed': request.seed, 'strategy': request.strategy,
            'cycle': request.cycle, 'initial_checkpoint':
                str(request.initial_checkpoint) if request.initial_checkpoint else None,
            'manifest_sha256': sha256(request.train_manifest),
            'validation_sha256': sha256(request.validation_manifest),
            'policy': asdict(self.policy), 'model_spec': asdict(self.spec),
            'best_epoch': best_epoch, 'epochs_completed': len(history),
            'best_validation_loss': best_loss, 'elapsed_seconds': elapsed,
            'peak_cuda_memory_bytes': peak,
            'class_weights': {k: float(v) for k, v in zip(CLASS_NAMES, w)},
            'training_rows': len(train_rows), 'validation_rows': len(val_rows),
            'stage_cache': os.environ.get('FUSION_IMAGE_CACHE_ROOT'),
        }
        write_json_atomic(request.output_dir/'training_summary.json', summary)
        return TrainingResult(checkpoint_path=request.output_dir/'best_vit_tower.pth',
                              history_path=request.output_dir/'training_history.json',
                              summary_path=request.output_dir/'training_summary.json',
                              best_epoch=best_epoch, epochs_completed=len(history),
                              best_validation_loss=best_loss, elapsed_seconds=elapsed,
                              peak_cuda_memory_bytes=peak)


class NewImagePoolPredictor:
    def __init__(self, *, spec: NewImageModelSpec, index: Mapping[str, Path],
                 pool_path: Path, device: str | torch.device = 'cpu', batch_size: int = 64):
        self.spec, self.index, self.pool_path = spec, index, Path(pool_path)
        self.device, self.batch_size = torch.device(device), batch_size

    def predict(self, *, object_ids, checkpoint_path: Path) -> ClassificationInferenceResult:
        model, cfg, _ = load_checkpoint(checkpoint_path, spec=self.spec,
                                         device=self.device, seed=135)
        manifest = CandidatePoolManifest.from_csv(self.pool_path/'candidate_pool.csv')
        requested = set(object_ids)
        if len(requested) != len(object_ids) or requested - set(manifest.object_ids):
            raise ValueError('Duplicate or unknown requested object IDs')
        # Score one latest available eligible stage, just as historical ViT does.
        latest: dict[str, int] = {}
        for r in manifest.records:
            if r.object_id in requested:
                latest[r.object_id] = max(r.alert, latest.get(r.object_id, -1))
        rows = [{'object_id': oid, 'alert': str(latest[oid])} for oid in object_ids]
        dl = loader(rows, index=self.index, runtime_config=cfg,
                    id_col='object_id', label_col=None, batch_size=self.batch_size)
        model.eval()
        ids: list[str] = []
        alerts: list[int] = []
        logits_batches = []
        embedding_batches = []
        with torch.inference_mode():
            for batch in dl:
                result = model(batch['metadata'].to(self.device),
                               batch['image'].to(self.device), return_features=True)
                logits, emb = result.logits, result.embedding
                if logits.ndim != 2 or logits.shape[1] != 5 or emb.ndim != 2 or \
                   emb.shape[0] != logits.shape[0]:
                    raise ValueError('New model logits/CLS embedding shape mismatch')
                if not torch.isfinite(logits).all() or not torch.isfinite(emb).all():
                    raise ValueError('Nonfinite logits or CLS embeddings')
                ids.extend(batch['obj_id'])
                alerts.extend(int(v) for v in batch['alert'].tolist())
                logits_batches.append(logits.cpu().float().numpy())
                embedding_batches.append(emb.cpu().float().numpy())
        if set(ids) != requested or len(ids) != len(requested):
            raise ValueError('Inference did not return exactly one row per object')
        return ClassificationInferenceResult(
            object_ids=tuple(ids), alerts=tuple(alerts),
            predictions=ClassificationPredictions(values=np.concatenate(logits_batches),
                                                  output_type='logits'),
            embeddings=np.concatenate(embedding_batches))



class NewImageEvaluator:
    def __init__(self, *, spec: NewImageModelSpec, index: Mapping[str, Path],
                 manifests: Mapping[str, Path], id_col: str, label_col: str,
                 device: str | torch.device = 'cpu', batch_size: int = 64):
        self.spec, self.index = spec, index
        self.manifests = manifests
        self.id_col, self.label_col = id_col, label_col
        self.device = torch.device(device)
        self.batch_size = batch_size

    def completed_result(self, *, checkpoint_path: Path, output_dir: Path) -> ModelEvaluationResult | None:
        p = output_dir/'evaluation_summary.json'
        if not p.is_file():
            return None
        data = json.loads(p.read_text())
        if data['checkpoint_sha256'] != sha256(checkpoint_path):
            raise ValueError('Evaluation checkpoint provenance mismatch')
        splits = tuple(SplitEvaluation(**entry) for entry in data['splits'])
        return ModelEvaluationResult(checkpoint_path=checkpoint_path, summary_path=p, splits=splits)

    def evaluate(self, *, checkpoint_path: Path, output_dir: Path) -> ModelEvaluationResult:
        existing = self.completed_result(checkpoint_path=checkpoint_path, output_dir=output_dir)
        if existing is not None:
            return existing
        model, cfg, _ = load_checkpoint(checkpoint_path, spec=self.spec,
                                        device=self.device, seed=135)
        results = []
        for name, path in self.manifests.items():
            dl = loader(load_rows(path), index=self.index, runtime_config=cfg,
                        id_col=self.id_col, label_col=self.label_col,
                        batch_size=self.batch_size)
            sample, obj, per_class, ns, no = metrics_for_loader(model, dl, self.device)
            results.append(SplitEvaluation(split=name, sample_count=ns,
                         object_count=no, sample_metrics=sample, object_metrics=obj))
            write_json_atomic(output_dir/f'{name}_per_class_metrics.json',
                {CLASS_NAMES[int(i)]: vals for i, vals in per_class.items()})
        payload = {'checkpoint_sha256': sha256(checkpoint_path),
                   'class_names': list(CLASS_NAMES),
                   'splits': [asdict(r) for r in results]}
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_dir/'evaluation_summary.json', payload)
        return ModelEvaluationResult(checkpoint_path=checkpoint_path,
                summary_path=output_dir/'evaluation_summary.json', splits=tuple(results))
