#!/usr/bin/env python3
"""Isolated full L0 *validation-only* fit on verified BTSbot cache v3.

No acquisition, no test-set evaluation, no baseline/matrix artifact overwrite.
Train with the EXISTING frozen L0 matrix policy (loss checkpoint and stopping).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.backends import (
    ImageTrainingPolicy, NewImageEvaluator, NewImageTrainer,
)
from al_pipeline.astronomy.fusion_image.contracts import load_index, sha256, write_json_atomic
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model
from al_pipeline.core.training.base import TrainingRequest

PERSISTENT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
PLAN = PERSISTENT / 'new_image_l0_matrix_v1/plan.json'
CACHE = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_l0_cache_v3')
OUTPUT = PERSISTENT / 'new_image_l0_cached_full_pretrained_seed135_v1'


def check_inputs(plan_path: Path, cache_root: Path) -> tuple[dict, object, dict]:
    """Fail closed on source, L0/validation, HF, and cache provenance."""
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    cache = json.loads((cache_root / 'cache_manifest.json').read_text(encoding='utf-8'))
    if cache.get('cache_version') != 2:
        raise ValueError('Expected finalized sharded cache version 2')
    source_sha = subprocess.check_output(
        ['git', '-C', plan['upstream_repo'], 'rev-parse', 'HEAD'], text=True
    ).strip()
    if source_sha != plan['model_sha']:
        raise ValueError('Upstream model source differs from audited L0 plan')
    if sha256(plan['new_index']) != cache['new_index_sha256']:
        raise ValueError('NPZ index differs from completed cache')
    if sha256(Path(plan['hf_snapshot']) / 'pytorch_model.bin') != plan['hf_weights_sha256']:
        raise ValueError('BTSbot pretrained weights changed')
    ds = DatasetRegistry.from_toml(plan['registry']).resolve_profile('astrominn_ztf').dataset
    manifests = {'train': Path(plan['l0_manifest']), 'validation': ds.validation_manifest}
    for split, path in manifests.items():
        if path is None or not path.is_file():
            raise FileNotFoundError(f'Missing {split} manifest: {path}')
        actual = sha256(path)
        if actual != cache['manifest_sha256'][split]:
            raise ValueError(f'Cached {split} manifest hash differs')
        reference = plan['file_hashes'][
            'l0_manifest' if split == 'train' else 'validation_manifest'
        ]
        if actual != reference:
            raise ValueError(f'{split} manifest changed since original L0 plan')
    cache_plan = cache_root / 'plan.json'
    if sha256(cache_plan) != cache['plan_sha256']:
        raise ValueError('Cached preprocessing plan changed after finalization')
    required = {'image_encoder_type': 'btsbot_convnext',
                'cutout_size': 63, 'image_norm': 'none'}
    for name, value in required.items():
        if cache['preprocessing_identity'].get(name) != value:
            raise ValueError(f'Unexpected cached image preprocessing: {name}')
    if cache['manifest_rows']['train'] != 2890 or cache['manifest_rows']['validation'] != 14148:
        raise ValueError('Historical L0/validation row count differs from successful benchmark')
    policy = ImageTrainingPolicy(**plan['training_policy'])
    if (policy.max_training_batches is not None or policy.checkpoint_metric != 'loss'
            or policy.max_epochs < 2):
        raise ValueError('Expected complete loss-selected L0 training policy')
    return plan, ds, cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'run'))
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    plan, ds, cache = check_inputs(args.plan, args.cache)
    policy = ImageTrainingPolicy(**plan['training_policy'])
    identity = {
        'plan': str(args.plan.resolve()),
        'plan_sha256': sha256(args.plan),
        'cache': str(args.cache.resolve()),
        'cache_manifest_sha256': sha256(args.cache / 'cache_manifest.json'),
        'source_revision': plan['model_sha'],
        'train_manifest_sha256': cache['manifest_sha256']['train'],
        'validation_manifest_sha256': cache['manifest_sha256']['validation'],
        'policy': plan['training_policy'],
        'initialization': 'pretrained', 'seed': 135,
        'test_access': False,
    }
    print(json.dumps({'action': args.action, 'identity': identity,
                      'train_rows': cache['manifest_rows']['train'],
                      'validation_rows': cache['manifest_rows']['validation']},
                     indent=2), flush=True)
    if args.action == 'check':
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Full L0 fit must run on a CUDA GPU')
    os.environ['FUSION_IMAGE_CACHE_ROOT'] = str(args.cache.resolve())
    os.environ['FUSION_IMAGE_CACHE_PRELOAD'] = '1'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    spec = NewImageModelSpec(
        variant='btsbot_images', image_initialization='pretrained',
        hf_snapshot=plan['hf_snapshot'],
        hf_weights_sha256=plan['hf_weights_sha256'],
        source_revision=plan['model_sha'],
    )
    output = args.output.resolve()
    fit_summary = output / 'training_summary.json'
    if output.exists() and any(output.iterdir()) and not fit_summary.is_file():
        raise FileExistsError(
            f'Partial outputs at {output}; inspect rather than overwrite/retrain'
        )
    # This is the same explicit initialization as the successful v3 GPU benchmark.
    started = time.perf_counter()
    _, cfg = build_model(spec, seed=135, device='cuda')
    index = load_index(plan['new_index'])
    trainer = NewImageTrainer(
        spec=spec, policy=policy, index=index, runtime_config=cfg,
        id_col=ds.object_id_column, label_col=ds.label_column,
        device='cuda', initialization='random',
    )
    request = TrainingRequest(
        train_manifest=Path(plan['l0_manifest']),
        validation_manifest=ds.validation_manifest,
        output_dir=output,
        run_id='fusion_l0_full_cached_pretrained_seed135_v1',
        cycle=0, strategy='bootstrap', seed=135, initial_checkpoint=None,
    )
    completed = trainer.completed_result(request)
    if completed is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Unrecognized existing output directory: {output}')
    result = completed if completed is not None else trainer.train(request)
    # Validation-only evaluation, using existing 5-class object aggregation.
    # The test split is intentionally never requested by this script.
    evaluator = NewImageEvaluator(
        spec=spec, index=index,
        manifests={'validation': ds.validation_manifest},
        id_col=ds.object_id_column, label_col=ds.label_column,
        device='cuda', batch_size=policy.batch_size,
    )
    evaluation = evaluator.evaluate(
        checkpoint_path=result.checkpoint_path, output_dir=output,
    )
    validation = evaluation.split('validation')
    history = json.loads(result.history_path.read_text(encoding='utf-8'))
    summary = {
        'status': 'completed',
        'identity': identity,
        'end_to_end_seconds': time.perf_counter() - started,
        'best_epoch': result.best_epoch,
        'epochs_completed': result.epochs_completed,
        'fit_seconds': result.elapsed_seconds,
        'last_epoch_seconds': history[-1]['epoch_seconds'],
        'best_validation_loss': result.best_validation_loss,
        'validation_object_metrics': validation.object_metrics,
        'validation_sample_metrics': validation.sample_metrics,
        'checkpoint': str(result.checkpoint_path),
        'validation_per_class': str(output / 'validation_per_class_metrics.json'),
    }
    write_json_atomic(output / 'cached_l0_baseline_report.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
