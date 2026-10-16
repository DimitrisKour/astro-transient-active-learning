#!/usr/bin/env python3
"""Opt-in compatible full closed-loop runner for the new image+metadata model.

Fails closed unless a complete original-historical-NPZ audit is explicitly
provided. Uses the existing SimulatedActiveLearningExperiment unchanged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import torch

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.experiments.simulated import (
    SimulatedALConfig, SimulatedActiveLearningExperiment, load_hidden_labels_csv,
)
from al_pipeline.astronomy.fusion_image.backends import (
    ImageTrainingPolicy, NewImageEvaluator, NewImagePoolPredictor, NewImageTrainer,
)
from al_pipeline.astronomy.fusion_image.contracts import CLASS_NAMES, LABEL_INDEX, load_index, sha256
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model
from al_pipeline.core.experiments.batch_policy import BatchAcquisitionConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--new-index', type=Path, required=True)
    ap.add_argument('--new-root', type=Path)
    ap.add_argument('--upstream-repo', type=Path, required=True)
    ap.add_argument('--suite-id', required=True)
    ap.add_argument('--pool-path', type=Path, required=True)
    ap.add_argument('--hidden-labels', type=Path, required=True)
    ap.add_argument('--l0-manifest', type=Path, required=True)
    ap.add_argument('--baseline-checkpoint', type=Path, required=True)
    ap.add_argument('--policy', choices=['random', 'entropy', 'entropy_diversity',
                                        'entropy_balanced_explore'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--cycle', type=int, required=True)
    ap.add_argument('--init-mode', choices=['restart_from_random', 'restart_from_baseline',
                                           'warm_start'], required=True)
    ap.add_argument('--variant', choices=['patch_images','btsbot_images'], required=True)
    ap.add_argument('--image-initialization', choices=['randomized','pretrained'], required=True)
    ap.add_argument('--hf-snapshot', type=Path)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--class-weight-exponent', type=float, default=0.85)
    ap.add_argument('--weighting', choices=['per_sample', 'equal_object'], default='per_sample')
    ap.add_argument('--checkpoint-metric', choices=['loss', 'object_macro_f1'], default='loss')
    ap.add_argument('--acquisition-size', type=int, default=473)
    ap.add_argument('--max-cycles', type=int, default=9)
    ap.add_argument('--run', action='store_true', help='No training without explicit --run')
    args = ap.parse_args()
    if not args.suite_id.startswith('fusion_image_'):
        raise ValueError('Use a new fusion_image_* suite ID; do not overwrite previous suites')
    if args.cycle < 1 or args.max_cycles < args.cycle:
        raise ValueError('Invalid cycle')
    audited = json.loads(args.audit.read_text())
    if not all(audited.get(k) for k in ('full_historical_npz_coverage',
            'all_historical_stages_verified', 'all_historical_labels_verified')):
        raise ValueError('Not all historical objects and alert stages were verified on the new NPZs')
    source_sha = subprocess.check_output(['git','-C',str(args.upstream_repo),
                                          'rev-parse','HEAD'], text=True).strip()
    if audited.get('new_model_repository_sha') != source_sha:
        raise ValueError('Audit was produced with another upstream model revision')
    if audited.get('new_index') != str(args.new_index.resolve()) or \
       audited.get('new_index_sha256') != sha256(args.new_index):
        raise ValueError('New NPZ index changed since audit')
    if audited.get('new_root') != (str(args.new_root.resolve()) if args.new_root else None):
        raise ValueError('New NPZ root changed since audit')
    ds_before = DatasetRegistry.from_toml(args.registry).resolve_profile('astrominn_ztf').dataset
    for split, file in {'train':ds_before.train_manifest,
                         'validation':ds_before.validation_manifest,
                         'test':ds_before.test_manifest}.items():
        saved = audited['historical_splits'][split]['manifest_sha256']
        if saved != sha256(file):
            raise ValueError(f'Historical {split} manifest changed since audit')
    spec = NewImageModelSpec(variant=args.variant,
                             image_initialization=args.image_initialization,
                             hf_snapshot=str(args.hf_snapshot) if args.hf_snapshot else None,
         hf_weights_sha256=sha256(args.hf_snapshot/'pytorch_model.bin') if args.hf_snapshot else None,
                             source_revision=source_sha)
    model, runtime = build_model(spec, seed=args.seed, device=args.device)
    del model
    index = load_index(args.new_index, root=args.new_root)
    registry = DatasetRegistry.from_toml(args.registry)
    res = registry.resolve_profile('astrominn_ztf')
    dataset = replace(res.dataset, train_manifest=args.l0_manifest)
    labels = load_hidden_labels_csv(args.hidden_labels)
    mode = args.init_mode
    policy = ImageTrainingPolicy(max_epochs=args.epochs, patience=args.patience,
          batch_size=args.batch_size, class_weight_exponent=args.class_weight_exponent,
          weighting=args.weighting, checkpoint_metric=args.checkpoint_metric)
    predictor = NewImagePoolPredictor(spec=spec, index=index, pool_path=args.pool_path,
                                     device=args.device, batch_size=args.batch_size)
    trainer = NewImageTrainer(spec=spec, policy=policy, index=index, runtime_config=runtime,
                              id_col=dataset.object_id_column, label_col=dataset.label_column,
                              device=args.device,
                              initialization='random' if mode=='restart_from_random' else 'checkpoint')
    evaluator = NewImageEvaluator(spec=spec, index=index,
        manifests={'validation': dataset.validation_manifest, 'test': dataset.test_manifest},
        id_col=dataset.object_id_column, label_col=dataset.label_column,
        device=args.device, batch_size=args.batch_size)
    run_id = f'{args.suite_id}_{args.policy}_seed{args.seed}'
    rel = Path(args.suite_id)/args.policy/f'seed_{args.seed}'
    cfg = SimulatedALConfig(run_id=run_id, strategy=args.policy,
       acquisition_size=args.acquisition_size, max_cycles=args.max_cycles,
       seed=args.seed, training_initialization=mode,
       allow_partial_final_cycle=True,
       batch_policy=BatchAcquisitionConfig() if args.policy in {
           'entropy_diversity','entropy_balanced_explore'} else None)
    exp = SimulatedActiveLearningExperiment(config=cfg, dataset=dataset,
       pool_path=args.pool_path, hidden_labels=labels,
       run_dir=res.storage.persistent_root/'runs'/rel,
       artifact_dir=res.storage.data_root/'runs'/rel,
       baseline_checkpoint=args.baseline_checkpoint, predictor=predictor,
       trainer=trainer, evaluator=evaluator, class_names=CLASS_NAMES,
       label_mapping={label: CLASS_NAMES[idx] for label,idx in LABEL_INDEX.items()})
    print(json.dumps({'run_id':run_id,'cycle':args.cycle,'init_mode':mode,
          'variant':args.variant,'source_sha':source_sha,'will_run':args.run},indent=2))
    if not args.run:
        print('PREPARE ONLY: no acquisition/training submitted')
        return
    # Existing coordinator handles checkpoint continuity, fixed validation,
    # state writes and strategy acquisition. Never queue until pilot approved.
    result = exp.initialize()
    if result.last_completed_cycle not in (args.cycle-1,args.cycle):
        raise ValueError('Wrong preceding cycle; refusing to advance trajectory')
    if result.last_completed_cycle != args.cycle:
        exp.evaluate_baseline()
        exp.run_next_cycle()


if __name__ == '__main__':
    main()
