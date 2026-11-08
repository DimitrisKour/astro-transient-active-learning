#!/usr/bin/env python3
"""Independent, audited, five-class BTSbot L0 matrix. No AL acquisition.

prepare: fail-closed provenance + exact historic L0 membership; creates immutable plan.
run: paired init/seed model training; pilot uses tiny frozen smoke split.
report: summarize *completed* validation/test evaluations, no tuning on test.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.backends import (
    ImageTrainingPolicy, NewImageEvaluator, NewImageTrainer,
)
from al_pipeline.astronomy.fusion_image.contracts import (
    CLASS_NAMES, label_index, load_index, load_rows, object_labels, sha256, write_json_atomic,
)
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model
from al_pipeline.core.training.base import TrainingRequest

SEEDS = (135, 246, 357)
INITS = ('pretrained', 'randomized')
RUNS = tuple((init, seed) for init in INITS for seed in SEEDS)
SCRIPT_REL = 'scripts/archive/fusion_image/early_l0/fusion_l0_matrix.py'
SOURCE_FILES = (SCRIPT_REL, 'src/al_pipeline/astronomy/fusion_image/contracts.py',
                'src/al_pipeline/astronomy/fusion_image/data.py',
                'src/al_pipeline/astronomy/fusion_image/model.py',
                'src/al_pipeline/astronomy/fusion_image/backends.py',
                'src/al_pipeline/astronomy/astrominn/inference/aggregation.py',
                'src/al_pipeline/astronomy/data/registry.py',
                'src/al_pipeline/core/training/base.py',
                'src/al_pipeline/core/evaluation/base.py',
                'src/al_pipeline/core/evaluation/metrics.py')


def source_hashes(repo: Path) -> dict[str, str]:
    return {rel: sha256(repo/rel) for rel in SOURCE_FILES}


def historical_splits(registry: Path):
    d = DatasetRegistry.from_toml(registry).resolve_profile('astrominn_ztf').dataset
    for name in ('train_manifest', 'validation_manifest', 'test_manifest'):
        if getattr(d, name) is None:
            raise ValueError(f'Historical {name} missing from registry')
    if not d.object_id_column or not d.label_column:
        raise ValueError('Missing object/label column in historical dataset')
    return d


def _manifest_info(path: Path, ds):
    rows = load_rows(path)
    objs = object_labels(rows, id_col=ds.object_id_column, label_col=ds.label_column)
    return rows, objs


def prepare(args):
    run_root, repo = args.run_root.resolve(), args.repo.resolve()
    audit = json.loads(args.audit.read_text())
    for k in ('full_historical_npz_coverage', 'all_historical_labels_verified',
              'stage_probe_passed', 'all_historical_stages_verified'):
        if audit.get(k) is not True:
            raise ValueError(f'Full historical audit did not pass: {k}')
    for split in ('train', 'validation', 'test'):
        st = audit['stage_checks'][split]
        if st['failed_rows'] or st['checked_rows'] != st['compatible_rows'] or \
           st['checked_rows'] != audit['historical_splits'][split]['sample_rows']:
            raise ValueError(f'Incomplete or failed audit of {split}')
    model_sha = subprocess.check_output(['git', '-C', str(args.upstream_repo),
                                         'rev-parse', 'HEAD'], text=True).strip()
    if model_sha != audit['new_model_repository_sha']:
        raise ValueError('Upstream code changed since stage audit')
    if sha256(args.new_index) != audit['new_index_sha256'] or \
       str(args.new_index.resolve()) != audit['new_index']:
        raise ValueError('NPZ index differs from audited data')
    if sha256(args.hf_snapshot/'pytorch_model.bin') != args.hf_weights_sha256:
        raise ValueError('BTSbot pretrained weights differ from validated checksum')
    ds = historical_splits(args.registry)
    for split, manifest in (('train', ds.train_manifest),
                            ('validation', ds.validation_manifest),
                            ('test', ds.test_manifest)):
        if sha256(manifest) != audit['historical_splits'][split]['manifest_sha256']:
            raise ValueError(f'Changed original {split} split')
    train_rows, train_obj = _manifest_info(ds.train_manifest, ds)
    _, valid_obj = _manifest_info(ds.validation_manifest, ds)
    _, test_obj = _manifest_info(ds.test_manifest, ds)
    l0_rows, l0_obj = _manifest_info(args.l0, ds)
    if not set(l0_obj).issubset(train_obj):
        raise ValueError('L0 contains objects outside historical TRAIN split')
    if set(l0_obj) & (set(valid_obj) | set(test_obj)):
        raise ValueError('Data leakage: L0 overlaps validation/test')
    if len(l0_obj) >= len(train_obj) or len(l0_obj) < 100:
        raise ValueError(f'Unexpected historical L0 object count: {len(l0_obj)}')
    # Match every labeled (object, alert, label) row including duplicates.
    def triplets(rows):
        return Counter((r[ds.object_id_column], int(r['alert']),
                        label_index(r[ds.label_column])) for r in rows)
    if triplets(l0_rows) - triplets(train_rows):
        raise ValueError('L0 contains rows not found in historical training data')
    counts = Counter(l0_obj.values())
    if set(counts) != set(range(5)) or any(c < 5 for c in counts.values()):
        raise ValueError('Historical L0 must have >=5 objects in every class')
    # The original Experiment B split was ~10%, not the complete historical train set.
    fraction = len(l0_obj)/len(train_obj)
    if not (0.08 <= fraction <= 0.12):
        raise ValueError(f'Unexpected L0 fraction {fraction:.3f}; inspect input manifest')
    for name, p in (('pilot_train', args.pilot_train), ('pilot_validation', args.pilot_validation)):
        rows, obj = _manifest_info(p, ds)
        expected = train_obj if name == 'pilot_train' else valid_obj
        if not set(obj).issubset(expected):
            raise ValueError(f'{name} contains objects outside its intended historical split')
        if set(obj.values()) != set(range(5)):
            raise ValueError(f'{name} must contain all five classes')
    identity = {
        'version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'repo': str(repo), 'upstream_repo': str(args.upstream_repo.resolve()),
        'model_sha': model_sha,
        'registry': str(args.registry.resolve()),
        'audit': str(args.audit.resolve()),
        'new_index': str(args.new_index.resolve()),
        'hf_snapshot': str(args.hf_snapshot.resolve()),
        'hf_weights_sha256': args.hf_weights_sha256,
        'l0_manifest': str(args.l0.resolve()),
        'pilot_train': str(args.pilot_train.resolve()),
        'pilot_validation': str(args.pilot_validation.resolve()),
        'source_hashes': source_hashes(repo),
        'file_hashes': {name: sha256(p) for name, p in {
            'registry': args.registry, 'audit': args.audit,
            'new_index': args.new_index, 'l0_manifest': args.l0,
            'train_manifest': ds.train_manifest,
            'validation_manifest': ds.validation_manifest,
            'test_manifest': ds.test_manifest,
            'pilot_train': args.pilot_train,
            'pilot_validation': args.pilot_validation,
        }.items()},
        'historical_train_objects': len(train_obj),
        'l0_objects': len(l0_obj), 'l0_rows': len(l0_rows),
        'l0_class_counts': {CLASS_NAMES[k]: counts[k] for k in range(5)},
        'runs': [{'initialization': ini, 'seed': seed} for ini, seed in RUNS],
        'training_policy': asdict(ImageTrainingPolicy(
            max_epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, learning_rate=args.lr,
            class_weight_exponent=args.class_weight_exponent,
            weighting='per_sample', checkpoint_metric='loss')),
        'pilot_policy': asdict(ImageTrainingPolicy(max_epochs=1, patience=1,
            batch_size=4, learning_rate=args.lr, max_training_batches=2)),
        'test_set_report_only': True,
        'no_active_learning_acquisition': True,
    }
    plan_path = run_root/'plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        comparable = dict(identity)
        comparable['created_utc'] = previous['created_utc']
        if previous != comparable:
            raise FileExistsError('Different L0 plan already exists: use another --run-root')
        print(f'Existing identical plan verified: {plan_path}', flush=True)
    else:
        if run_root.exists() and list(run_root.iterdir()):
            raise FileExistsError('Nonempty run-root without plan; inspect before proceeding')
        write_json_atomic(plan_path, identity)
    print(json.dumps({'plan': str(plan_path), 'l0_objects':len(l0_obj),
                      'l0_rows': len(l0_rows), 'l0_fraction':fraction,
                      'class_counts':identity['l0_class_counts'],
                      'full_fits':len(RUNS),'gpu_pilots':2}, indent=2), flush=True)


def verify(plan_path: Path):
    plan = json.loads(plan_path.read_text())
    repo = Path(plan['repo'])
    if source_hashes(repo) != plan['source_hashes']:
        raise ValueError('L0 source changed after the plan was written')
    for key, expected in plan['file_hashes'].items():
        lookup = 'l0_manifest' if key == 'l0_manifest' else key
        if key in ('train_manifest','validation_manifest','test_manifest'):
            ds = historical_splits(Path(plan['registry']))
            path = getattr(ds, key)
        else:
            path = Path(plan[lookup])
        if sha256(path) != expected:
            raise ValueError(f'Changed input: {key}: {path}')
    model_sha = subprocess.check_output(['git','-C',plan['upstream_repo'],
                                         'rev-parse','HEAD'],text=True).strip()
    if model_sha != plan['model_sha']:
        raise ValueError('Model code revision changed')
    if sha256(Path(plan['hf_snapshot'])/'pytorch_model.bin') != plan['hf_weights_sha256']:
        raise ValueError('BTSbot weights changed')
    audit = json.loads(Path(plan['audit']).read_text())
    if not all(audit.get(k) is True for k in ('all_historical_stages_verified',
                     'all_historical_labels_verified','full_historical_npz_coverage')):
        raise ValueError('Historical audit no longer validates all data')
    return plan


def run(args):
    root = args.run_root.resolve()
    plan = verify(root/'plan.json')
    if args.pilot:
        if args.initialization not in INITS or args.seed != 135:
            raise ValueError('Pilot uses seed 135 for either initialization')
    else:
        if (args.initialization, args.seed) not in RUNS:
            raise ValueError('Unexpected matrix member')
        for initialization in INITS:
            if not (root/'pilots'/initialization/'training_summary.json').is_file():
                raise RuntimeError('Both GPU pilots must succeed before full matrix')
    ds = historical_splits(Path(plan['registry']))
    spec = NewImageModelSpec(variant='btsbot_images',
        image_initialization=args.initialization,
        hf_snapshot=plan['hf_snapshot'], hf_weights_sha256=plan['hf_weights_sha256'],
        source_revision=plan['model_sha'])
    policy = ImageTrainingPolicy(**(plan['pilot_policy'] if args.pilot else plan['training_policy']))
    device = args.device
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU job requested but torch.cuda.is_available() is false')
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    model, runtime = build_model(spec, seed=args.seed, device=device)
    del model
    index = load_index(plan['new_index'])
    out = (root/'pilots'/args.initialization if args.pilot else
           root/'fits'/args.initialization/f'seed_{args.seed}')
    trainer = NewImageTrainer(spec=spec, policy=policy, index=index,
        runtime_config=runtime, id_col=ds.object_id_column,
        label_col=ds.label_column, device=device, initialization='random')
    request = TrainingRequest(
        train_manifest=Path(plan['pilot_train'] if args.pilot else plan['l0_manifest']),
        validation_manifest=Path(plan['pilot_validation'] if args.pilot else ds.validation_manifest),
        output_dir=out, run_id=f'fusion_l0_{args.initialization}_seed{args.seed}' + ('_pilot' if args.pilot else ''),
        cycle=0, strategy='bootstrap', seed=args.seed, initial_checkpoint=None)
    completed = trainer.completed_result(request)
    print(json.dumps({'mode':'pilot' if args.pilot else 'l0',
         'init':args.initialization,'seed':args.seed,'output':str(out),
         'epochs':policy.max_epochs,'samples':len(load_rows(request.train_manifest)),
         'device':device},indent=2),flush=True)
    result = completed if completed is not None else trainer.train(request)
    print(json.dumps({'training_complete':str(result.checkpoint_path),
                      'elapsed_seconds':result.elapsed_seconds,
                      'best_epoch':result.best_epoch},indent=2),flush=True)
    evaluator = NewImageEvaluator(spec=spec,index=index,
        manifests={'validation':Path(plan['pilot_validation'] if args.pilot else ds.validation_manifest),
                   **({} if args.pilot else {'test':ds.test_manifest})},
        id_col=ds.object_id_column,label_col=ds.label_column,
        device=device,batch_size=policy.batch_size)
    evaluation = evaluator.evaluate(checkpoint_path=result.checkpoint_path,output_dir=out)
    print(json.dumps({'evaluation_summary':str(evaluation.summary_path),
                      'test_report_only':True},indent=2),flush=True)


def report(args):
    root = args.run_root.resolve()
    plan = verify(root/'plan.json')
    records = []
    for ini, seed in RUNS:
        fit_dir = root/'fits'/ini/f'seed_{seed}'
        sp, ep = fit_dir/'training_summary.json',fit_dir/'evaluation_summary.json'
        if not (sp.is_file() and ep.is_file()):
            print(f'INCOMPLETE {ini} seed {seed}',flush=True)
            continue
        summary, evaluation = json.loads(sp.read_text()),json.loads(ep.read_text())
        if summary['seed'] != seed or summary['model_spec']['image_initialization'] != ini:
            raise ValueError(f'Run identity mismatch: {fit_dir}')
        by_split = {s['split']:s for s in evaluation['splits']}
        if set(by_split) != {'validation','test'}:
            raise ValueError(f'Missing required evaluated split for {ini}/{seed}')
        def get(s, key):
            return by_split[s]['object_metrics'].get(key)
        records.append({'initialization':ini,'seed':seed,
             'best_epoch':summary['best_epoch'],
             'epochs_completed':summary['epochs_completed'],
             'training_seconds':summary['elapsed_seconds'],
             'validation_f1_macro':get('validation','f1_macro'),
             'test_f1_macro_report_only':get('test','f1_macro'),
             'validation_balanced_accuracy':get('validation','balanced_accuracy'),
             'test_balanced_accuracy_report_only':get('test','balanced_accuracy'),
             'training_summary':str(sp),'evaluation_summary':str(ep)})
    out = {'expected_runs':len(RUNS),'completed_runs':len(records),
           'results':records,'warning':'Test numbers are descriptive only; never select models from test.'}
    write_json_atomic(root/'matrix_report.json',out)
    print(json.dumps(out,indent=2),flush=True)
    if len(records) != len(RUNS):
        raise RuntimeError('L0 matrix is incomplete')


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command', required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--repo',type=Path,required=True)
    prep.add_argument('--run-root',type=Path,required=True)
    prep.add_argument('--registry',type=Path,required=True)
    prep.add_argument('--audit',type=Path,required=True)
    prep.add_argument('--new-index',type=Path,required=True)
    prep.add_argument('--upstream-repo',type=Path,required=True)
    prep.add_argument('--hf-snapshot',type=Path,required=True)
    prep.add_argument('--hf-weights-sha256',required=True)
    prep.add_argument('--l0',type=Path,required=True)
    prep.add_argument('--pilot-train',type=Path,required=True)
    prep.add_argument('--pilot-validation',type=Path,required=True)
    prep.add_argument('--epochs',type=int,default=30)
    prep.add_argument('--patience',type=int,default=6)
    prep.add_argument('--batch-size',type=int,default=64)
    prep.add_argument('--lr',type=float,default=1e-4)
    prep.add_argument('--class-weight-exponent',type=float,default=0.85)
    training=sub.add_parser('run')
    training.add_argument('--run-root',type=Path,required=True)
    training.add_argument('--initialization',choices=INITS,required=True)
    training.add_argument('--seed',type=int,required=True)
    training.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    training.add_argument('--pilot',action='store_true')
    rpt=sub.add_parser('report')
    rpt.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare': prepare(args)
    elif args.command=='run': run(args)
    else: report(args)

if __name__=='__main__': main()
