#!/usr/bin/env python3
"""Prepare, validate, run and report PRE-Task-0 full-history recipe fits.

No test data, no AL oracle/pool, no implicit GPU submissions.
Use --prepare after the complete historical cache is finalized and verified.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.cache_helpers import preprocessing_identity
from al_pipeline.astronomy.fusion_image.contracts import (
    class_order_config, label_index, load_rows, object_labels, sha256, write_json_atomic,
)
from al_pipeline.astronomy.fusion_image.recipe_training import Recipe, train_recipe
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec

BASE = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
SOURCE_PLAN = BASE/'new_image_l0_matrix_v1/plan.json'
CACHE = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_train_cache_v1')
OUTPUT = BASE/'new_image_pre_task0_recipe_v1'
SOURCE_LABEL = 'historical_full_train_plus_fixed_validation'

CASES = {
    'constant': Recipe(name='constant'),
    'cosine': Recipe(name='cosine', lr_schedule='warmup_cosine'),
    'cosine_aug': Recipe(name='cosine_aug', lr_schedule='warmup_cosine',image_augmentation=True),
    'cosine_aug_ema': Recipe(name='cosine_aug_ema', lr_schedule='warmup_cosine',
                             image_augmentation=True,ema=True),
    'cosine_aug_effnum': Recipe(name='cosine_aug_effnum',lr_schedule='warmup_cosine',
                                image_augmentation=True,class_weighting='effective_number'),
}
DEFAULT_CASES = ('constant','cosine','cosine_aug')


def get_context(source_plan: Path, cache_root: Path):
    """Fail closed on full training extent, model, cache, splits, and labels."""
    old = json.loads(source_plan.read_text())
    finalized = json.loads((cache_root/'cache_manifest.json').read_text())
    cache_plan = json.loads((cache_root/'plan.json').read_text())
    identity = json.loads((cache_root/'full_historical_identity.json').read_text())
    if finalized.get('source') != SOURCE_LABEL or finalized.get('labels_saved_in_cache') is not False:
        raise ValueError('Not the immutable full-historical label-free cache')
    if finalized['cache_version'] != 2 or sha256(cache_root/'plan.json') != finalized['plan_sha256']:
        raise ValueError('Finalized cache plan identity mismatch')
    if finalized['manifest_rows']['train'] != 29161 or finalized['manifest_rows']['validation'] != 14148:
        raise ValueError('Unexpected historical full training/validation sample counts')
    if identity['training_objects'] != 4729 or identity['training_rows'] != 29161:
        raise ValueError('Cache describes AL initial subset rather than complete history')
    if identity['validation_objects'] != 2365 or identity['validation_rows'] != 14148:
        raise ValueError('Unexpected historical validation count')
    if sha256(source_plan) != identity['source_matrix_plan_sha256']:
        raise ValueError('Original experiment source plan was modified')
    if finalized['new_index_sha256'] != sha256(old['new_index']):
        raise ValueError('New NPZ index mismatch')
    if finalized['manifest_sha256'] != cache_plan['manifest_sha256']:
        raise ValueError('Cache manifests differ from audited plan')
    rev = subprocess.check_output(['git','-C',old['upstream_repo'],'rev-parse','HEAD'], text=True).strip()
    if rev != old['model_sha']:
        raise ValueError('Upstream model source changed')
    if sha256(Path(old['hf_snapshot'])/'pytorch_model.bin') != old['hf_weights_sha256']:
        raise ValueError('Pretrained checkpoint checksum changed')
    ds = DatasetRegistry.from_toml(old['registry']).resolve_profile('astrominn_ztf').dataset
    train = Path(ds.train_manifest)
    val = Path(ds.validation_manifest)
    if sha256(train) != finalized['manifest_sha256']['train'] or sha256(val) != finalized['manifest_sha256']['validation']:
        raise ValueError('Registry manifests differ from finalized cache')
    if str(train.resolve()) != str(Path(identity['full_train_manifest']).resolve()):
        raise ValueError('Training population differs from complete historical split')
    if sha256(old['l0_manifest']) == sha256(train):
        raise ValueError('Not allowed to use the old AL initial L0 as Task 0 full history')
    train_rows, val_rows = load_rows(train), load_rows(val)
    train_labels = object_labels(train_rows, id_col=ds.object_id_column,label_col=ds.label_column)
    val_labels = object_labels(val_rows,id_col=ds.object_id_column,label_col=ds.label_column)
    if set(train_labels) & set(val_labels):
        raise ValueError('Train/validation objects overlap')
    if (len(train_labels),len(val_labels)) != (4729,2365):
        raise ValueError('Object counts differ from audited full history')
    # Verify original cache used same preprocessing as this trainer.
    from fusion_model_basics.config import build_astrominn_config
    cfg = build_astrominn_config('btsbot_images',coord_checkpoint=None,
          hf_image_model_dir=old['hf_snapshot'],classes=class_order_config(),
          device='cpu',dir='.')
    if preprocessing_identity(cfg) != finalized['preprocessing_identity']:
        raise ValueError('Upstream preprocessing/config changed since caching')
    for shard in range(cache_plan['n_shards']):
        s = cache_root/f'shard_{shard:03d}'
        done = json.loads((s/'done.json').read_text())
        if done.get('plan_sha256') != finalized['plan_sha256']:
            raise ValueError(f'Shard {shard} plan identity mismatch')
        for file in ('images.npy','metadata.npy','stage_index.csv'):
            if not (s/file).is_file():
                raise FileNotFoundError(s/file)
    return old, ds, cfg, train, val, train_labels, val_labels


def recipe_plan(*, source_plan: Path, cache_root: Path, output: Path):
    old, ds, cfg, train, val, train_labels, val_labels = get_context(source_plan, cache_root)
    file = Path(__file__).resolve()
    modfile = Path(sys.modules[Recipe.__module__].__file__).resolve()
    from al_pipeline.astronomy.fusion_image import model, cache_v2, contracts, backends
    impl_checksums = {
        'runner':sha256(file), 'recipe_training':sha256(modfile),
        'model':sha256(model.__file__), 'cache_v2':sha256(cache_v2.__file__),
        'contracts':sha256(contracts.__file__), 'backends':sha256(backends.__file__),
    }
    payload = {
        'version': 1, 'experiment': 'pre_task0_training_recipe_full_history',
        'source_plan': str(source_plan.resolve()), 'source_plan_sha256':sha256(source_plan),
        'cache_root':str(cache_root.resolve()),
        'cache_manifest_sha256':sha256(cache_root/'cache_manifest.json'),
        'train_manifest':str(train.resolve()), 'train_manifest_sha256':sha256(train),
        'validation_manifest':str(val.resolve()), 'validation_manifest_sha256':sha256(val),
        'training_objects':len(train_labels), 'validation_objects':len(val_labels),
        'training_rows':29161,'validation_rows':14148,
        'train_object_class_counts':dict(Counter(map(str, train_labels.values()))),
        'train_sample_class_counts':dict(Counter(str(label_index(r[ds.label_column])) for r in load_rows(train))),
        'model_spec':asdict(NewImageModelSpec(variant='btsbot_images',image_initialization='pretrained',
                      hf_snapshot=old['hf_snapshot'],hf_weights_sha256=old['hf_weights_sha256'],
                      source_revision=old['model_sha'])),
        'recipes':{name:asdict(recipe) for name,recipe in CASES.items()},
        'default_cases':list(DEFAULT_CASES),
        'source_checksums':impl_checksums,
        'output_dir':str(output.resolve()),
        'test_access':False, 'acquisition':False,
        'checkpoint_selection':'validation_loss',
    }
    return payload,(old,ds,cfg,train,val)


def ensure_plan(args):
    planned,ctx = recipe_plan(source_plan=args.source_plan,cache_root=args.cache,output=args.output)
    file = args.output/'plan.json'
    if file.is_file():
        if json.loads(file.read_text()) != planned:
            raise ValueError(f'Frozen experiment plan changed: {file}; use a new output directory')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Refusing to create plan in existing nonempty output folder')
        args.output.mkdir(parents=True,exist_ok=True)
        write_json_atomic(file, planned)
    return planned,ctx


def launch(args):
    plan,(_,ds,cfg,train,val) = ensure_plan(args)
    recipe_name = args.case
    if recipe_name not in plan['recipes']:
        raise ValueError('Unknown frozen recipe name')
    recipe = Recipe(**plan['recipes'][recipe_name])
    spec = NewImageModelSpec(**plan['model_spec'])
    return train_recipe(recipe=recipe, spec=spec,runtime_config=cfg,
         cache_root=args.cache,train_manifest=train,validation_manifest=val,
         id_col=ds.object_id_column,label_col=ds.label_column,
         output_dir=args.output/recipe_name,
         identity={'plan_sha256':sha256(args.output/'plan.json'),
                   'case':recipe_name,'recipe':asdict(recipe),
                   'train_manifest_sha256':plan['train_manifest_sha256'],
                   'validation_manifest_sha256':plan['validation_manifest_sha256'],
                   'cache_manifest_sha256':plan['cache_manifest_sha256'],
                   'test_access':False})


def report(output: Path):
    plan=json.loads((output/'plan.json').read_text())
    rows=[]
    for name, recipe in plan['recipes'].items():
        summary=output/name/'training_summary.json'
        if not summary.is_file():
            rows.append({'case':name,'status':'not_completed'})
            continue
        obj=json.loads(summary.read_text())
        ident=obj['identity']
        if ident['plan_sha256'] != sha256(output/'plan.json') or ident['case'] != name:
            raise ValueError(f'Invalid finished run identity: {name}')
        metrics=obj['validation_object_metrics']
        rows.append({'case':name,'status':'completed','best_epoch':obj['best_epoch'],
             'epochs_completed':obj['epochs_completed'],
             'validation_loss':obj['best_validation_loss'],
             'object_macro_f1':metrics['f1_macro'],
             'object_balanced_accuracy':metrics['balanced_accuracy'],
             'object_accuracy':metrics['accuracy'],
             'best_f1_during_fit':obj['best_macro_f1_during_fit'],
             'fit_seconds':obj['fit_seconds'],'wall_seconds':obj['end_to_end_seconds']})
    with (output/'comparison.csv').open('w',newline='') as f:
        cols=['case','status','best_epoch','epochs_completed','validation_loss',
              'object_macro_f1','object_balanced_accuracy','object_accuracy',
              'best_f1_during_fit','fit_seconds','wall_seconds']
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader(); writer.writerows(rows)
    write_json_atomic(output/'comparison.json',{'cases':rows,
        'warning':'Single-seed model-selection pilot; no statistical claims; validation only.'})
    print(json.dumps(rows,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=('check','prepare','run','report'))
    ap.add_argument('--source-plan',type=Path,default=SOURCE_PLAN)
    ap.add_argument('--cache',type=Path,default=CACHE)
    ap.add_argument('--output',type=Path,default=OUTPUT)
    ap.add_argument('--case',choices=list(CASES))
    args=ap.parse_args()
    if args.command=='report':
        report(args.output)
    elif args.command=='check':
        plan,_=recipe_plan(source_plan=args.source_plan,cache_root=args.cache,output=args.output)
        print(json.dumps({'validated':True,'cache':plan['cache_root'],
                          'train_objects':plan['training_objects'],
                          'train_rows':plan['training_rows'],
                          'validation_objects':plan['validation_objects'],
                          'cases':list(plan['recipes']),
                          'submitted_jobs':0},indent=2),flush=True)
    elif args.command=='prepare':
        plan,_=ensure_plan(args)
        print(json.dumps({'plan':str(args.output/'plan.json'),
                          'train_rows':plan['training_rows'],
                          'validation_rows':plan['validation_rows'],
                          'default_cases':plan['default_cases'],
                          'submitted_jobs':0},indent=2),flush=True)
    else:
        if not args.case:
            ap.error('run requires --case')
        print(json.dumps(launch(args),indent=2),flush=True)

if __name__=='__main__':
    main()
