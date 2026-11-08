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
import errno
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.cache_helpers import preprocessing_identity
from al_pipeline.astronomy.fusion_image.contracts import (
    class_order_config, label_index, load_rows, object_labels, sha256, write_json_atomic,
)
from al_pipeline.astronomy.fusion_image.recipe_training_pretask0_ext import Recipe, train_recipe
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec

BASE = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
SOURCE_PLAN = BASE/'new_image_l0_matrix_v1/plan.json'
SOURCE_CACHE = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_train_cache_v1')
CACHE = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_hdd_v1')
OUTPUT = BASE/'new_image_pre_task0_extensions_v1'
SOURCE_LABEL = 'historical_full_train_plus_fixed_validation'

CASES = {
    'cosine_aug_short': Recipe(name='cosine_aug_short',
                              lr_schedule='warmup_cosine',
                              image_augmentation=True,
                              cosine_decay_epochs=25),
    'cosine_aug_effnum': Recipe(name='cosine_aug_effnum',
                               lr_schedule='warmup_cosine',
                               image_augmentation=True,
                               class_weighting='effective_number',
                               effective_count_basis='samples',
                               effective_beta=0.999),
}
DEFAULT_CASES = tuple(CASES)



def sha256_with_eio_retry(path: Path, retries: int = 3) -> str:
    """Preserve full source verification; retry only transient filesystem EIO."""
    for attempt in range(retries):
        try:
            return sha256(path)
        except OSError as err:
            if err.errno != errno.EIO or attempt + 1 >= retries:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise AssertionError('unreachable')


def check_fast_cache(cache_root: Path, *, source_cache: Path, source_final: dict) -> dict:
    """Validate immutable consolidated cache identity WITHOUT a second 2-GB copy."""
    import numpy as np
    cache_root = Path(cache_root)
    marker = cache_root / 'fast_cache_manifest.json'
    fast = json.loads(marker.read_text())
    if fast.get('format_version') != 1 or fast.get('source') != SOURCE_LABEL:
        raise ValueError('Expected complete historical consolidated cache')
    if fast.get('writer') != 'buffered_npy_sequential_v2':
        raise ValueError('Unknown cache builder')
    if fast.get('labels_saved_in_cache') is not False or fast.get('test_split_accessed') is not False:
        raise ValueError('Cache contains forbidden labels or test data')
    if fast.get('manifest_rows') != {'train':29161,'validation':14148}:
        raise ValueError('Fast cache is not the complete historical dataset')
    if fast.get('unique_objects') != 7094 or fast.get('unique_stages') != 43309:
        raise ValueError('Unexpected consolidated cache extent')
    if Path(fast['source_cache']).resolve() != Path(source_cache).resolve():
        raise ValueError('Consolidated cache comes from wrong source')
    if fast['source_cache_manifest_sha256'] != sha256(Path(source_cache) / 'cache_manifest.json'):
        raise ValueError('Source historical cache identity mismatch')
    for k in ('manifest_sha256','preprocessing_identity'):
        if fast.get(k) != source_final[k]:
            raise ValueError(f'Consolidated cache differs in {k}')
    if fast['stage_index_sha256'] != sha256(cache_root/'stage_index.csv'):
        raise ValueError('Consolidated index checksum mismatch')
    expected = (("images.npy", (43309,3,63,63), 'images_nbytes'),
                ("metadata.npy", (43309,55), 'metadata_nbytes'))
    for filename, shape, nbytes_key in expected:
        path=cache_root/filename
        if path.stat().st_size != fast[nbytes_key]:
            raise ValueError(f'Consolidated array size differs for {filename}')
        x=np.load(path, mmap_mode='r', allow_pickle=False)
        if x.shape != shape or x.dtype != np.dtype('float32'):
            raise ValueError(f'Consolidated array schema mismatch for {filename}')
        del x
    return fast


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
    if finalized['new_index_sha256'] != sha256_with_eio_retry(Path(old['new_index'])):
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
    old, ds, cfg, train, val, train_labels, val_labels = get_context(source_plan, SOURCE_CACHE)
    check_fast_cache(cache_root, source_cache=SOURCE_CACHE, source_final=json.loads((SOURCE_CACHE/'cache_manifest.json').read_text()))
    file = Path(__file__).resolve()
    modfile = Path(sys.modules[Recipe.__module__].__file__).resolve()
    from al_pipeline.astronomy.fusion_image import model, cache_v2, contracts, backends, fast_cache, resident_cache
    impl_checksums = {
        'runner':sha256(file), 'recipe_training':sha256(modfile),
        'model':sha256(model.__file__), 'cache_v2':sha256(cache_v2.__file__),
        'contracts':sha256(contracts.__file__), 'backends':sha256(backends.__file__),
        'fast_cache':sha256(fast_cache.__file__),
        'resident_cache':sha256(resident_cache.__file__),
    }
    payload = {
        'version': 4, 'experiment': 'pre_task0_training_recipe_extensions_full_history_resident',
        'source_plan': str(source_plan.resolve()), 'source_plan_sha256':sha256(source_plan),
        'cache_root':str(cache_root.resolve()),
        'source_sharded_cache':str(SOURCE_CACHE.resolve()),
        'cache_manifest_sha256':sha256(cache_root/'fast_cache_manifest.json'),
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
        'experiment_note':'Two individually controlled changes from cosine_aug: 25-epoch cosine horizon OR sample-count effective-number weights.',
        'data_access_mode':'shared_host_resident_arrays_v3',
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
             'fit_seconds':obj['fit_seconds'],'wall_seconds':obj['end_to_end_seconds'],
             'model_init_seconds':obj.get('setup_timing',{}).get('model_init_seconds'),
             'dataset_init_seconds':obj.get('setup_timing',{}).get('dataset_init_seconds'),
             'pretraining_setup_seconds':obj.get('setup_timing',{}).get('total_setup_seconds')})
    with (output/'comparison.csv').open('w',newline='') as f:
        cols=['case','status','best_epoch','epochs_completed','validation_loss',
              'object_macro_f1','object_balanced_accuracy','object_accuracy',
              'best_f1_during_fit','fit_seconds','wall_seconds',
              'model_init_seconds','dataset_init_seconds','pretraining_setup_seconds']
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
                          'cache_type':'consolidated_hdd_loaded_to_shared_host_RAM',
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
