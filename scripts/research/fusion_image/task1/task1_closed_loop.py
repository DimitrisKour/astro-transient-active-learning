#!/usr/bin/env python3
"""Immutable nine-cycle ConvNeXt historical AL matrix on resident HDD caches.

The test split is a report-only sink, evaluated after the validation-selected
checkpoint has been saved. Acquisition sees no historical candidate labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.contracts import (
    CLASS_NAMES, label_index, load_rows, object_labels, sha256, write_json_atomic,
)
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec
from al_pipeline.astronomy.fusion_image.task1_acquisition import (
    POLICIES, PHASE1, PHASE2, choose,
)

ROOT=Path('/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task1_closed_loop_grouped_v1')
TASK0=Path('/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task0_full_historical_grouped_v1')
CACHE=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_hdd_v1')
TEST_CACHE=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_historical_test_fastcache_grouped_v1')
REPO=Path(os.environ.get('AL_PIPELINE_REPO', Path(__file__).resolve().parents[4]))
SEEDS=(135,246,357)
CONFIGS={
    'random_warm':{'initialization':'randomized','continuation':'warm_start'},
    'random_l0':{'initialization':'randomized','continuation':'restart_from_l0'},
    'pretrained_warm':{'initialization':'pretrained','continuation':'warm_start'},
}
ACQUISITION_SIZE=473
CYCLES=9
SOURCES=(
    'scripts/research/fusion_image/task1/task1_closed_loop.py',
    'src/al_pipeline/astronomy/fusion_image/task1_training.py',
    'src/al_pipeline/astronomy/fusion_image/task1_acquisition.py',
    'src/al_pipeline/astronomy/fusion_image/recipe_training_resident.py',
    'src/al_pipeline/astronomy/fusion_image/model.py',
    'src/al_pipeline/astronomy/fusion_image/fast_cache.py',
    'src/al_pipeline/astronomy/fusion_image/resident_cache.py',
    'src/al_pipeline/astronomy/fusion_image/test_fastcache_v1.py',
    'src/al_pipeline/astronomy/fusion_image/cache_helpers.py',
    'src/al_pipeline/astronomy/fusion_image/backends.py',
    'src/al_pipeline/astronomy/fusion_image/contracts.py',
    'src/al_pipeline/core/batch_selection.py',
    'src/al_pipeline/core/scoring/entropy.py',
    'src/al_pipeline/core/scoring/random.py',
    'src/al_pipeline/core/selection.py',
    'src/al_pipeline/core/evaluation/metrics.py',
    'src/al_pipeline/astronomy/astrominn/inference/aggregation.py',
)


def sha_sources(repo:Path)->dict:
    return {rel:sha256(repo/rel) for rel in SOURCES}


def stage_rows_no_labels(train:list[dict],id_col:str,l0_ids:set[str])->list[dict]:
    """Only candidate object IDs and latest historical alert stage, no targets."""
    latest={}
    for row in train:
        oid=str(row[id_col]);a=int(row['alert'])
        if oid not in l0_ids:
            latest[oid]=max(a,latest.get(oid,-1))
    return [{'object_id':oid,'alert':str(a)} for oid,a in sorted(latest.items())]


def all_train_rows_for_ids(rows:list[dict], ids:set[str], id_col:str)->list[dict]:
    """Oracle revelation: only chosen L_t object labels enter training manifest."""
    selected=[row for row in rows if row[id_col] in ids]
    present={str(r[id_col]) for r in selected}
    if present!=ids:
        raise ValueError(f'Labeled objects absent from historical training source: {len(ids-present)}')
    return selected


def write_csv(path:Path,rows:list[dict],fields:list[str]):
    if path.is_file():
        with path.open(newline='',encoding='utf-8') as f:
            prev=list(csv.DictReader(f))
        if prev!=rows:
            raise ValueError(f'Immutable CSV rows changed: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='raise')
        writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def prepare_plan(*,repo:Path=REPO,task0:Path=TASK0,cache:Path=CACHE,test_cache:Path=TEST_CACHE)->tuple[dict,list[dict],list[dict]]:
    frozen=json.loads((task0/'plan.json').read_text())
    if frozen.get('experiment')!='task0_full_history_btsbot_pretrained_vs_randomized':
        raise ValueError('Wrong Task 0 identity; requires full-history experiment')
    source=Path(frozen['source_plan'])
    if sha256(source)!=frozen['source_plan_sha256']:
        raise ValueError('Original L0 matrix source has changed since Task 0')
    src=json.loads(source.read_text())
    ds=DatasetRegistry.from_toml(src['registry']).resolve_profile('astrominn_ztf').dataset
    train,validation,test=(Path(ds.train_manifest),Path(ds.validation_manifest),Path(ds.test_manifest))
    if sha256(train)!=frozen['train_manifest_sha256'] or sha256(validation)!=frozen['validation_manifest_sha256']:
        raise ValueError('Changed historical train/validation split since Task 0')
    l0=Path(src['l0_manifest'])
    if sha256(l0)!=src['file_hashes']['l0_manifest']:
        raise ValueError('L0 historical source differs from original experiment')
    train_rows=load_rows(train); l0_rows=load_rows(l0)
    tr_labels=object_labels(train_rows,id_col=ds.object_id_column,label_col=ds.label_column)
    l0_labels=object_labels(l0_rows,id_col=ds.object_id_column,label_col=ds.label_column)
    va_labels=object_labels(load_rows(validation),id_col=ds.object_id_column,label_col=ds.label_column)
    # Audit TEST object membership only here. Do not even parse its targets
    # during a training/acquisition preflight; TEST labels are opened strictly
    # by sealed_test() after checkpoint selection.
    with test.open(newline='',encoding='utf-8') as tf:
        test_ids={str(r[ds.object_id_column]) for r in csv.DictReader(tf)}
    if (len(tr_labels),len(l0_labels),len(va_labels),len(test_ids))!=(4729,476,2365,2365):
        raise ValueError('Expected train/L0/validation/test object counts 4729/476/2365/2365')
    if set(l0_labels)-set(tr_labels) or set(tr_labels)&set(va_labels) or set(tr_labels)&test_ids or set(va_labels)&test_ids:
        raise ValueError('Object split leakage or L0 membership mismatch')
    def triplets(rows):
        return Counter((r[ds.object_id_column],int(r['alert']),label_index(r[ds.label_column])) for r in rows)
    if triplets(l0_rows)!=triplets(all_train_rows_for_ids(train_rows,set(l0_labels),ds.object_id_column)):
        raise ValueError('L0 must contain exactly the original full TRAIN stage rows for its 476 objects')
    if any(tr_labels[oid]!=y for oid,y in l0_labels.items()):
        raise ValueError('L0 object ground-truth differs from historical train split')
    pool=stage_rows_no_labels(train_rows,ds.object_id_column,set(l0_labels))
    if len(pool)!=4253 or len({r['object_id'] for r in pool})!=4253:
        raise ValueError('Historical unlabeled acquisition pool is not 4253 objects')
    if set(x['object_id'] for x in pool)!=set(tr_labels)-set(l0_labels):
        raise ValueError('Candidate objects differ from train minus L0')
    # Cached arrays explicitly have NO labels, and cannot include TEST.
    train_marker=json.loads((cache/'fast_cache_manifest.json').read_text())
    if train_marker.get('labels_saved_in_cache') is not False or train_marker.get('test_split_accessed') is not False:
        raise ValueError('Unsafe train cache includes labels or test data')
    if train_marker['manifest_sha256']!={'train':sha256(train),'validation':sha256(validation)}:
        raise ValueError('Train/validation cache is not the audited full historical split')
    if train_marker['unique_stages']!=43309:
        raise ValueError('Incomplete resident train/validation cache')
    specs={init:asdict(NewImageModelSpec(**{
        **frozen['model_spec'],'image_initialization':init})) for init in ('randomized','pretrained')}
    plan={
        'version':1,'experiment':'task1_btsbot_historical_closed_loop_v1',
        'task0_plan':str((task0/'plan.json').resolve()),'task0_plan_sha256':sha256(task0/'plan.json'),
        'l0_manifest':str(l0.resolve()),'l0_sha256':sha256(l0),
        'full_train_manifest':str(train.resolve()),'full_train_sha256':sha256(train),
        'validation_manifest':str(validation.resolve()),'validation_sha256':sha256(validation),
        'test_manifest':str(test.resolve()),'test_sha256':sha256(test),
        'test_cache':str(test_cache.resolve()),
        'train_cache':str(cache.resolve()),'train_cache_manifest_sha256':sha256(cache/'fast_cache_manifest.json'),
        'repo':str(repo.resolve()),'source_sha256':sha_sources(repo),
        'id_col':ds.object_id_column,'label_col':ds.label_column,
        'l0_ids':sorted(l0_labels),'n_l0':476,'n_pool':4253,
        'pool_stage_content_sha256':hashlib.sha256(json.dumps(pool,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'training_objects':4729,'validation_objects':2365,'test_objects':2365,
        'n_cycles':CYCLES,'acquisition_size':ACQUISITION_SIZE,
        'seeds':list(SEEDS),'configurations':CONFIGS,
        'policies':list(POLICIES),'phase1_policies':list(PHASE1),'phase2_policies':list(PHASE2),
        'model_specs':specs,
        'training_recipe':asdict(__import__('al_pipeline.astronomy.fusion_image.task1_training',fromlist=['frozen_recipe']).frozen_recipe(135)),
        'checkpoint_selection':'minimum_weighted_validation_loss',
        'training_optimizer_reset_every_cycle':True,
        'test_inference':'automatic_report_only_after_checkpoint_selection',
        'test_data_available_to_acquisition':False,
        'test_data_available_to_training':False,
        'test_cache_required_at_execution':True,
        'acquisition_unit':'one_object_latest_available_train_alert_stage',
        'test_embargo':'not_in_default_reports; phase1 report release requires all trajectories',
        'advanced_policy_config':{'entropy_diversity_shortlist_multiplier':3.0,
                                  'entropy_balanced_explore_uniform_fraction':0.2},
    }
    return plan,train_rows,pool


def verify_plan(root:Path,*,require_test_cache:bool=False)->dict:
    """Check immutable bytes without opening candidate or TEST label columns.

    Full label/split audits occur at the single prepare step. GPU acquisition
    jobs re-check the frozen file hashes, NOT all historical candidate targets.
    """
    saved=json.loads((root/'plan.json').read_text())
    if saved.get('experiment')!='task1_btsbot_historical_closed_loop_v1':
        raise ValueError('Unexpected Task 1 experiment plan')
    if sha_sources(Path(saved['repo']))!=saved['source_sha256']:
        raise ValueError('Frozen Task 1 scientific source changed')
    inputs=(('task0_plan','task0_plan_sha256'),('l0_manifest','l0_sha256'),
            ('full_train_manifest','full_train_sha256'),
            ('validation_manifest','validation_sha256'),('test_manifest','test_sha256'))
    for filename,checksum in inputs:
        if sha256(Path(saved[filename]))!=saved[checksum]:
            raise ValueError(f'Frozen Task 1 input changed: {filename}')
    cache=Path(saved['train_cache'])
    if sha256(cache/'fast_cache_manifest.json')!=saved['train_cache_manifest_sha256']:
        raise ValueError('Frozen resident training cache changed')
    m=json.loads((cache/'fast_cache_manifest.json').read_text())
    if m.get('labels_saved_in_cache') is not False or m.get('test_split_accessed') is not False:
        raise ValueError('Unsafe resident training cache')
    pool=load_rows(root/'pool_stages.csv')
    if len(pool)!=4253 or len({r['object_id'] for r in pool})!=4253 or any(set(r)!={'object_id','alert'} for r in pool):
        raise ValueError('Changed or label-bearing AL pool stage index')
    stage_hash=hashlib.sha256(json.dumps(pool,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if stage_hash != saved['pool_stage_content_sha256']:
        raise ValueError('Frozen pool latest-stage manifest changed')
    if set(saved['l0_ids']) & {r['object_id'] for r in pool}:
        raise ValueError('L0/pool overlap in frozen plan')
    if require_test_cache:
        testcache=Path(saved['test_cache'])
        if not (testcache/'fast_cache_manifest.json').is_file():
            raise FileNotFoundError('Test cache not consolidated; finish current CPU jobs first')
        t=json.loads((testcache/'fast_cache_manifest.json').read_text())
        if t['test_manifest_sha256']!=saved['test_sha256'] or t.get('labels_saved_in_cache') is not False:
            raise ValueError('Test cache mismatches frozen historical test split')
        if t['train_manifest_sha256']!=saved['full_train_sha256'] or t['validation_manifest_sha256']!=saved['validation_sha256']:
            raise ValueError('Test cache train/validation separation differs from Task 1')
        if t['preprocessing_identity']!=m['preprocessing_identity']:
            raise ValueError('TEST cache preprocessing differs from TRAIN cache')
        if t['test_objects']!=2365:
            raise ValueError('Wrong TEST object population')
    return saved


def prepare(root:Path):
    plan,_,pool=prepare_plan()
    f=root/'plan.json'
    if f.is_file():
        if json.loads(f.read_text())!=plan:
            raise ValueError('Task 1 plan exists with different identity')
        write_csv(root/'pool_stages.csv',pool,['object_id','alert'])
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Nonempty Task 1 root without frozen plan')
        root.mkdir(parents=True,exist_ok=True)
        write_csv(root/'pool_stages.csv',pool,['object_id','alert'])
        write_json_atomic(f,plan)
    print(json.dumps({'prepared':str(f),'l0_objects':476,'pool_objects':4253,
        'train_cache_reused':plan['train_cache'],
        'test_cache_required_only_at_execution':True,
        'phase1_trajectories':18,'phase2_trajectories':18},indent=2))


def l0_dir(root:Path,init:str,seed:int)->Path:
    if init not in ('randomized','pretrained') or seed not in SEEDS:
        raise ValueError('Unknown L0 run')
    return root/'l0'/init/f'seed_{seed}'


def cycle_dir(root:Path,config:str,policy:str,seed:int,cycle:int)->Path:
    if config not in CONFIGS or policy not in POLICIES or seed not in SEEDS or not 1<=cycle<=CYCLES:
        raise ValueError('Unknown Task 1 case')
    return root/'trajectories'/config/policy/f'seed_{seed}'/f'cycle_{cycle:03d}'


def load_complete(path:Path,plan:dict,expected_cycle:int):
    f=path/'complete.json'
    if not f.is_file():
        raise FileNotFoundError(f'Parent not fully trained and evaluated: {f}')
    d=json.loads(f.read_text())
    if d.get('plan_sha256')!=sha256(Path(plan['task1_plan_path'])) or d.get('cycle')!=expected_cycle:
        raise ValueError('Wrong parent trajectory completion record')
    if sha256(path/'fit'/'best_vit_tower.pth')!=d['checkpoint_sha256']:
        raise ValueError('Parent selected checkpoint changed')
    manifest=path/'train_manifest.csv'
    if sha256(manifest)!=d['train_manifest_sha256']:
        raise ValueError('Parent cumulative labeled manifest changed')
    rows=load_rows(manifest)
    if set(r[plan['id_col']] for r in rows)!=set(d['labeled_ids']):
        raise ValueError('Parent labeled IDs do not match its original train manifest')
    if not (path/'sealed_test'/'evaluation_summary.json').is_file():
        raise ValueError('Parent TEST evaluation is not completed')
    return d


def resident_arrays(root:Path)->tuple[np.ndarray,np.ndarray]:
    """One buffered read per cycle, shared by acquisition and training."""
    from al_pipeline.astronomy.fusion_image.fast_cache import SOURCE
    m=json.loads((root/'fast_cache_manifest.json').read_text())
    if m['source']!=SOURCE or m.get('labels_saved_in_cache') is not False or m.get('test_split_accessed') is not False:
        raise ValueError('Unsafe AL resident data cache')
    im=np.load(root/'images.npy',allow_pickle=False)
    md=np.load(root/'metadata.npy',allow_pickle=False)
    if (im.shape!=(43309,3,63,63) or md.shape!=(43309,55) or
        im.dtype!=np.float32 or md.dtype!=np.float32 or isinstance(im,np.memmap)):
        raise ValueError('Unexpected resident array shape/dtype')
    im.setflags(write=False);md.setflags(write=False)
    return im,md


def execute(*,root:Path,initialization:str,seed:int,config:str|None,
            policy:str|None,cycle:int):
    from al_pipeline.astronomy.fusion_image.task1_training import fit,infer_acquisition,sealed_test
    plan=verify_plan(root,require_test_cache=True)
    plan={**plan,'task1_plan_path':str(root/'plan.json')}
    if seed not in SEEDS:raise ValueError('Unknown training seed')
    if cycle==0:
        if initialization not in ('randomized','pretrained') or config is not None or policy is not None:
            raise ValueError('L0 accepts initialization+seed only')
        out=l0_dir(root,initialization,seed)
        labeled=set(plan['l0_ids']);source=None;selected=[];parent_sha=None
    else:
        if config not in CONFIGS or policy not in POLICIES or not 1<=cycle<=CYCLES:
            raise ValueError('Invalid AL trajectory coordinate')
        initialization=CONFIGS[config]['initialization']
        out=cycle_dir(root,config,policy,seed,cycle)
        parent=l0_dir(root,initialization,seed) if cycle==1 else cycle_dir(root,config,policy,seed,cycle-1)
        previous=load_complete(parent,plan,cycle-1)
        source=parent/'fit'/'best_vit_tower.pth'
        parent_sha=previous['checkpoint_sha256']
        labeled=set(previous['labeled_ids'])
        if len(labeled)!=(476 if cycle==1 else 476+ACQUISITION_SIZE*(cycle-1)):
            raise ValueError('Previous cycle labeled-set size incorrect')
    test_manifest=Path(plan['test_manifest']);val=Path(plan['validation_manifest'])
    cache=Path(plan['train_cache'])
    spec=NewImageModelSpec(**plan['model_specs'][initialization])
    if (out/'complete.json').is_file():
        completed=load_complete(out,plan,cycle)
        print(json.dumps({'already_complete':str(out),'labeled_objects':len(completed['labeled_ids'])},indent=2))
        return
    arrays=None
    if cycle>0:
        # Evaluation job never asks oracle for labels until the selected IDs
        # have been persisted. Candidate rows strictly have TWO fields.
        st=load_rows(root/'pool_stages.csv')
        rows=[r for r in st if r['object_id'] not in labeled]
        if len(rows)!=4253-ACQUISITION_SIZE*(cycle-1):
            raise ValueError('Unlabeled pool size out of sync with cycle count')
        k=min(ACQUISITION_SIZE,len(rows))
        selection_file=out/'acquisition.json'
        if selection_file.is_file():
            sel=json.loads(selection_file.read_text())
            if sel.get('parent_checkpoint_sha256')!=parent_sha or sel.get('policy')!=policy or sel.get('candidate_count')!=len(rows):
                raise ValueError('Acquisition persisted with incompatible parent identity')
        else:
            arrays=resident_arrays(cache)
            ids,logits,emb=infer_acquisition(checkpoint=source,spec=spec,seed=seed,
                stage_rows=rows,cache_root=cache,resident_arrays=arrays)
            sel=choose(policy=policy,ids=ids,logits=logits,embeddings=emb,
                       k=k,seed=seed,cycle=cycle)
            sel.update({'parent_checkpoint_sha256':parent_sha,
                        'candidate_ids_sha256':__import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest(),
                        'label_information_available_to_selection':False})
            write_json_atomic(selection_file,sel)
        selected=list(sel['selected_ids'])
        if len(selected)!=k or len(set(selected))!=k or set(selected)&labeled or set(selected)-{r['object_id'] for r in rows}:
            raise ValueError('Persisted selection contains invalid object IDs')
        labeled.update(selected) # Oracle reveal happens only AFTER selection persisted.
    # Historical oracle is opened ONLY after acquisition.json has been written.
    train=load_rows(plan['full_train_manifest'])
    train_rows=all_train_rows_for_ids(train,labeled,plan['id_col'])
    if len(labeled)!=(476+ACQUISITION_SIZE*cycle if cycle<9 else 4729):
        raise ValueError('Cumulative AL training size mismatch')
    manifest=out/'train_manifest.csv'
    write_csv(manifest,train_rows,list(train[0]))
    if cycle==0:
        initial_checkpoint=None
    elif CONFIGS[config]['continuation']=='warm_start':
        initial_checkpoint=source
    else:
        initial_checkpoint=l0_dir(root,initialization,seed)/'fit'/'best_vit_tower.pth'
    initial_sha=sha256(initial_checkpoint) if initial_checkpoint else None
    identity={'task1_plan_sha256':sha256(root/'plan.json'),
        'initialization':initialization,'seed':seed,'configuration':config,
        'policy':policy,'cycle':cycle,'parent_checkpoint_sha256':parent_sha,
        'start_checkpoint_sha256':initial_sha,'train_manifest_sha256':sha256(manifest),
        'checkpoint_selection':'minimum_weighted_validation_loss'}
    result=fit(spec=spec,seed=seed,train_manifest=manifest,
        validation_manifest=val,cache_root=cache,id_col=plan['id_col'],label_col=plan['label_col'],
        output_dir=out/'fit',identity=identity,initial_checkpoint=initial_checkpoint,
        resident_arrays=arrays)
    test=sealed_test(checkpoint=out/'fit'/'best_vit_tower.pth',spec=spec,seed=seed,
        test_manifest=test_manifest,test_cache=Path(plan['test_cache']),
        id_col=plan['id_col'],label_col=plan['label_col'],
        output_dir=out/'sealed_test',expected_test_sha=plan['test_sha256'])
    resolved_labels=object_labels(train_rows,id_col=plan['id_col'],label_col=plan['label_col'])
    # Only revealed classes for the newly selected objects; NEVER candidate labels.
    selected_classes=Counter(CLASS_NAMES[resolved_labels[oid]] for oid in selected)
    completed={'status':'completed','cycle':cycle,'plan_sha256':sha256(root/'plan.json'),
        'initialization':initialization,'seed':seed,'configuration':config,'policy':policy,
        'labeled_ids':sorted(labeled),'selected_ids':selected,
        'selected_revealed_class_counts':dict(selected_classes),
        'train_manifest_sha256':sha256(manifest),
        'checkpoint_sha256':result['primary_checkpoint_sha256'],
        'validation_macro_f1':result['validation_object_metrics']['f1_macro'],
        'test_evaluation_status':'completed_sealed_report_only',
        'test_manifest_sha256':plan['test_sha256'],
        'test_result_sha256':sha256(out/'sealed_test'/'evaluation_summary.json'),
        'parent_checkpoint_sha256':parent_sha}
    write_json_atomic(out/'complete.json',completed)
    print(json.dumps({'completed':str(out),'cycle':cycle,'labeled_objects':len(labeled),
                      'validation_macro_f1':completed['validation_macro_f1'],
                      'test_saved_report_only':True},indent=2),flush=True)


def _all_required_complete(root:Path,plan:dict,policies:tuple[str,...])->bool:
    for ini in ('randomized','pretrained'):
        for seed in SEEDS:
            if not (l0_dir(root,ini,seed)/'complete.json').is_file():return False
    for config in CONFIGS:
        for policy in policies:
            for seed in SEEDS:
                for cycle in range(1,CYCLES+1):
                    if not (cycle_dir(root,config,policy,seed,cycle)/'complete.json').is_file():return False
    return True


def _area(y,x):
    rule=getattr(np,'trapezoid',None)
    if rule is None:rule=np.trapz  # compatible with NumPy < 2.0
    return rule(y,x)


def report(root:Path,include_test:bool=False):
    p=verify_plan(root)
    if include_test and not _all_required_complete(root,p,PHASE1):
        raise ValueError('TEST metrics remain embargoed until ALL phase1 trajectories finish')
    fields=('initialization','configuration','policy','seed','cycle','labeled_objects',
            'labeled_fraction','validation_macro_f1','validation_balanced_accuracy',
            'validation_accuracy','best_epoch','fit_seconds')
    if include_test:
        fields+=('test_macro_f1','test_balanced_accuracy','test_accuracy')
    records=[]
    cases=[(i,None,None,s,0,l0_dir(root,i,s)) for i in ('randomized','pretrained') for s in SEEDS]
    cases += [(CONFIGS[c]['initialization'],c,p,s,cy,cycle_dir(root,c,p,s,cy))
              for c in CONFIGS for p in POLICIES for s in SEEDS for cy in range(1,CYCLES+1)]
    for ini,conf,policy,seed,cycle,out in cases:
        file=out/'complete.json'
        if not file.is_file():continue
        comp=json.loads(file.read_text())
        fit=json.loads((out/'fit'/'training_summary.json').read_text())
        if comp['checkpoint_sha256']!=fit['primary_checkpoint_sha256']:
            raise ValueError('Fitted checkpoint and completion identity mismatch')
        vm=fit['validation_object_metrics']
        row={'initialization':ini,'configuration':conf or 'shared_l0','policy':policy or 'shared_l0',
             'seed':seed,'cycle':cycle,'labeled_objects':len(comp['labeled_ids']),
             'labeled_fraction':len(comp['labeled_ids'])/4729,
             'validation_macro_f1':vm['f1_macro'],
             'validation_balanced_accuracy':vm['balanced_accuracy'],
             'validation_accuracy':vm['accuracy'],
             'best_epoch':fit['best_epoch'],'fit_seconds':fit['fit_seconds']}
        if include_test:
            t=json.loads((out/'sealed_test'/'evaluation_summary.json').read_text())
            if sha256(out/'sealed_test'/'evaluation_summary.json')!=comp['test_result_sha256']:
                raise ValueError('Sealed report-only TEST result changed')
            tm=t['object_metrics']
            row.update(test_macro_f1=tm['f1_macro'],test_balanced_accuracy=tm['balanced_accuracy'],
                       test_accuracy=tm['accuracy'])
        records.append(row)
    outfile=root/('task1_validation_and_test_report.csv' if include_test else 'task1_validation_report.csv')
    # Reports are derived views; unlike frozen training manifests they must
    # be refreshable when more independent cycle jobs finish.
    outfile.parent.mkdir(parents=True,exist_ok=True)
    tmp=outfile.with_suffix(outfile.suffix+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(fields))
        writer.writeheader();writer.writerows(records)
    tmp.replace(outfile)
    # Curves are deterministic derived reports, NEVER used by next-cycle jobs.
    # Include the shared L0 point for each trajectory and normalize AULC by
    # the actual labeled-fraction interval [476/4729, 1].
    by={(r['configuration'],r['policy'],r['seed'],r['cycle']):r for r in records
        if r['cycle']>0}
    l0={(r['initialization'],r['seed']):r for r in records if r['cycle']==0}
    curves=[]
    for conf in CONFIGS:
        ini=CONFIGS[conf]['initialization']
        for policy in POLICIES:
            for seed in SEEDS:
                key=(conf,policy,seed)
                if (ini,seed) not in l0 or any((*key,cy) not in by for cy in range(1,10)):
                    continue
                points=[l0[ini,seed]]+[by[(*key,cy)] for cy in range(1,10)]
                xs=np.asarray([r['labeled_fraction'] for r in points],dtype=float)
                ys=np.asarray([r['validation_macro_f1'] for r in points],dtype=float)
                if not np.all(np.diff(xs)>0) or abs(xs[-1]-1)>1e-9:
                    raise ValueError('Invalid AL labeled-fraction progression')
                aulc=float(_area(ys,xs)/(xs[-1]-xs[0]))
                curve={'configuration':conf,'policy':policy,'seed':seed,
                       'validation_macro_f1_early_cycles_1_to_5':float(np.mean(ys[1:6])),
                       'validation_macro_f1_final_cycle_9':float(ys[-1]),
                       'validation_macro_f1_normalized_AULC':aulc}
                if include_test:
                    yt=np.asarray([r['test_macro_f1'] for r in points],dtype=float)
                    curve.update(test_macro_f1_early_cycles_1_to_5=float(np.mean(yt[1:6])),
                        test_macro_f1_final_cycle_9=float(yt[-1]),
                        test_macro_f1_normalized_AULC=float(_area(yt,xs)/(xs[-1]-xs[0])))
                curves.append(curve)
    curvefile=root/('task1_test_learning_curves.csv' if include_test else 'task1_validation_learning_curves.csv')
    if curves:
        with curvefile.with_suffix(curvefile.suffix+'.tmp').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(curves[0]))
            writer.writeheader();writer.writerows(curves)
        curvefile.with_suffix(curvefile.suffix+'.tmp').replace(curvefile)
    count=len(records)
    print(json.dumps({'report':str(outfile),'curve_report':str(curvefile) if curves else None,
        'complete_curves':len(curves),'complete_case_cycles':count,
        'phase1_complete':_all_required_complete(root,p,PHASE1),
        'test_metrics_released':include_test},indent=2))


def status(root:Path):
    plan=json.loads((root/'plan.json').read_text())
    result={}
    result['l0_complete']=sum((l0_dir(root,ini,s)/'complete.json').is_file()
         for ini in ('randomized','pretrained') for s in SEEDS)
    for phase,policies in (('phase1',PHASE1),('phase2',PHASE2)):
        count=sum((cycle_dir(root,c,p,s,cy)/'complete.json').is_file()
              for c in CONFIGS for p in policies for s in SEEDS for cy in range(1,10))
        result[phase]={'completed_cycle_jobs':count,'total_cycle_jobs':3*len(policies)*3*9}
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','check','run-l0','run-cycle','report','status'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--initialization',choices=('randomized','pretrained'))
    parser.add_argument('--configuration',choices=tuple(CONFIGS))
    parser.add_argument('--policy',choices=POLICIES)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--cycle',type=int)
    parser.add_argument('--include-test',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.root)
    elif args.command=='check':
        info=verify_plan(args.root,require_test_cache=False)
        print(json.dumps({'identity':'passed','test_cache_present':(Path(info['test_cache'])/'fast_cache_manifest.json').exists(),
             'test_cache_may_be_pending':True},indent=2))
    elif args.command=='status':status(args.root)
    elif args.command=='report':report(args.root,include_test=args.include_test)
    elif args.command=='run-l0':
        if args.initialization is None or args.seed is None:parser.error('run-l0 needs --initialization and --seed')
        execute(root=args.root,initialization=args.initialization,seed=args.seed,
                config=None,policy=None,cycle=0)
    else:
        if args.configuration is None or args.policy is None or args.seed is None or args.cycle is None:
            parser.error('run-cycle needs --configuration --policy --seed --cycle')
        execute(root=args.root,initialization=CONFIGS[args.configuration]['initialization'],seed=args.seed,
                config=args.configuration,policy=args.policy,cycle=args.cycle)


if __name__=='__main__':main()
