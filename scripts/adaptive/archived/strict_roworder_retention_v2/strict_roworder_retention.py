#!/usr/bin/env python3
"""Strict single-model row-order retention with one Slurm GPU fit per job.

Task4 is immutable. The original rejected c4 model is attempt 0 (read-only).
For c>=5, attempt 0 is the unshuffled standard fit. Attempts 1..4 are fresh
fits with independent row permutations, fixed training seed/loader seed and
fixed incumbent checkpoint. A cycle is accepted iff BOTH macro-F1 and balanced
accuracy strictly improve relative to the previously accepted checkpoint.

Only the necessary next attempt is submitted, at most one at a time, using
Slurm afterok on the preceding GPU job. Scientific exhaustion ends submission;
unexpected system failures block downstream work. No ensemble is constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HDD = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
PERSISTENT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
TASK4 = 'astrominn_closed_loop_warm_v1'
BASE = 'astrominn_historical_replay_v1'
SUITE = 'astrominn_entropy135_strict_roworder_retention_v2'
ROOT = HDD/'recovery'/SUITE
SEED = 135
FIRST = 4
LAST = 9
MAX_RETRIES = 4
PERMUTATIONS = (41004, 42004, 43004, 44004)
REFERENCE_STATE = PERSISTENT/'runs'/TASK4/'entropy'/f'seed_{SEED}'
REFERENCE_DATA = HDD/'runs'/TASK4/'entropy'/f'seed_{SEED}'
TASK4_CONTROL = PERSISTENT/'closed_loop'/TASK4
L0 = PERSISTENT/'runs'/TASK4/'shared/historical_replay_split'
BASELINE = HDD/'runs'/BASE/'shared'/f'seed_{SEED}'/'bootstrap_training/best_vit_tower.pth'
EXCLUDE = PERSISTENT/'runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n',encoding='utf-8')
    tmp.replace(path)


def read_table(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames:raise ValueError('No CSV header: '+str(path))
        return list(reader.fieldnames),list(reader)


def write_table(path,fields,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def metrics(record):
    return {'macro_f1':float(record['f1_macro']),'balanced_accuracy':float(record['balanced_accuracy'])}


def decision(incumbent, candidate):
    """Both metrics must strictly improve. No threshold, tie, or trade-off."""
    return (candidate['macro_f1']>incumbent['macro_f1'] and
            candidate['balanced_accuracy']>incumbent['balanced_accuracy'])


def ref_history():
    path=REFERENCE_STATE/'evaluation_history.json'
    rows=read(path)
    v={int(row['cycle']):metrics(row['metrics']) for row in rows if row['split']=='validation'}
    if sorted(v)!=list(range(10)):
        raise ValueError('Original validation history must have cycles 0..9 exactly')
    return rows,v


def verify_origin():
    if not (TASK4_CONTROL/'source_snapshot.json').exists():
        raise FileNotFoundError('Original frozen Task4 source snapshot not found')
    manifest=read(TASK4_CONTROL/'source_snapshot.json')
    for name,digest in manifest.items():
        f=TASK4_CONTROL/'source'/name
        if not f.is_file() or sha(f)!=digest:
            raise ValueError('Original Task4 source hash mismatch: '+str(f))
    rows,v=ref_history()
    if any(not decision(v[c-1],v[c]) for c in (1,2,3)):
        raise ValueError('Entropy seed135 cycles 1..3 are not strictly accepted')
    if decision(v[3],v[4]):
        raise ValueError('Original entropy seed135 cycle 4 unexpectedly passes strict rule')
    if len(rows)<10 or not BASELINE.is_file():
        raise ValueError('Original evaluation history or bootstrap missing')
    for c in range(1,5):
        if not (REFERENCE_STATE/f'cycle_{c:03d}'/'cycle_summary.json').is_file():
            raise FileNotFoundError('Source cycle summary missing')
    c3=REFERENCE_DATA/'cycle_003/training/best_vit_tower.pth'
    c4=REFERENCE_DATA/'cycle_004/training/best_vit_tower.pth'
    if not c3.is_file() or not c4.is_file():
        raise FileNotFoundError('Original c3/c4 checkpoints unavailable')
    for c in (3,4):
        p=REFERENCE_DATA/f'cycle_{c:03d}/evaluation/evaluation_summary.json'
        q=read(p)
        if sha(REFERENCE_DATA/f'cycle_{c:03d}/training/best_vit_tower.pth')!=q['checkpoint_sha256']:
            raise ValueError('Source checkpoint/evaluation mismatch at cycle '+str(c))
        m=q['splits']
        record=next((x for x in m if x['split']=='validation'),None)
        if record is None or any(abs(record['object_metrics'][k]-v[c][k if k!='macro_f1' else 'macro_f1'])>1e-10
                                 for k in ('balanced_accuracy',) ):
            raise ValueError('Source evaluation object-level metrics mismatch')
        if abs(record['object_metrics']['f1_macro']-v[c]['macro_f1'])>1e-10:
            raise ValueError('Source F1 history inconsistent')
    c4sum=read(REFERENCE_STATE/'cycle_004/cycle_summary.json')
    if Path(c4sum['training_initial_checkpoint']) != c3:
        raise ValueError('Original c4 did not warm-start from c3')
    if Path(c4sum['trained_checkpoint']) != c4:
        raise ValueError('Original c4 candidate mismatch')
    if len(c4sum['selected_ids'])!=473 or len(set(c4sum['selected_ids']))!=473:
        raise ValueError('Wrong source cycle4 acquisition batch')
    if not EXCLUDE.is_file():raise FileNotFoundError(EXCLUDE)
    return v


def reroute_registry(source: str) -> str:
    import tomllib
    cfg=tomllib.loads(source)
    match=re.search(r'(?ms)^\[storage\]\s*\n(.*?)(?=^\[|\Z)',source)
    if not match: raise ValueError('Missing [storage]')
    section=match.group(0)
    for key in ('persistent_root','data_root'):
        section,n=re.subn(rf'(?m)^\s*{key}\s*=.*$',f'{key} = "{ROOT}"',section)
        if n!=1:raise ValueError('Missing '+key)
    result=source[:match.start()]+section+source[match.end():]
    modified=tomllib.loads(result)
    if {k:v for k,v in cfg.items() if k!='storage'} != {k:v for k,v in modified.items() if k!='storage'}:
        raise ValueError('Registry changed something beyond storage')
    return result


def run_dir():return ROOT/'runs'/SUITE/'entropy'/f'seed_{SEED}'


def copy_once(src,dest):
    src=Path(src);dest=Path(dest)
    if not src.is_file():raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if sha(src)!=sha(dest):raise ValueError('Staged artifact mismatch: '+str(dest))
    else:
        shutil.copy2(src,dest)
        if sha(src)!=sha(dest):raise ValueError('Copy integrity failure: '+str(dest))


def symlink_once(src,dst):
    src=Path(src);dst=Path(dst)
    if not src.is_file():raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink():
        if dst.resolve()!=src.resolve():raise ValueError('Existing link points to different artifact: '+str(dst))
    elif dst.exists():
        if sha(src)!=sha(dst):raise ValueError('Existing canonical checkpoint differs')
    else:
        dst.symlink_to(src)


def freeze():
    v=verify_origin()
    if (ROOT/'plan.json').is_file():return verify()
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError('Partial nonempty experiment root; inspect manually: '+str(ROOT))
    ROOT.mkdir(parents=True)
    original=TASK4_CONTROL/'source/src'
    shutil.copytree(original,ROOT/'source/src')
    # Freeze controller and worker alongside the scientific source. The
    # Slurm jobs must never depend on a mutable extracted upload directory.
    (ROOT/'scripts').mkdir(exist_ok=True)
    for name in ('strict_roworder_retention.py','strict_roworder_worker.sbatch'):
        copy_once(Path(__file__).resolve().with_name(name), ROOT/'scripts'/name)
    # The original Task4 source can evaluate validation and test. For this
    # gating experiment, patch the COPY to evaluate validation exclusively.
    astro=ROOT/'source/src/al_pipeline/astronomy/experiments/astrominn.py'
    content=astro.read_text(encoding='utf-8')
    old="""    evaluation_manifests = {
        "validation": dataset.validation_manifest,
        "test": dataset.test_manifest,
    }"""
    new="""    evaluation_manifests = {
        "validation": dataset.validation_manifest,
    }"""
    if content.count(old)!=1:
        raise ValueError('Unexpected frozen Task4 evaluator; cannot enforce validation-only')
    astro.write_text(content.replace(old,new,1),encoding='utf-8')
    shutil.copy2(TASK4_CONTROL/'registry.toml',ROOT/'registry_original.toml')
    (ROOT/'registry.toml').write_text(reroute_registry((TASK4_CONTROL/'registry.toml').read_text(encoding='utf-8')),encoding='utf-8')
    codehash={str(p.relative_to(ROOT/'source')):sha(p) for p in sorted((ROOT/'source').rglob('*.py'))}
    basepolicy=read(REFERENCE_DATA/'cycle_004/training/training_summary.json')['policy']
    if (float(basepolicy['class_weight_exponent'])!=.85 or basepolicy['loader_seed']!=SEED or
        basepolicy['max_epochs']!=50 or basepolicy['patience']!=10 or
        basepolicy['new_object_weighting']!='per_sample' or basepolicy['overwrite'] is not False):
        raise ValueError('Unexpected original training policy')
    checkpoints={c:sha(REFERENCE_DATA/f'cycle_{c:03d}/training/best_vit_tower.pth') for c in (3,4)}
    plan={'schema':'strict_roworder_retention_v2','source_suite':TASK4,'seed':SEED,
          'first_cycle':FIRST,'last_cycle':LAST,'max_retries':MAX_RETRIES,
          'permutations':list(PERMUTATIONS),'source_verified_predecessor':3,
          'incumbent_metrics_at_c3':v[3],'original_c4_metrics':v[4],
          'original_c3_c4_checkpoint_sha256':checkpoints,
          'source_snapshot_sha256':sha(TASK4_CONTROL/'source_snapshot.json'),
          'source_code_sha256':codehash,
          'controller_sha256':{name:sha(ROOT/'scripts'/name) for name in ('strict_roworder_retention.py','strict_roworder_worker.sbatch')},
          'registry_sha256':sha(ROOT/'registry.toml'),
          'training_policy':basepolicy,
          'acceptance':'candidate_macro_f1>incumbent_macro_f1 AND candidate_balanced_accuracy>incumbent_balanced_accuracy',
          'candidate_at_c4':'original_saved_rejected_model; do not retrain attempt zero',
          'training_seeds':'identical seed 135 and loader_seed 135 for all retries',
          'data':'hold every acquisition batch fixed during retries; never ensemble; no test evaluation',
          'storage':'new run state, artifacts and logs all under HDD; Task4 read-only',
          'scheduling':'one fit per GPU job, max 20 minutes; automatic afterok one-job-at-a-time dispatch',
          'max_new_fits':4+(LAST-FIRST)*5}
    # Stage source progress only through the three accepted cycles. This run is
    # scientifically a branch from its c3 checkpoint, not a clone of c4-c9.
    target=run_dir()
    for c in range(4):
        copy_once(REFERENCE_STATE/'manifests'/f'train_cycle_{c:03d}.csv',
                  target/'manifests'/f'train_cycle_{c:03d}.csv')
    for c in range(1,4):
        copy_once(REFERENCE_STATE/'manifests'/f'acquired_cycle_{c:03d}.csv',
                  target/'manifests'/f'acquired_cycle_{c:03d}.csv')
    fields,ledger=read_table(REFERENCE_STATE/'acquisitions.csv')
    if set(int(x['cycle']) for x in ledger)!={1,2,3,4,5,6,7,8,9}:
        raise ValueError('Original acquisition ledger unexpected')
    retained=[x for x in ledger if int(x['cycle'])<=3]
    if len(retained)!=1419:raise ValueError('Three prior cycles must acquire exactly 1419 objects')
    write_table(target/'acquisitions.csv',fields,retained)
    # Record the historical accepted validation metrics for plots/provenance.
    hist,_=ref_history()
    history=[dict(row,run_id=f'{SUITE}_entropy_seed{SEED}') for row in hist
             if int(row['cycle'])<=3 and row['split']=='validation']
    write(target/'evaluation_history.json',history)
    write(target/'run_state.json',{'last_completed_cycle':3,'active_cycle':None,'status':'ready'})
    for c in range(1,4):
        copy_once(REFERENCE_STATE/f'cycle_{c:03d}/selection.json',target/f'cycle_{c:03d}/selection.json')
        copy_once(REFERENCE_STATE/f'cycle_{c:03d}/revealed_labels.json',target/f'cycle_{c:03d}/revealed_labels.json')
    # Symlink the immutable previous checkpoint: no retraining of c1-c3.
    symlink_once(REFERENCE_DATA/'cycle_003/training/best_vit_tower.pth',
                 target/'cycle_003/training/best_vit_tower.pth')
    # Reuse the exact original cycle4 acquisition; no selection drift.
    for name in ('selection.json','revealed_labels.json'):
        copy_once(REFERENCE_STATE/'cycle_004'/name,target/'cycle_004'/name)
    # We can reuse verified inference from the same previous checkpoint.
    oldinfer=REFERENCE_DATA/'cycle_004/inference.npz'
    if oldinfer.is_file():symlink_once(oldinfer,target/'cycle_004/inference.npz')
    (ROOT/'logs').mkdir(exist_ok=True)
    write(ROOT/'plan.json',plan)
    return verify()


def verify():
    plan=read(ROOT/'plan.json')
    if (plan.get('schema')!='strict_roworder_retention_v2' or plan['seed']!=SEED or
        plan['first_cycle']!=FIRST or plan['last_cycle']!=LAST or
        plan['permutations']!=list(PERMUTATIONS)):
        raise ValueError('Plan identity differs')
    if sha(ROOT/'registry.toml')!=plan['registry_sha256']:
        raise ValueError('Routed registry modified')
    for name,d in plan['controller_sha256'].items():
        if sha(ROOT/'scripts'/name)!=d:raise ValueError('Frozen controller/worker modified: '+name)
    for name,d in plan['source_code_sha256'].items():
        if sha(ROOT/'source'/name)!=d:raise ValueError('Frozen source modified: '+name)
    if sha(TASK4_CONTROL/'source_snapshot.json')!=plan['source_snapshot_sha256']:
        raise ValueError('Original Task4 source snapshot changed')
    v=verify_origin()
    if v[3]!=plan['incumbent_metrics_at_c3'] or v[4]!=plan['original_c4_metrics']:
        raise ValueError('Original evaluation changed')
    for c in (3,4):
        if sha(REFERENCE_DATA/f'cycle_{c:03d}/training/best_vit_tower.pth') != plan['original_c3_c4_checkpoint_sha256'][str(c)]:
            raise ValueError('Original reference checkpoint changed')
    target=run_dir()
    if read(target/'run_state.json')['last_completed_cycle']<3:
        raise ValueError('Missing seeded accepted progress')
    for c in range(4):
        if sha(target/'manifests'/f'train_cycle_{c:03d}.csv')!=sha(REFERENCE_STATE/'manifests'/f'train_cycle_{c:03d}.csv'):
            raise ValueError(f'Copied accepted manifest changed: cycle{c}')
    for c in range(1,4):
        if sha(target/'manifests'/f'acquired_cycle_{c:03d}.csv')!=sha(REFERENCE_STATE/'manifests'/f'acquired_cycle_{c:03d}.csv'):
            raise ValueError(f'Copied acquired manifest changed: cycle{c}')
    for f in ('selection.json','revealed_labels.json'):
        if sha(target/'cycle_004'/f)!=sha(REFERENCE_STATE/'cycle_004'/f):
            raise ValueError('Frozen original c4 acquisition changed: '+f)
    return plan


def shuffle_manifest(src,dst,seed):
    import numpy as np
    cols,rows=read_table(src)
    original=Counter(tuple(r[x] for x in cols) for r in rows)
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(rows))
    if np.array_equal(perm,np.arange(len(rows))):raise ValueError('Identity permutation')
    shuffled=[rows[int(i)] for i in perm]
    if Counter(tuple(r[x] for x in cols) for r in shuffled)!=original:
        raise ValueError('Row ordering changed sample multiset')
    write_table(dst,cols,shuffled)
    return sha(dst)


def accepted_baseline(cycle):
    if cycle==FIRST:
        return read(ROOT/'plan.json')['incumbent_metrics_at_c3']
    previous=run_dir()/'roworder_decisions'/f'cycle_{cycle-1:03d}.json'
    d=read(previous)
    if not d['accepted'] or d['cycle']!=cycle-1:
        raise RuntimeError('Previous cycle has no accepted model')
    return d['accepted_metrics']


def new_experiment(device="cuda"):
    from al_pipeline.astronomy.experiments.astrominn import build_astrominn_simulated_suite
    from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainingConfig
    training_policy=AstroMiNNViTTrainingConfig(**read(ROOT/'plan.json')['training_policy'])
    suite=build_astrominn_simulated_suite(suite_id=SUITE,registry_path=ROOT/'registry.toml',
        pool_path=L0/'candidate_pool',hidden_labels_path=L0/'hidden_labels.csv',
        training_policy=training_policy,strategies=('entropy',),acquisition_size=473,
        max_cycles=9,allow_partial_final_cycle=True,seed=SEED,training_initialization='warm_start',
        device=device,inference_num_workers=0,evaluation_num_workers=0,
        baseline_checkpoint=BASELINE,training_manifest_override=L0/'initial_labeled_manifest.csv')
    e=suite.experiments['entropy']
    if e.run_dir!=run_dir() or e.artifact_dir!=run_dir():
        raise ValueError('Experiment attempted to write outside its HDD run directory')
    if e.evaluator.manifests.keys()!={'validation'}:
        raise ValueError('This experiment must never evaluate test set')
    return e


def attempt_request(base,cycle,index,manifest):
    from al_pipeline.core.training.base import TrainingRequest
    return TrainingRequest(train_manifest=manifest,validation_manifest=base.validation_manifest,
        output_dir=run_dir()/f'cycle_{cycle:03d}'/'attempts'/f'attempt_{index:02d}'/'training',
        run_id=f'{SUITE}_entropy_seed{SEED}_c{cycle}_a{index}',
        cycle=cycle,strategy='entropy',seed=SEED,initial_checkpoint=base.initial_checkpoint)


def link_accepted(cycle,completed,evaluation_dir):
    canonical=run_dir()/f'cycle_{cycle:03d}'
    symlink_once(completed.checkpoint_path,canonical/'training/best_vit_tower.pth')
    dest=canonical/'evaluation'
    if dest.is_symlink():
        if dest.resolve()!=evaluation_dir.resolve():
            raise ValueError('Canonical evaluation symlink differs')
    elif dest.exists():
        raise RuntimeError('Unexpected canonical evaluation directory; refusing overwrite')
    else:
        dest.symlink_to(evaluation_dir,target_is_directory=True)


class RetryPending(Exception):
    """One valid training attempt failed the strict two-metric gate."""


def attempt_marker(cycle, index):
    return run_dir()/f'cycle_{cycle:03d}'/'attempts'/f'attempt_{index:02d}'/'attempt_result.json'


def outcome_path(cycle, index):
    return ROOT/'outcomes'/f'cycle_{cycle:03d}_attempt_{index:02d}.json'


def attempts_for(cycle):
    start = 1 if cycle==FIRST else 0
    out=[]
    for index in range(0,MAX_RETRIES+1):
        if index==0 and cycle==FIRST:
            # This read-only candidate is registered during first retry.
            pass
        marker=attempt_marker(cycle,index)
        if not marker.is_file():
            if index<start:raise RuntimeError('Original rejected candidate not recorded')
            break
        row=read(marker)
        if row['attempt']!=index or row['cycle']!=cycle:
            raise ValueError('Attempt provenance inconsistent')
        out.append(row)
    return out


def record_original_rejection(cycle,input_manifest,incumbent):
    """Materialize provenance, not a training job, for original rejected c4."""
    assert cycle==FIRST
    expected=REFERENCE_STATE/'manifests/train_cycle_004.csv'
    if sha(input_manifest)!=sha(expected):
        raise ValueError('Cycle4 acquired manifest differs from original rejected cycle')
    marker=attempt_marker(cycle,0)
    orig=REFERENCE_DATA/'cycle_004'
    raw=read(ROOT/'plan.json')['original_c4_metrics']
    row={'cycle':cycle,'attempt':0,'kind':'reused_original_rejected',
        'manifest_sha256':sha(input_manifest),'manifest_path':str(input_manifest),
        'permutation_seed':None,'training_seed':SEED,'loader_seed':SEED,
        'checkpoint_path':str(orig/'training/best_vit_tower.pth'),
        'checkpoint_sha256':sha(orig/'training/best_vit_tower.pth'),
        'evaluation_summary':str(orig/'evaluation/evaluation_summary.json'),
        'incumbent_metrics':incumbent,'candidate_metrics':raw,
        'delta_macro_f1':raw['macro_f1']-incumbent['macro_f1'],
        'delta_balanced_accuracy':raw['balanced_accuracy']-incumbent['balanced_accuracy'],
        'accepted':decision(incumbent,raw)}
    if row['accepted']:
        raise RuntimeError('Saved original rejected candidate unexpectedly passes strict gate')
    if marker.is_file():
        if read(marker)!=row:raise ValueError('Previously recorded original candidate changed')
    else:
        write(marker,row)


def recover_attempt(exp,request,incumbent,cycle,index):
    """Train/evaluate at most ONE model; reuse exact completed artifacts on retry."""
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refuse CPU fallback')
    if cycle not in range(FIRST,LAST+1):raise ValueError('Invalid cycle')
    if index not in range(1 if cycle==FIRST else 0,MAX_RETRIES+1):
        raise ValueError('Invalid attempt for this cycle')
    cycle_dir=run_dir()/f'cycle_{cycle:03d}'
    final=run_dir()/'roworder_decisions'/f'cycle_{cycle:03d}.json'
    if final.exists():
        recorded=read(final)
        if not recorded.get('accepted') or recorded.get('accepted_attempt')!=index:
            raise RuntimeError('Cycle already has a terminal acceptance/exhaustion decision')
        # Crash-safe replay: the accepted result might be durable while the
        # ordinary AL coordinator still has to commit cycle_summary/run_state.
        req=attempt_request(request,cycle,index,Path(recorded['accepted_manifest']))
        done=exp.trainer.completed_result(req)
        if done is None or sha(done.checkpoint_path)!=recorded['accepted_checkpoint_sha256']:
            raise ValueError('Accepted trained checkpoint unavailable on resume')
        evaldir=cycle_dir/'attempts'/f'attempt_{index:02d}'/'evaluation'
        evaluated=exp.evaluator.completed_result(checkpoint_path=done.checkpoint_path,output_dir=evaldir)
        if evaluated is None or metrics(evaluated.split('validation').object_metrics)!=recorded['accepted_metrics']:
            raise ValueError('Accepted validation metrics not reproducible on resume')
        link_accepted(cycle,done,evaldir)
        return done
    input_manifest=request.train_manifest
    if cycle==FIRST:
        record_original_rejection(cycle,input_manifest,incumbent)
    for prior in range(index):
        p=attempt_marker(cycle,prior)
        if not p.exists():raise ValueError(f'Missing previous attempt {prior}; refusing out-of-order retraining')
        item=read(p)
        if item['cycle']!=cycle or item['attempt']!=prior or item['incumbent_metrics']!=incumbent or item['accepted']:
            raise ValueError('Previous attempt failed provenance/rejection check')
    if attempt_marker(cycle,index+1).exists():
        raise ValueError('A later attempt already exists; refusing backward replay')
    folder=cycle_dir/'attempts'/f'attempt_{index:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    marker=attempt_marker(cycle,index)
    manifest=folder/'train_manifest.csv'
    if index==0:
        copy_once(input_manifest,manifest)
        order_seed=None
    else:
        order_seed=PERMUTATIONS[index-1]+cycle*100
        ref=folder/'expected_order.json'
        if manifest.exists():
            if not ref.exists() or read(ref).get('permutation_seed')!=order_seed:
                raise ValueError('Existing manifest lacks permutation provenance')
            test=folder/'check_manifest_tmp.csv'
            try:
                if shuffle_manifest(input_manifest,test,order_seed)!=sha(manifest):
                    raise ValueError('Persisted permutation does not match deterministic replay')
            finally:test.unlink(missing_ok=True)
            if read(ref).get('manifest_sha256')!=sha(manifest):
                raise ValueError('Existing permutation checksum changed')
        else:
            shuffle_manifest(input_manifest,manifest,order_seed)
            write(ref,{'permutation_seed':order_seed,'manifest_sha256':sha(manifest)})
    req=attempt_request(request,cycle,index,manifest)
    finished=exp.trainer.completed_result(req)
    if finished is None:finished=exp.trainer.train(req)
    evaldir=folder/'evaluation'
    evaluated=exp.evaluator.completed_result(checkpoint_path=finished.checkpoint_path,output_dir=evaldir)
    if evaluated is None:
        evaluated=exp.evaluator.evaluate(checkpoint_path=finished.checkpoint_path,output_dir=evaldir)
    score=metrics(evaluated.split('validation').object_metrics)
    item={'cycle':cycle,'attempt':index,
        'kind':'original_order_fit' if index==0 else 'row_order_retry',
        'manifest_sha256':sha(manifest),'manifest_path':str(manifest),
        'permutation_seed':order_seed,'training_seed':SEED,'loader_seed':SEED,
        'checkpoint_path':str(finished.checkpoint_path),
        'checkpoint_sha256':sha(finished.checkpoint_path),
        'evaluation_summary':str(evaluated.summary_path),
        'incumbent_metrics':incumbent,'candidate_metrics':score,
        'delta_macro_f1':score['macro_f1']-incumbent['macro_f1'],
        'delta_balanced_accuracy':score['balanced_accuracy']-incumbent['balanced_accuracy'],
        'accepted':decision(incumbent,score)}
    if marker.is_file():
        if read(marker)!=item:raise ValueError('Completed attempt result changed on resume')
    else:write(marker,item)
    attempts=attempts_for(cycle)
    if len(attempts)!=(index+1):raise ValueError('Attempt audit is incomplete or has gaps')
    print(f'CYCLE {cycle} attempt {index}: dF1={item["delta_macro_f1"]:+.5f}, dBA={item["delta_balanced_accuracy"]:+.5f}, accepted={item["accepted"]}',flush=True)
    if not item['accepted']:
        if index==MAX_RETRIES:
            write(final,{'cycle':cycle,'accepted':False,'incumbent_metrics':incumbent,
                'attempts':attempts,'stop_reason':'strict_metric_gate_failed_after_four_retries'})
        raise RetryPending()
    link_accepted(cycle,finished,evaldir)
    write(final,{'cycle':cycle,'accepted':True,'accepted_attempt':index,
        'accepted_metrics':score,'accepted_manifest':str(manifest),
        'accepted_checkpoint':str(finished.checkpoint_path),
        'accepted_checkpoint_sha256':sha(finished.checkpoint_path),
        'incumbent_metrics':incumbent,'attempts':attempts,
        'fixed_acquired_objects':True,'only_rows_permuted':True})
    return finished


def job(cycle,attempt):
    """Execute exactly one fresh fit and evaluate its strict acceptance."""
    if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Requires GPU Slurm job')
    if cycle not in range(FIRST,LAST+1):raise ValueError('Invalid cycle')
    if attempt not in range(1 if cycle==FIRST else 0,MAX_RETRIES+1):raise ValueError('Invalid attempt')
    verify()
    source=str(ROOT/'source/src')
    if source not in sys.path:sys.path.insert(0,source)
    e=new_experiment()
    st=e.initialize()
    if st.last_completed_cycle==cycle and st.active_cycle is None:
        final=read(run_dir()/'roworder_decisions'/f'cycle_{cycle:03d}.json')
        if not final['accepted'] or final['accepted_attempt']!=attempt:
            raise ValueError('Completed cycle does not match this attempt')
        out={'cycle':cycle,'attempt':attempt,'verdict':'accepted',
             'job_id':str(os.environ['SLURM_JOB_ID']),'metrics':final['accepted_metrics']}
        existing=outcome_path(cycle,attempt)
        if existing.is_file() and read(existing)!=out:
            raise ValueError('Previously recorded job outcome belongs to a different Slurm job')
        write(existing,out)
        print('Recovered previously committed accepted cycle '+str(cycle),flush=True)
        return out
    if st.last_completed_cycle!=cycle-1 or st.active_cycle not in (None,cycle):
        raise ValueError(f'Out of order: cycle={cycle}, last={st.last_completed_cycle}, active={st.active_cycle}')
    incumbent=accepted_baseline(cycle)
    saved=e._train_or_resume
    e._train_or_resume=lambda request:recover_attempt(e,request,incumbent,cycle,attempt)
    try:
        try:
            result=e.run_next_cycle()
        except RetryPending:
            final=run_dir()/'roworder_decisions'/f'cycle_{cycle:03d}.json'
            exhausted=final.is_file() and not read(final)['accepted']
            verdict='exhausted' if exhausted else 'retry'
            if not exhausted and not attempt_marker(cycle,attempt).is_file():
                raise ValueError('Retry decision without durable attempt result')
            out={'cycle':cycle,'attempt':attempt,'verdict':verdict,
                 'job_id':str(os.environ['SLURM_JOB_ID'])}
            write(outcome_path(cycle,attempt),out)
            print(json.dumps(out,indent=2),flush=True)
            return out
    finally:e._train_or_resume=saved
    if result.cycle!=cycle or e.initialize().last_completed_cycle!=cycle:
        raise RuntimeError('Accepted cycle not committed')
    final=read(run_dir()/'roworder_decisions'/f'cycle_{cycle:03d}.json')
    if not final['accepted'] or final['accepted_attempt']!=attempt:
        raise RuntimeError('Run committed a rejected or mismatched model')
    # No later cycle can train until canonical checkpoint is installed.
    if sha(e._inference_checkpoint(cycle+1))!=final['accepted_checkpoint_sha256']:
        raise ValueError('Canonical accepted checkpoint integrity failure')
    out={'cycle':cycle,'attempt':attempt,'verdict':'accepted',
         'job_id':str(os.environ['SLURM_JOB_ID']),
         'metrics':final['accepted_metrics']}
    write(outcome_path(cycle,attempt),out)
    print(json.dumps(out,indent=2),flush=True)
    return out


def parse_walltime(walltime):
    if not re.fullmatch(r'(?:\d{1,2}):[0-5]\d:[0-5]\d',walltime):
        raise ValueError('Walltime must be HH:MM:SS')
    h,m,s=map(int,walltime.split(':'))
    seconds=h*3600+m*60+s
    if not 1<=seconds<=20*60:
        raise ValueError('GPU walltime must be at most 00:20:00')
    return seconds


def command(cycle,attempt,walltime,*,hold=False,dependency=None):
    parse_walltime(walltime)
    cmd=['sbatch','--parsable','--time',walltime,
         '--job-name',f'RO_s135_c{cycle}_a{attempt}',
         '--output',str(ROOT/'logs/RO_%x_%j.out'),
         '--error',str(ROOT/'logs/RO_%x_%j.err')]
    if hold:cmd.append('--hold')
    if dependency:
        if not str(dependency).isdigit():raise ValueError('Invalid Slurm parent job ID')
        cmd += ['--dependency',f'afterok:{dependency}','--kill-on-invalid-dep=yes']
    cmd += [str(ROOT/'scripts/strict_roworder_worker.sbatch'),str(cycle),str(attempt)]
    return cmd


def next_target(cycle,attempt,verdict):
    if verdict=='retry':
        if attempt>=MAX_RETRIES:raise ValueError('Exhausted attempt cannot request a retry')
        return (cycle,attempt+1)
    if verdict=='accepted':
        return None if cycle==LAST else (cycle+1,0)
    if verdict=='exhausted':
        if attempt!=MAX_RETRIES:raise ValueError('Premature exhaustion marker')
        return None
    raise ValueError('Unrecognized verdict')


def submit(walltime,actually_submit):
    verify()
    parse_walltime(walltime)
    if (ROOT/'submission.json').exists():raise RuntimeError('Already submitted; refuse duplicate')
    gate=ROOT/'preflight_pass.json'
    if not gate.is_file() or read(gate).get('plan_sha256')!=sha(ROOT/'plan.json'):
        raise RuntimeError('Scientific preflight required before submission')
    command0=command(FIRST,1,walltime,hold=True)
    if not actually_submit:
        print(json.dumps({'dry_run':True,'maximum_minutes_per_gpu_job':parse_walltime(walltime)/60,
          'pilot':{'cycle':FIRST,'attempt':1,'command':command0},
          'dynamic_next_job':'exactly one next fit submitted after each completed attempt; afterok on its Slurm job ID',
          'maximum_jobs':4+(LAST-FIRST)*5,
          'exhaustion':'no next job is submitted; experiment stops'},indent=2))
        return
    ROOT.joinpath('logs').mkdir(parents=True,exist_ok=True)
    # No other GPU jobs are submitted until this pilot is deliberately released.
    jobid=subprocess.check_output(command0,text=True).strip().split(';')[0]
    if not jobid.isdigit():raise ValueError('Unexpected sbatch job id '+jobid)
    write(ROOT/'submission.json',{'state':'pilot_held','walltime':walltime,
      'jobs':[{'cycle':FIRST,'attempt':1,'job_id':jobid,'parent':None,'released':False}],
      'max_possible_jobs':4+(LAST-FIRST)*5})
    print(f'Submitted one HELD pilot job {jobid} (cycle4 retry1).')
    print('Release it with release-pilot; all later work automatically follows accepted/rejected decisions.')


def release_pilot():
    verify()
    ledger=read(ROOT/'submission.json')
    if ledger['state']!='pilot_held' or len(ledger['jobs'])!=1:
        raise ValueError('Pilot already released or submission ledger inconsistent')
    first=ledger['jobs'][0]
    subprocess.run(['scontrol','release',first['job_id']],check=True)
    first['released']=True
    ledger['state']='running_dynamic_chain'
    write(ROOT/'submission.json',ledger)
    print(f'Released first GPU retry {first["job_id"]}; further attempts auto-submit only when needed.')


def advance():
    """Called from successful GPU job, after its durable outcome is written."""
    current=os.environ.get('SLURM_JOB_ID')
    if not current or not current.isdigit():raise RuntimeError('advance requires active Slurm job ID')
    verify()
    ledger_path=ROOT/'submission.json'
    ledger=read(ledger_path)
    if ledger['state']!='running_dynamic_chain':
        raise ValueError('Experiment was not released')
    if not ledger['jobs'] or ledger['jobs'][-1]['job_id']!=current:
        raise ValueError('Dispatch mismatch: caller is not the latest scheduled job')
    current_job=ledger['jobs'][-1]
    cycle,attempt=current_job['cycle'],current_job['attempt']
    out=read(outcome_path(cycle,attempt))
    if out['job_id']!=current or out['cycle']!=cycle or out['attempt']!=attempt:
        raise ValueError('Job outcome does not match current Slurm allocation')
    verdict=out['verdict']
    target=next_target(cycle,attempt,verdict)
    if verdict=='accepted':
        d=read(run_dir()/'roworder_decisions'/f'cycle_{cycle:03d}.json')
        if not d['accepted'] or d['accepted_attempt']!=attempt:
            raise ValueError('Accepted verdict without committed accepted checkpoint')
        if read(run_dir()/'run_state.json')['last_completed_cycle']!=cycle:
            raise ValueError('Cycle must be committed before next job can be queued')
    elif verdict=='retry':
        a=read(attempt_marker(cycle,attempt))
        if a['accepted'] or (run_dir()/'roworder_decisions'/f'cycle_{cycle:03d}.json').exists():
            raise ValueError('Retry requested despite terminal acceptance/exhaustion')
    else:
        d=read(run_dir()/'roworder_decisions'/f'cycle_{cycle:03d}.json')
        if d['accepted'] or len(d['attempts'])!=5:
            raise ValueError('Exhaustion decision does not reflect five rejected attempts')
    if target is None:
        ledger['state']='complete' if verdict=='accepted' else 'stopped_exhausted'
        ledger['terminal']={'cycle':cycle,'attempt':attempt,'verdict':verdict,'job_id':current}
        write(ledger_path,ledger)
        print('Experiment terminal state: '+ledger['state'],flush=True)
        return
    nc,na=target
    cmd=command(nc,na,ledger['walltime'],dependency=current)
    new_id=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not new_id.isdigit():raise ValueError('Unexpected sbatch response '+new_id)
    ledger['jobs'].append({'cycle':nc,'attempt':na,'job_id':new_id,'parent':current,'released':True})
    write(ledger_path,ledger)
    print(f'Auto-submitted next attempt: c{nc} a{na}, job {new_id}, afterok:{current}',flush=True)


def status():
    verify()
    rows=[]
    for c in range(FIRST,LAST+1):
        path=run_dir()/'roworder_decisions'/f'cycle_{c:03d}.json'
        d=read(path) if path.exists() else None
        observed=[read(attempt_marker(c,i)) for i in range(MAX_RETRIES+1)
                  if attempt_marker(c,i).is_file()]
        rows.append({'cycle':c,
          'decision':('ACCEPTED' if d['accepted'] else 'EXHAUSTED') if d else
                     ('IN_PROGRESS' if observed else 'PENDING'),
          'observed_attempts':[{'attempt':a['attempt'],'accepted':a['accepted'],
                                'delta_macro_f1':a['delta_macro_f1'],
                                'delta_balanced_accuracy':a['delta_balanced_accuracy']}
                               for a in observed],
          'accepted_attempt':d.get('accepted_attempt') if d else None})
    ledger=read(ROOT/'submission.json') if (ROOT/'submission.json').exists() else None
    print(json.dumps({'suite':SUITE,'state':read(run_dir()/'run_state.json'),
        'cycles':rows,'submission_state':ledger['state'] if ledger else 'not_submitted',
        'submitted_jobs':len(ledger['jobs']) if ledger else 0,
        'latest_job':ledger['jobs'][-1] if ledger else None,
        'terminal':ledger.get('terminal') if ledger else None},indent=2))


def report():
    verify()
    path=ROOT/'report/attempts.csv'
    cols=['cycle','attempt','kind','permutation_seed','macro_f1','balanced_accuracy','delta_macro_f1','delta_balanced_accuracy','accepted','checkpoint_sha256']
    rows=[]
    for cycle in range(FIRST,LAST+1):
        marker=run_dir()/'roworder_decisions'/f'cycle_{cycle:03d}.json'
        if not marker.exists():break
        d=read(marker)
        for a in d['attempts']:
            rows.append({'cycle':cycle,'attempt':a['attempt'],'kind':a['kind'],
                'permutation_seed':a['permutation_seed'] or '',
                'macro_f1':a['candidate_metrics']['macro_f1'],
                'balanced_accuracy':a['candidate_metrics']['balanced_accuracy'],
                'delta_macro_f1':a['delta_macro_f1'],'delta_balanced_accuracy':a['delta_balanced_accuracy'],
                'accepted':a['accepted'],'checkpoint_sha256':a['checkpoint_sha256']})
    write_table(path,cols,rows)
    print(json.dumps({'attempts':len(rows),'report':str(path),'cycles':len(set(x['cycle'] for x in rows))},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','verify','preflight','job','advance','status','submit','release-pilot','report'))
    p.add_argument('--cycle',type=int)
    p.add_argument('--attempt',type=int)
    p.add_argument('--walltime',default='00:20:00')
    p.add_argument('--submit',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':print(json.dumps({'prepared':str(ROOT),'plan':freeze()},indent=2));return
    if a.action=='verify':verify();print('Frozen source and reference verified');return
    if a.action=='preflight':
        verify()
        source=str(ROOT/'source/src')
        if source not in sys.path:sys.path.insert(0,source)
        # Construct the real frozen suite on CPU without loading a model,
        # acquiring labels, retraining or evaluating the test set.
        e=new_experiment(device='cpu')
        st=e.initialize()
        if st.last_completed_cycle!=FIRST-1 or st.active_cycle is not None:
            raise ValueError('Preflight expected idle cycle-3 accepted state')
        recorded=e.labeled_sets.get_acquired_through(3)
        if len(recorded)!=3*473:
            raise ValueError('Imported accepted acquisitions missing')
        if len(e.available_ids())!=4253-3*473:
            raise ValueError('Imported candidate pool/ledger mismatch')
        write(ROOT/'preflight_pass.json',{'plan_sha256':sha(ROOT/'plan.json'),
              'source':str(ROOT/'source'),'first_cycle':FIRST,'seed':SEED,
              'last_completed_cycle':st.last_completed_cycle,'remaining_candidates':len(e.available_ids()),
              'evaluation_splits':list(e.evaluator.manifests)})
        print(json.dumps({'preflight':'PASS','seed':SEED,'first_failed_cycle':FIRST,
             'incumbent':accepted_baseline(FIRST),'original_rejected':read(ROOT/'plan.json')['original_c4_metrics'],
             'prior_acquired_objects':len(recorded),'remaining_candidates':len(e.available_ids()),
             'evaluation_splits':list(e.evaluator.manifests),
             'initial_gpu_jobs':1,'max_new_gpu_jobs':4+(LAST-FIRST)*5},indent=2));return
    if a.action=='job':
        if a.cycle is None or a.attempt is None:p.error('--cycle and --attempt are required for job')
        job(a.cycle,a.attempt);return
    if a.action=='advance':advance();return
    if a.action=='submit':submit(a.walltime,a.submit);return
    if a.action=='release-pilot':release_pilot();return
    if a.action=='status':status();return
    if a.action=='report':report();return


if __name__=='__main__':main()
