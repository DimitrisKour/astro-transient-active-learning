#!/usr/bin/env python3
"""Task11: frozen, Task4-matched BADGE/core-set/Fisher closed loops.

Preparation and inspection are CPU/read-only except creation of isolated suite
and snapshot directories. The explicit pilot command scores U0 without training;
submission is blocked until all three pilots have succeeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

POLICIES=('badge','core_set','fisher_batch')
SEEDS=(135,246,357)
SUITE='astrominn_task11_advanced_acquisition_v1'
PERSISTENT=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
JOB=PERSISTENT/'closed_loop'/SUITE
TASK4='astrominn_closed_loop_warm_v1'
TASK4_FROZEN=PERSISTENT/'closed_loop'/TASK4
BASE='astrominn_historical_replay_v1'
EXCLUDE=PERSISTENT/'runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def registry_routed(text):
    import tomllib
    original=tomllib.loads(text)
    match=re.search(r'(?ms)^\[storage\]\s*\n(.*?)(?=^\[|\Z)',text)
    if match is None: raise ValueError('Source registry missing storage section')
    section=match.group(0)
    for name,path in [('persistent_root',PERSISTENT),('data_root',DATA)]:
        section,count=re.subn(rf'(?m)^\s*{name}\s*=.*$',f'{name} = "{path}"',section)
        if count!=1:raise ValueError('Unable to route registry storage.'+name)
    routed=text[:match.start()]+section+text[match.end():]
    result=tomllib.loads(routed)
    if {k:v for k,v in result.items() if k!='storage'} != {k:v for k,v in original.items() if k!='storage'}:
        raise ValueError('Registry configuration changed other than storage')
    return routed


def source_files(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.sbatch','.toml'))


def frozen_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in source_files(root)}


def scientific_plan():
    return {'suite':SUITE,'seeds':list(SEEDS),'policies':list(POLICIES),
            'cycles':9,'fits':81,'training_initialization':'warm_start',
            'l0_bootstrap_source_suite':BASE,'split_seed':135,
            'initial_fraction':0.1,'minimum_initial_per_class':5,
            'acquisition_fraction_of_historical':0.10,'complete_pool':True,
            'max_epochs':50,'patience':10,'batch_size':512,
            'validation_batch_size':2048,'learning_rate':1e-4,
            'training_sample_weighting':'per_sample',
            'class_weights':'training_manifest',
            'evaluation':'frozen_validation_only',
            'fisher':{'type':'projected_expected_last_layer_Fisher_A_optimal',
                      'selection':'forward_greedy_Woodbury','rank':32,'ridge':1.0,
                      'projection_seed':271828,'not_full_BAIT':True},
            'badge':{'gradient':'last_linear_softmax_pseudolabel_gradient',
                     'selection':'kmeans++_factored','include_bias':True},
            'core_set':{'selection':'labeled_anchored_greedy_k_center',
                         'geometry':'unit_normalized_final_classifier_inputs'},
            'policy_development':'fixed_before_full_closed_loop',
            'task4_baseline':TASK4,
            'source_baseline':'Task4 frozen source tree, not mutable checkout'}


def prepare(repo,overlay):
    job=JOB
    if (job/'frozen_manifest.json').exists():
        verify(job)
        print('Task11 already prepared, frozen source verified')
        return
    if (job/'source').exists() or (job/'registry.toml').exists():
        raise RuntimeError('Incomplete Task11 preparation: inspect instead of overwriting')
    from task11_patch import patch_source
    repo=Path(repo).resolve();overlay=Path(overlay).resolve()
    base=TASK4_FROZEN/'source'
    baseline_snapshot=TASK4_FROZEN/'source_snapshot.json'
    if not baseline_snapshot.is_file():
        raise FileNotFoundError('Original frozen Task4 source snapshot missing')
    old_hashes=load(baseline_snapshot)
    for rel,expected in old_hashes.items():
        actual=base/rel
        if not actual.is_file() or sha(actual)!=expected:
            raise ValueError(f'Original Task4 source changed: {rel}')
    if not (base/'src/al_pipeline').is_dir():
        raise FileNotFoundError('Task4 frozen astronomy source tree missing')
    if not (base/'scripts/run_astrominn_closed_loop.py').is_file():
        raise FileNotFoundError('Original Task4 runner missing')
    registry=TASK4_FROZEN/'registry.toml'
    if not registry.is_file():raise FileNotFoundError('Task4 frozen registry missing')
    for seed in SEEDS:
        checkpoint=DATA/'runs'/BASE/'shared'/f'seed_{seed}'/'bootstrap_training/best_vit_tower.pth'
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if not EXCLUDE.is_file():raise FileNotFoundError(EXCLUDE)
    job.mkdir(parents=True,exist_ok=True)
    job.joinpath('logs').mkdir(exist_ok=True)
    tmp=job/'source.tmp'
    if tmp.exists():raise RuntimeError('Source staging directory already exists')
    try:
        shutil.copytree(base/'src/al_pipeline',tmp/'src/al_pipeline')
        (tmp/'scripts/adaptive').mkdir(parents=True)
        shutil.copy2(base/'scripts/run_astrominn_closed_loop.py',tmp/'scripts/run_astrominn_closed_loop.py')
        # Task4 may predate opt-in advanced selectors. Import those three
        # model-independent modules only from the current verified checkout;
        # never substitute its training or closed-loop runner implementations.
        selectors=Path('src/al_pipeline/core/advanced_acquisition')
        if not (repo/selectors/'badge.py').is_file() or not (repo/selectors/'core_set.py').is_file():
            raise FileNotFoundError('Advanced selector package missing from checkout')
        for incoming in (repo/selectors).glob('*.py'):
            rel=selectors/incoming.name
            if not (tmp/rel).exists():
                (tmp/rel).parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(incoming,tmp/rel)
        for file in (overlay/'scripts/adaptive').glob('task11*'):
            shutil.copy2(file,tmp/'scripts/adaptive'/file.name)
        patch_source(tmp,overlay)
        for py in source_files(tmp):
            if py.suffix=='.py':
                import ast
                ast.parse(py.read_text(),filename=str(py))
        manifest=frozen_hashes(tmp)
        routed=registry_routed(registry.read_text())
        (job/'source').parent.mkdir(parents=True,exist_ok=True)
        tmp.rename(job/'source')
        (job/'registry.toml').write_text(routed)
        dump(job/'scientific_plan.json',scientific_plan())
        dump(job/'frozen_manifest.json',manifest)
    except Exception:
        if tmp.exists():shutil.rmtree(tmp)
        raise
    verify(job)
    print(f'Prepared {len(manifest)} frozen source files: {job}')


def verify(job=JOB):
    job=Path(job)
    plan=load(job/'scientific_plan.json')
    if plan!=scientific_plan():raise ValueError('Scientific plan changed')
    frozen=load(job/'frozen_manifest.json')
    actual=frozen_hashes(job/'source')
    if actual!=frozen:
        missing=set(frozen)-set(actual);added=set(actual)-set(frozen)
        changed=[x for x in frozen.keys() & actual.keys() if frozen[x]!=actual[x]]
        raise ValueError(f'Frozen Task11 source mismatch missing={missing}, added={added}, changed={changed[:10]}')
    if not (job/'registry.toml').is_file():raise FileNotFoundError('Frozen registry missing')
    print(f'Task11 frozen source verified ({len(frozen)} files)')


def task4_args(seed,policy):
    if policy not in POLICIES or seed not in SEEDS:raise ValueError('Invalid case')
    return ['--registry',str(JOB/'registry.toml'),'--suite-id',SUITE,
            '--bootstrap-source-suite-id',BASE,'--require-existing-bootstrap',
            '--training-initialization','warm_start','--strategies',policy,
            '--seed',str(seed),'--split-seed','135','--initial-fraction','0.10',
            '--minimum-initial-per-class','5','--acquisition-fraction','0.10',
            '--max-cycles','9','--complete-pool','--max-epochs','50',
            '--patience','10','--batch-size','512','--validation-batch-size','2048',
            '--learning-rate','0.0001','--train-workers','0',
            '--validation-workers','0','--inference-workers','0',
            '--evaluation-workers','0','--assert-excluded-manifest',str(EXCLUDE)]


def preflight(seed=135):
    verify()
    env=dict(os.environ,PYTHONPATH=str(JOB/'source/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    for policy in POLICIES:
        subprocess.run([sys.executable,str(JOB/'source/scripts/run_astrominn_closed_loop.py'),
                        *task4_args(seed,policy)],env=env,check=True)
    print('Task11 no-training preflight completed')


def pilot(policy,seed=135):
    verify()
    if policy not in POLICIES or seed not in SEEDS:raise ValueError('Invalid pilot')
    # Read same initial L0, U0, and bootstrap as Task4.
    import torch
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.experiments.historical_replay import materialize_astrominn_historical_replay_split
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (DEFAULT_LEGACY_CODE_DIR,predict_astrominn_vit_pool)
    from al_pipeline.astronomy.analysis.task11_labeled_features import predict_confirmed_objects
    from al_pipeline.core.advanced_acquisition.badge import BadgeInput,BadgeGradientNormScorer,BadgeSelector
    from al_pipeline.core.advanced_acquisition.core_set import CoreSetInput,CoreSetScorer,CoreSetSelector
    from al_pipeline.core.advanced_acquisition.fisher_batch import FisherBatchInput,FisherBatchScorer,FisherBatchSelector
    from al_pipeline.core.pool import Pool
    import numpy as np

    dest=JOB/'pilot'/policy/f'seed_{seed}'
    done=dest/'result.json'
    if done.is_file():
        print(done.read_text());return
    registry=DatasetRegistry.from_toml(JOB/'registry.toml')
    resolved=registry.resolve_profile('astrominn_ztf')
    ds=resolved.dataset
    replay=materialize_astrominn_historical_replay_split(
        dataset=ds,destination=PERSISTENT/'runs'/SUITE/'shared'/'historical_replay_split',
        initial_fraction=.10,minimum_initial_per_class=5,split_seed=135,
        pool_id=f'{SUITE}_historical_u0',excluded_object_manifests=[EXCLUDE])
    original=PERSISTENT/'runs'/TASK4/'shared/historical_replay_split'
    # Split identity is verified via object-set equality (pool ID may differ).
    # Scientific constraints: same initial manifest and candidate IDs, not just counts.
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    def ids_of_manifest(file):
        import csv
        with open(file,newline='') as f:
            return {r[ds.object_id_column] for r in csv.DictReader(f)}
    current_ids=ids_of_manifest(replay.initial_manifest)
    task4_initial=original/'initial_labeled_manifest.csv'
    if not task4_initial.is_file():
        raise FileNotFoundError(f'Task4 reference L0 manifest missing: {task4_initial}')
    if current_ids!=ids_of_manifest(task4_initial):
        raise ValueError('Task11 L0 differs from Task4 L0')
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    task4_u=CandidatePoolManifest.from_csv(original/'candidate_pool/candidate_pool.csv')
    this_u=CandidatePoolManifest.from_csv(replay.candidate_pool_dir/'candidate_pool.csv')
    if tuple(task4_u.object_ids)!=tuple(this_u.object_ids):
        raise ValueError('Task11 initial U0 order differs from Task4')
    checkpoint=DATA/'runs'/BASE/'shared'/f'seed_{seed}'/'bootstrap_training/best_vit_tower.pth'
    artifact_dir=Path(resolved.profile.options['checkpoint']).parent
    t0=time.perf_counter()
    prediction=predict_astrominn_vit_pool(pool_path=replay.candidate_pool_dir,device='cuda',
                 batch_size=64,num_workers=0,legacy_code_dir=DEFAULT_LEGACY_CODE_DIR,
                 artifact_dir=artifact_dir,vit_checkpoint=checkpoint)
    infer_seconds=time.perf_counter()-t0
    if prediction.embeddings is None:raise ValueError('Pool predictions missing classifier inputs')
    object_ids=tuple(prediction.object_ids)
    prob=prediction.predictions
    anchors=None
    if policy!='badge':
        t1=time.perf_counter()
        anchors=predict_confirmed_objects(
            manifest_path=replay.initial_manifest,dataset=replace(ds,train_manifest=replay.initial_manifest),
            checkpoint_path=checkpoint,artifact_dir=artifact_dir,
            device='cuda',batch_size=64,num_workers=0,legacy_code_dir=DEFAULT_LEGACY_CODE_DIR)
        anchor_seconds=time.perf_counter()-t1
    else:anchor_seconds=0.
    if policy=='badge':
        data=BadgeInput(predictions=prob,embeddings=prediction.embeddings,candidate_ids=object_ids)
        scorer=BadgeGradientNormScorer()
        selector=BadgeSelector(data,seed=int(np.random.SeedSequence([seed,1,11]).generate_state(1)[0]))
    elif policy=='core_set':
        def unit(x):
            v=np.asarray(x,dtype=np.float64);return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
        data=CoreSetInput(candidate_embeddings=unit(prediction.embeddings),
              labeled_embeddings=unit(anchors.embeddings),candidate_ids=object_ids,
              labeled_ids=tuple(anchors.object_ids))
        scorer=CoreSetScorer();selector=CoreSetSelector(data)
    else:
        data=FisherBatchInput(candidate_ids=object_ids,
                probabilities=prob.as_probabilities(),embeddings=prediction.embeddings,
                labeled_probabilities=anchors.predictions.as_probabilities(),
                labeled_embeddings=anchors.embeddings)
        scorer=FisherBatchScorer();selector=FisherBatchSelector(data)
    from al_pipeline.core.experiments.budget import plan_cycle_budget
    historical_objects=load(replay.summary_path)['historical_objects']
    import math
    k=min(len(object_ids),math.ceil(historical_objects*.1))
    t2=time.perf_counter()
    scores=scorer.score(data)
    selection=selector.select(ids=list(object_ids),scores=scores,k=k,pool=Pool.from_ids(object_ids))
    selection_seconds=time.perf_counter()-t2
    if len(set(selection.ids))!=k:raise RuntimeError('Acquisition contains duplicate IDs')
    entropy_reference=PERSISTENT/'runs'/TASK4/'entropy'/f'seed_{seed}'/'cycle_001/selection.json'
    if not entropy_reference.is_file():raise FileNotFoundError(entropy_reference)
    entropy_ids=load(entropy_reference)['ids']
    if len(entropy_ids)!=k:raise ValueError('Task4 entropy comparison budget differs')
    overlap=len(set(selection.ids)&set(entropy_ids))
    dest.mkdir(parents=True,exist_ok=True)
    dump(done,{'policy':policy,'seed':seed,'selection_budget':k,
               'candidate_count':len(object_ids),'anchor_count':None if anchors is None else len(anchors.object_ids),
               'selected_ids':list(selection.ids),'entropy_overlap_count':overlap,
               'entropy_overlap_fraction':overlap/k,
               'pool_inference_seconds':infer_seconds,'anchor_inference_seconds':anchor_seconds,
               'selector_seconds':selection_seconds,'selector_metadata':selection.metadata,
               'checkpoint_sha256':sha(checkpoint),
               'initial_manifest_sha256':sha(replay.initial_manifest),
               'selected_unique':True,'not_validation_tuned':True})
    print(json.dumps({k:v for k,v in load(done).items() if k!='selected_ids'},indent=2))


def pilot_status():
    for policy in POLICIES:
        f=JOB/'pilot'/policy/'seed_135/result.json'
        if not f.exists():
            print(policy, 'MISSING')
        else:
            data=load(f)
            print(policy,f'overlap {data["entropy_overlap_fraction"]:.3%}',
                  f'select {data["selector_seconds"]:.1f}s',
                  f'infer {data["pool_inference_seconds"]:.1f}s')


def jobs():
    for policy in POLICIES:
        for seed in SEEDS:
            for cycle in range(1,10):
                yield (policy,seed,cycle)


def submit(walltime,require_pilots=True):
    verify()
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d',walltime) or walltime=='00:00:00':
        raise ValueError('Invalid walltime')
    if require_pilots:
        for policy in POLICIES:
            result=JOB/'pilot'/policy/'seed_135/result.json'
            if not result.exists() or not load(result)['selected_unique']:
                raise ValueError(f'Acquisition pilot not complete: {policy}')
            if policy=='fisher_batch' and load(result)['entropy_overlap_fraction']>=.95:
                raise ValueError('Fisher pilot overlap >=95% with entropy; inspect before committing 81 fits')
    if (JOB/'submitted.json').exists():raise RuntimeError('Task11 already submitted')
    made=[];alljobs=[];previous={}
    try:
        for policy,seed,cycle in jobs():
            opts=['sbatch','--parsable','--hold','--time',walltime,
                  '--job-name',f'T11_{policy[:5]}_{seed}_{cycle}',
                  '--output',str(JOB/'logs/%x_%j.out'),
                  '--error',str(JOB/'logs/%x_%j.err')]
            last=previous.get((policy,seed))
            if last:opts+=['--dependency',f'afterok:{last}']
            opts += [str(JOB/'source/scripts/adaptive/task11_worker.sbatch'),str(JOB),'job',policy,str(seed),str(cycle)]
            result=subprocess.check_output(opts,text=True).strip()
            jobid=result.split(';')[0]
            if not jobid.isdigit():raise RuntimeError(f'Unexpected sbatch ID {result}')
            made.append(jobid)
            previous[(policy,seed)]=jobid
            alljobs.append({'policy':policy,'seed':seed,'cycle':cycle,'job_id':jobid,'parent':last})
        if len(made)!=81:raise RuntimeError('Expected exactly 81 training fits')
        # Record before releasing, preserving audit even after crash.
        dump(JOB/'submitted.json',{'jobs':alljobs,'status':'held_awaiting_release'})
        for jobid in reversed(made):
            subprocess.run(['scontrol','release',jobid],check=True)
        dump(JOB/'submitted.json',{'jobs':alljobs,'status':'released'})
    except Exception:
        if not (JOB/'submitted.json').exists() and made:
            subprocess.run(['scancel',*made],check=False)
        raise
    print('Submitted 81 jobs as 9 independent dependency chains')


def status():
    rows=[]
    for policy in POLICIES:
        for seed in SEEDS:
            base=PERSISTENT/'runs'/SUITE/policy/f'seed_{seed}'
            state=base/'run_state.json'
            data=load(state) if state.exists() else {}
            completed=int(data.get('last_completed_cycle',0))
            summaries=sum((base/f'cycle_{i:03d}/cycle_summary.json').is_file() for i in range(1,10))
            rows.append({'policy':policy,'seed':seed,'last_completed_cycle':completed,
                         'cycle_summaries':summaries,'complete':completed==9 and summaries==9})
    payload={'suite':SUITE,'complete':sum(x['complete'] for x in rows),'planned':9,'cases':rows}
    print(json.dumps(payload,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['plan','prepare','verify','preflight','pilot','pilot-status','submit','status','job'])
    p.add_argument('policy',nargs='?',choices=POLICIES)
    p.add_argument('seed',nargs='?',type=int)
    p.add_argument('cycle',nargs='?',type=int)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--overlay',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--job-dir',type=Path,default=JOB)
    p.add_argument('--walltime',default='00:20:00')
    p.add_argument('--seed',dest='pilot_seed',type=int,default=135)
    p.add_argument('--policy',dest='pilot_policy',choices=POLICIES)
    args=p.parse_args()
    if args.command=='plan':print(json.dumps(scientific_plan(),indent=2));return
    if args.command=='prepare':prepare(args.repo,args.overlay);return
    if args.command=='verify':verify(args.job_dir);return
    if args.command=='preflight':preflight(args.pilot_seed);return
    if args.command=='pilot':pilot(args.pilot_policy or args.policy,args.pilot_seed);return
    if args.command=='pilot-status':pilot_status();return
    if args.command=='status':status();return
    if args.command=='submit':submit(args.walltime);return
    if args.command=='job':
        policy=args.policy;seed=args.seed;cycle=args.cycle
        if policy not in POLICIES or seed not in SEEDS or cycle not in range(1,10):raise ValueError('Invalid job case')
        verify(args.job_dir)
        runner=args.job_dir/'source/scripts/run_astrominn_closed_loop.py'
        env=dict(os.environ,PYTHONPATH=str(args.job_dir/'source/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
        subprocess.run([sys.executable,str(runner),*task4_args(seed,policy),
                        '--device','cuda','--cycle',str(cycle),'--run'],env=env,check=True)
        return

if __name__=='__main__':main()
