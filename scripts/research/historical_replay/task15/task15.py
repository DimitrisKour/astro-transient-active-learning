#!/usr/bin/env python3
"""Task15: frozen Task11-matched DIRECT / VReduce interactive acquisition study.

No jobs submitted by prepare, preflight or pilot. Jobs are held at submission;
only an explicit release command makes any job runnable. Independent policy/seed
chains preserve afterok dependencies between acquisition-training cycles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

POLICIES=('direct',)
SEEDS=(135,246,357)
CYCLES=9
SUITE='astrominn_task15_direct_vreduce_v1'
PERSISTENT=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
JOB=PERSISTENT/'closed_loop'/SUITE
TASK11=PERSISTENT/'closed_loop/astrominn_task11_advanced_acquisition_v1'
BASE='astrominn_historical_replay_v1'
TASK4='astrominn_closed_loop_warm_v1'
EXCLUDE=PERSISTENT/'runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):h.update(buf)
    return h.hexdigest()


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    tmp.replace(path)


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.sbatch','.toml'))


def hashes(root):return {str(p.relative_to(root)):sha(p) for p in files(root)}


def plan():
    return {
      'suite':SUITE,'method':'DIRECT interactive VReduce one-vs-rest',
      'paper':'Nuggehalli, Zhang, Jain, Nowak, ICML 2025, Algorithms 1 and 2',
      'policies':list(POLICIES),'seeds':list(SEEDS),'cycles':CYCLES,'fits':27,
      'classes':5,'parallel_annotation_batch':5,
      'oracle':'clean retrospective hidden-label provider; queried subbatches only',
      'oracle_journal':'atomic query transcript, deterministic replay after interruption',
      'phase_budget':'floor(B/2) VReduce and remainder near thresholds; allocated by seeded class order',
      'class_score':'p_k - max_j(p_j), j includes k, ties p_k descending',
      'initial_version':'conservative confirmed label interval; full-range fallback on overlap',
      'interval_objective':'minimize max empirical threshold loss at interval endpoints',
      'interval_rounding':'integer geometric reduction with half-up rounding',
      'threshold_ties':'closest to midpoint; neighbor ties by ranked index',
      'dataset':'Task4 identical L0/U0, 10% batch, complete pool',
      'bootstrap_source_suite':BASE,'split_seed':135,'warm_start':True,
      'training':{'epochs':50,'patience':10,'batch_size':512,'validation_batch_size':2048,
                  'learning_rate':1e-4,'sample_weighting':'per_sample'},
      'evaluation':'frozen_validation_only; no test selection',
      'source_baseline':'verified immutable Task11 frozen source; only Task15 patches',
      'dependency':'within policy/seed afterok only; no Task11-14 DAG dependencies',
      'noisy_oracle_extension':'not included; clean labels only in primary benchmark',
      'caveat':'object-level budget rounding and interval initialization specified here; not a claim of bitwise reproduction of paper code'}


def registry_routed(text):
    import tomllib
    original=tomllib.loads(text)
    match=re.search(r'(?ms)^\[storage\]\s*\n(.*?)(?=^\[|\Z)',text)
    if match is None:raise ValueError('Missing registry storage section')
    part=match.group(0)
    for key,val in (('persistent_root',PERSISTENT),('data_root',DATA)):
        part,count=re.subn(rf'(?m)^\s*{key}\s*=.*$',f'{key} = "{val}"',part)
        if count!=1:raise ValueError('Registry missing '+key)
    changed=text[:match.start()]+part+text[match.end():]
    if {k:v for k,v in original.items() if k!='storage'}!={k:v for k,v in tomllib.loads(changed).items() if k!='storage'}:
        raise ValueError('Nonstorage registry settings changed')
    return changed


def prepare(overlay):
    from task15_patch import patch_source
    if (JOB/'frozen_manifest.json').exists():verify();print('Already prepared');return
    if (JOB/'source').exists() or (JOB/'registry.toml').exists():
        raise RuntimeError('Partial Task15 preparation: inspect without overwriting')
    expected=load(TASK11/'frozen_manifest.json')
    old=hashes(TASK11/'source')
    if old!=expected:raise RuntimeError('Task11 source differs from its frozen manifest')
    for seed in SEEDS:
        checkpoint=DATA/'runs'/BASE/'shared'/f'seed_{seed}/bootstrap_training/best_vit_tower.pth'
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if not EXCLUDE.is_file():raise FileNotFoundError(EXCLUDE)
    overlay=Path(overlay).resolve()
    JOB.mkdir(parents=True,exist_ok=True);(JOB/'logs').mkdir(exist_ok=True)
    tmp=JOB/'source.tmp'
    if tmp.exists():raise RuntimeError('Previous staging directory exists')
    try:
        shutil.copytree(TASK11/'source',tmp,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for old_path in (tmp/'scripts/research/historical_replay/task15').glob('task11*'):
            old_path.unlink()
        for path in (overlay/'scripts/research/historical_replay/task15').glob('task15*'):
            shutil.copy2(path,tmp/'scripts/research/historical_replay/task15'/path.name)
        patch_source(tmp,overlay)
        import ast
        for p in files(tmp):
            if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        manifest=hashes(tmp)
        tmp.rename(JOB/'source')
        (JOB/'registry.toml').write_text(registry_routed((TASK11/'registry.toml').read_text()))
        dump(JOB/'scientific_plan.json',plan())
        dump(JOB/'source_parent.json',{'suite':'Task11','manifest_sha256':sha(TASK11/'frozen_manifest.json')})
        dump(JOB/'frozen_manifest.json',manifest)
    except Exception:
        if tmp.exists():shutil.rmtree(tmp)
        raise
    verify()


def verify(job=None):
    job=JOB if job is None else Path(job)
    if load(job/'scientific_plan.json')!=plan():raise RuntimeError('Scientific plan modified')
    if hashes(job/'source')!=load(job/'frozen_manifest.json'):
        raise RuntimeError('Frozen source no longer matches hash manifest')
    if not (job/'registry.toml').is_file():raise FileNotFoundError(job/'registry.toml')
    print(f'Task15 frozen source verified ({len(files(job/"source"))} files)')


def runner_args(policy,seed):
    if policy not in POLICIES or seed not in SEEDS:raise ValueError('Unknown case')
    return ['--registry',str(JOB/'registry.toml'),'--suite-id',SUITE,
      '--bootstrap-source-suite-id',BASE,'--require-existing-bootstrap',
      '--training-initialization','warm_start','--strategies',policy,
      '--seed',str(seed),'--split-seed','135','--initial-fraction','0.10',
      '--minimum-initial-per-class','5','--acquisition-fraction','0.10',
      '--max-cycles','9','--complete-pool','--max-epochs','50','--patience','10',
      '--batch-size','512','--validation-batch-size','2048','--learning-rate','0.0001',
      '--train-workers','0','--validation-workers','0',
      '--inference-workers','0','--evaluation-workers','0',
      '--assert-excluded-manifest',str(EXCLUDE)]


def preflight(seed):
    verify()
    env=dict(os.environ,PYTHONPATH=str(JOB/'source/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    for policy in POLICIES:
        subprocess.run([sys.executable,str(JOB/'source/scripts/run_astrominn_closed_loop.py'),
              *runner_args(policy,seed)],env=env,check=True)
    print('Task15 preflight complete; no GPU training/inference')


def pilot(policy,seed):
    """Pilot REAL interactive acquisition; no training or validation/test reads.

    Unlike non-oracle policies, this pilot intentionally queries *selected*
    historical hidden labels in subrounds. All query IDs are persisted.
    """
    verify()
    if policy not in POLICIES or seed not in SEEDS:raise ValueError('Unknown case')
    out=JOB/'pilot'/policy/f'seed_{seed}'/'result.json'
    if out.exists():
        print(json.dumps({k:v for k,v in load(out).items() if k!='selected_ids'},indent=2));return
    import numpy as np
    from dataclasses import replace
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.experiments.historical_replay import materialize_astrominn_historical_replay_split
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import DEFAULT_LEGACY_CODE_DIR,predict_astrominn_vit_pool
    from al_pipeline.astronomy.analysis.task11_labeled_features import predict_confirmed_objects
    from al_pipeline.astronomy.astrominn.training import MODEL_CLASS_NAMES, LABEL_TO_INDEX
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    from al_pipeline.core.advanced_acquisition.direct import DirectInput,DirectSelector
    registry=DatasetRegistry.from_toml(JOB/'registry.toml')
    resolved=registry.resolve_profile('astrominn_ztf');ds=resolved.dataset
    replay=materialize_astrominn_historical_replay_split(
        dataset=ds,destination=PERSISTENT/'runs'/SUITE/'shared/historical_replay_split',
        initial_fraction=.10,minimum_initial_per_class=5,split_seed=135,
        pool_id=SUITE+'_historical_u0',excluded_object_manifests=[EXCLUDE])
    def object_ids(path):
        with Path(path).open(newline='') as stream:
            return {row[ds.object_id_column] for row in csv.DictReader(stream)}
    task4=PERSISTENT/'runs'/TASK4/'shared/historical_replay_split'
    if object_ids(replay.initial_manifest)!=object_ids(task4/'initial_labeled_manifest.csv'):
        raise RuntimeError('Task15 L0 mismatch')
    u=CandidatePoolManifest.from_csv(replay.candidate_pool_dir/'candidate_pool.csv')
    u4=CandidatePoolManifest.from_csv(task4/'candidate_pool/candidate_pool.csv')
    if tuple(u.object_ids)!=tuple(u4.object_ids):raise RuntimeError('Task15 U0 mismatch')
    cp=DATA/'runs'/BASE/'shared'/f'seed_{seed}/bootstrap_training/best_vit_tower.pth'
    artifact=Path(resolved.profile.options['checkpoint']).parent
    t0=time.perf_counter()
    pred=predict_astrominn_vit_pool(pool_path=replay.candidate_pool_dir,
        object_ids=tuple(u.object_ids),batch_size=64,num_workers=0,device='cuda',
        legacy_code_dir=DEFAULT_LEGACY_CODE_DIR,artifact_dir=artifact,vit_checkpoint=cp)
    pool_time=time.perf_counter()-t0
    t0=time.perf_counter()
    anchor=predict_confirmed_objects(manifest_path=replay.initial_manifest,
        dataset=replace(ds,train_manifest=replay.initial_manifest),checkpoint_path=cp,
        artifact_dir=artifact,device='cuda',batch_size=64,num_workers=0,
        legacy_code_dir=DEFAULT_LEGACY_CODE_DIR)
    anchor_time=time.perf_counter()-t0
    known={}
    with replay.initial_manifest.open(newline='') as stream:
        for row in csv.DictReader(stream):
            oid=row[ds.object_id_column]
            label=MODEL_CLASS_NAMES[LABEL_TO_INDEX[row[ds.label_column]]]
            if oid in known and known[oid]!=label:raise ValueError('Conflicting L0 labels')
            known[oid]=label
    if set(anchor.object_ids)!=set(known):raise ValueError('Confirmed L0 inference mismatch')
    obj=DirectInput(candidate_ids=tuple(pred.object_ids),
        probabilities=pred.predictions.as_probabilities(),
        confirmed_ids=tuple(anchor.object_ids),
        confirmed_probabilities=anchor.predictions.as_probabilities(),
        confirmed_labels=tuple(known[x] for x in anchor.object_ids),
        class_names=tuple(MODEL_CLASS_NAMES))
    # The pilot oracle exposes only labels explicitly requested via query.
    with replay.hidden_labels.open(newline='') as stream:
        reader=csv.DictReader(stream)
        rows=list(reader)
    if not rows:raise RuntimeError('Hidden-label file empty')
    candidates={row['object_id']:MODEL_CLASS_NAMES[LABEL_TO_INDEX[row['target_label']]] for row in rows}
    if set(candidates)!=set(pred.object_ids):raise RuntimeError('Hidden table does not match U0')
    logs=[]
    def query(ids,*,phase,class_name,subround):
        if any(x not in candidates for x in ids):raise KeyError('Unknown requested object')
        labels=[candidates[x] for x in ids]
        logs.append({'phase':phase,'class':class_name,'subround':subround,'ids':list(ids),'labels':labels})
        return labels
    budget=min(len(pred.object_ids),math.ceil(load(replay.summary_path)['historical_objects']*.10))
    t0=time.perf_counter()
    seed_int=int(np.random.SeedSequence([seed,1,15015]).generate_state(1)[0])
    selected=DirectSelector(obj,parallel_batch=5,seed=seed_int).select(budget=budget,query=query)
    selection_seconds=time.perf_counter()-t0
    revealed={x for entry in logs for x in entry['ids']}
    if len(revealed)!=budget or set(selected.ids)!=revealed:raise RuntimeError('Pilot oracle trace mismatch')
    reference=load(PERSISTENT/'runs'/TASK4/'entropy'/f'seed_{seed}/cycle_001/selection.json')['ids']
    dump(out,{'policy':policy,'seed':seed,'selected_ids':selected.ids,
        'selected_unique':True,'selection_budget':budget,'candidate_count':len(pred.object_ids),
        'pilot_oracle_transcript':logs,'metadata':selected.metadata,
        'pool_inference_seconds':pool_time,'anchor_inference_seconds':anchor_time,
        'selection_seconds':selection_seconds,'oracle_subrounds':len(logs),
        'entropy_overlap_count':len(set(selected.ids)&set(reference)),
        'checkpoint_sha256':sha(cp),'initial_manifest_sha256':sha(replay.initial_manifest),
        'task4_L0_U0_equal':True,'unqueried_labels_used_in_acquisition':False,
        'validation_labels_accessed':False,'test_labels_accessed':False})
    print(json.dumps({k:v for k,v in load(out).items() if k not in ('selected_ids','pilot_oracle_transcript')},indent=2))


def pilot_status():
    rows=[]
    for policy in POLICIES:
        file=JOB/'pilot'/policy/'seed_135/result.json'
        rows.append({'policy':policy,'exists':file.is_file(),
                     'selection_seconds':load(file)['selection_seconds'] if file.is_file() else None})
    print(json.dumps(rows,indent=2))


def filesystem_probe():
    path=DATA/'runs'/SUITE/'shared/probe'
    path.mkdir(parents=True,exist_ok=True)
    temp=path/f'probe_{os.getpid()}.tmp'
    payload=os.urandom(16*1024*1024)
    try:
        with temp.open('wb') as fh:
            fh.write(payload);fh.flush();os.fsync(fh.fileno())
        if temp.read_bytes()!=payload:raise RuntimeError('Task15 /work/hdd readback failed')
        print('Task15 /work/hdd 16MiB filesystem probe passed')
    finally:temp.unlink(missing_ok=True)


def jobs():
    for policy in POLICIES:
        for seed in SEEDS:
            for cycle in range(1,CYCLES+1):yield policy,seed,cycle


def submit(walltime):
    verify()
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d',walltime) or walltime=='00:00:00':
        raise ValueError('Invalid walltime')
    for policy in POLICIES:
        f=JOB/'pilot'/policy/'seed_135/result.json'
        if not f.is_file() or not load(f).get('selected_unique'):
            raise RuntimeError(f'Missing successful acquisition-only GPU pilot: {policy}')
    if (JOB/'submitted.json').exists():raise RuntimeError('Task15 already submitted')
    made=[];ledger=[];prev={}
    try:
        for policy,seed,cycle in jobs():
            before=prev.get((policy,seed))
            opts=['sbatch','--parsable','--hold','--time',walltime,
                  '--job-name',f'T15_{policy}_{seed}_{cycle}',
                  '--output',str(JOB/'logs/%x_%j.out'),
                  '--error',str(JOB/'logs/%x_%j.err')]
            if before:opts.extend(['--dependency',f'afterok:{before}'])
            opts.extend([str(JOB/'source/scripts/research/historical_replay/task15/task15_worker.sbatch'),
                         str(JOB),'job',policy,str(seed),str(cycle)])
            result=subprocess.check_output(opts,text=True).strip().split(';')[0]
            if not result.isdigit():raise RuntimeError('Unexpected sbatch response '+result)
            made.append(result);prev[(policy,seed)]=result
            ledger.append({'policy':policy,'seed':seed,'cycle':cycle,'job_id':result,
                           'parent':before,'released':False})
        dump(JOB/'submitted.json',{'state':'held','jobs':ledger,'walltime':walltime})
    except Exception:
        if made and not (JOB/'submitted.json').exists():subprocess.run(['scancel',*made],check=False)
        raise
    print(f'Submitted {len(made)} HELD Task15 jobs in three independent chains. No jobs released.')


def release(policy=None,seed=None,cycle=None,all_jobs=False):
    ledger=JOB/'submitted.json'
    state=load(ledger)
    if all_jobs:
        if any(x is not None for x in (policy,seed,cycle)):
            raise ValueError('Cannot mix --all with --policy/--seed/--cycle')
        target=[j for j in state['jobs'] if not j['released']]
    else:
        if policy not in POLICIES or seed not in SEEDS or cycle not in range(1,CYCLES+1):
            raise ValueError('Use --policy, --seed, and --cycle together, or --all')
        target=[j for j in state['jobs'] if (j['policy'],j['seed'],j['cycle'])==(policy,seed,cycle)]
    for job in target:
        if job['released']:continue
        subprocess.run(['scontrol','release',job['job_id']],check=True)
        job['released']=True
        dump(ledger,state)  # crash safe / explicit records
        print('Released',job['job_id'],job['policy'],job['seed'],job['cycle'])
    if all(j['released'] for j in state['jobs']):state['state']='released_all';dump(ledger,state)


def status():
    rows=[]
    for policy in POLICIES:
        for seed in SEEDS:
            root=PERSISTENT/'runs'/SUITE/policy/f'seed_{seed}'
            state=root/'run_state.json'
            done=int(load(state).get('last_completed_cycle',0)) if state.is_file() else 0
            summaries=sum((root/f'cycle_{i:03d}/cycle_summary.json').is_file() for i in range(1,CYCLES+1))
            rows.append({'policy':policy,'seed':seed,'completed_cycles':done,'summaries':summaries,
                         'complete':done==CYCLES and summaries==CYCLES})
    print(json.dumps({'suite':SUITE,'finished_trajectories':sum(x['complete'] for x in rows),
                      'total_trajectories':len(rows),'planned_fits':len(rows)*CYCLES,
                      'submission':load(JOB/'submitted.json')['state'] if (JOB/'submitted.json').is_file() else 'not_submitted',
                      'series':rows},indent=2))


def collect(destination):
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    paths=[JOB/'scientific_plan.json',JOB/'frozen_manifest.json',JOB/'source_parent.json',
           JOB/'submitted.json']
    paths.extend((JOB/'pilot').rglob('result.json') if (JOB/'pilot').exists() else [])
    for policy in POLICIES:
        for seed in SEEDS:
            run=PERSISTENT/'runs'/SUITE/policy/f'seed_{seed}'
            artifact=DATA/'runs'/SUITE/policy/f'seed_{seed}'
            for name in ('run_state.json','evaluation_history.json','experiment.json'):
                paths.append(run/name)
            for cycle in range(0,CYCLES+1):
                cy=run/f'cycle_{cycle:03d}'
                for name in ('selection.json','revealed_labels.json','direct_oracle_journal.json','cycle_summary.json'):
                    paths.append(cy/name)
                acy=artifact/f'cycle_{cycle:03d}'/'evaluation'
                for name in ('evaluation_summary.json','validation_per_class_metrics.json',
                             'validation_object_confusion.csv','validation_sample_confusion.csv'):
                    paths.append(acy/name)
    seen=set();count=0
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file() and path not in seen:
                seen.add(path)
                name='experiment/'+str(path.relative_to(PERSISTENT)) if path.is_relative_to(PERSISTENT) else 'artifacts/'+str(path.relative_to(DATA))
                archive.write(path,name);count+=1
    print(json.dumps({'output':str(dest),'files':count,'bytes':dest.stat().st_size,'sha256':sha(dest)},indent=2))


def training_job(policy,seed,cycle,job_dir):
    if policy not in POLICIES or seed not in SEEDS or cycle not in range(1,CYCLES+1):
        raise ValueError('Invalid Task15 training case')
    verify(job_dir)
    env=dict(os.environ,PYTHONPATH=str(Path(job_dir)/'source/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    runner=Path(job_dir)/'source/scripts/run_astrominn_closed_loop.py'
    subprocess.run([sys.executable,str(runner),*runner_args(policy,seed),
                    '--device','cuda','--cycle',str(cycle),'--run'],env=env,check=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=('plan','prepare','verify','preflight','pilot','pilot-status',
                     'filesystem-probe','submit','release','status','collect','job'))
    ap.add_argument('policy_pos',nargs='?');ap.add_argument('seed_pos',nargs='?',type=int)
    ap.add_argument('cycle_pos',nargs='?',type=int)
    ap.add_argument('--overlay',type=Path,default=Path(__file__).resolve().parents[4])
    ap.add_argument('--job-dir',type=Path,default=JOB)
    ap.add_argument('--seed',type=int,default=None)
    ap.add_argument('--policy',choices=POLICIES)
    ap.add_argument('--cycle',type=int)
    ap.add_argument('--walltime',default='00:35:00')
    ap.add_argument('--destination',type=Path)
    ap.add_argument('--all',action='store_true')
    arg=ap.parse_args()
    command=arg.command
    if command=='plan':print(json.dumps(plan(),indent=2));return
    if command=='prepare':prepare(arg.overlay);return
    if command=='verify':verify(arg.job_dir);return
    if command=='preflight':preflight(arg.seed if arg.seed is not None else 135);return
    if command=='pilot':pilot(arg.policy or arg.policy_pos,arg.seed if arg.seed is not None else 135);return
    if command=='pilot-status':pilot_status();return
    if command=='filesystem-probe':filesystem_probe();return
    if command=='submit':submit(arg.walltime);return
    if command=='release':release(arg.policy,arg.seed,arg.cycle,arg.all);return
    if command=='status':status();return
    if command=='collect':collect(arg.destination or Path('/u/dkourtesis/task15_results.zip'));return
    if command=='job':training_job(arg.policy_pos,arg.seed_pos,arg.cycle_pos,arg.job_dir);return

if __name__=='__main__':main()
