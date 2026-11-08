#!/usr/bin/env python3
"""Task12: validation-only 3x9 MC-dropout BALD benchmark, Task4 matched.

Freeze source from Task4, verify real ViT dropout with an acquisition-only
pilot, and schedule chains only behind completed/successful Task11 chains.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEEDS=(135,246,357)
POLICY='bald'
SUITE='astrominn_task12_bald_mc20_v1'
PERSISTENT=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
JOB=PERSISTENT/'closed_loop'/SUITE
TASK4='astrominn_closed_loop_warm_v1'
TASK4_FROZEN=PERSISTENT/'closed_loop'/TASK4
TASK11='astrominn_task11_advanced_acquisition_v1'
BASE='astrominn_historical_replay_v1'
EXCLUDE=PERSISTENT/'runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):
            h.update(c)
    return h.hexdigest()


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    tmp.replace(path)


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def routed_registry(text):
    import tomllib
    original=tomllib.loads(text)
    match=re.search(r'(?ms)^\[storage\]\s*\n(.*?)(?=^\[|\Z)',text)
    if match is None:raise ValueError('Missing [storage] section')
    sect=match.group(0)
    for key,val in [('persistent_root',PERSISTENT),('data_root',DATA)]:
        sect,n=re.subn(rf'(?m)^\s*{key}\s*=.*$',f'{key} = "{val}"',sect)
        if n!=1:raise ValueError('Missing '+key)
    result=text[:match.start()]+sect+text[match.end():]
    parsed=tomllib.loads(result)
    if {k:v for k,v in original.items() if k!='storage'} != {k:v for k,v in parsed.items() if k!='storage'}:
        raise ValueError('Registry non-storage sections changed')
    return result


def source_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.sbatch','.toml'))


def source_hashes(root):return {str(p.relative_to(root)):sha(p) for p in source_files(root)}


def plan():
    return {'suite':SUITE,'method':'BALD',
        'bald_objective':'H[E_w p(y|x,w)] - E_w H[p(y|x,w)] in nats',
        'approximate_posterior':'MC_dropout_ViT_active_modules',
        'coherent_shared_dropout_masks_across_pool':True,
        'freeze_batchnorm_eval':True,
        'posterior_samples':20,
        'latest_alert_only_stochastic_inference':True,
        'object_selection':'TopK_BaldMI',
        'seeds':list(SEEDS),'cycles':9,'fits':27,
        'split_seed':135,'L0_initial_fraction':.10,'minimum_initial_per_class':5,
        'acquisition_fraction_of_historical':.10,'complete_pool':True,
        'l0_bootstrap_source_suite':BASE,
        'training_initialization':'warm_start','training_sample_weighting':'per_sample',
        'class_weights':'training_manifest',
        'max_epochs':50,'patience':10,'batch_size':512,
        'validation_batch_size':2048,'learning_rate':1e-4,
        'evaluation':'frozen_validation_only','test_used_for_selection':False,
        'source_baseline':'original_verified_Task4_frozen_source',
        'external_submission_dependency':'successful_nine_Task11_final_cycle_jobs',
        'batchbald':'deferred_due_large_473_object_batches_and_approximation_requirements',
        'policy_development':'fixed_prior_to_closed_loop_after_real_dropout_pilot'}


def prepare(repo, overlay):
    from task12_patch import patch_source
    if (JOB/'frozen_manifest.json').is_file(): verify();print('Already prepared');return
    if (JOB/'source').exists() or (JOB/'registry.toml').exists():
        raise RuntimeError('Partial preparation exists; inspect before proceeding')
    base=TASK4_FROZEN/'source'
    snap=TASK4_FROZEN/'source_snapshot.json'
    if not snap.is_file(): raise FileNotFoundError(snap)
    expected=load(snap)
    for rel,hashval in expected.items():
        if not (base/rel).is_file() or sha(base/rel)!=hashval:
            raise ValueError(f'Task4 frozen source mismatch: {rel}')
    for seed in SEEDS:
        source=DATA/'runs'/BASE/'shared'/f'seed_{seed}'/'bootstrap_training/best_vit_tower.pth'
        if not source.is_file():raise FileNotFoundError(source)
    if not EXCLUDE.is_file():raise FileNotFoundError(EXCLUDE)
    reg=TASK4_FROZEN/'registry.toml'
    if not reg.is_file():raise FileNotFoundError(reg)
    overlay=Path(overlay).resolve()
    JOB.mkdir(parents=True,exist_ok=True)
    (JOB/'logs').mkdir(exist_ok=True)
    temp=JOB/'source.tmp'
    if temp.exists():raise RuntimeError('Source staging area already exists')
    try:
        shutil.copytree(base/'src/al_pipeline',temp/'src/al_pipeline')
        (temp/'scripts/research/historical_replay/task12').mkdir(parents=True)
        shutil.copy2(base/'scripts/run_astrominn_closed_loop.py',temp/'scripts/run_astrominn_closed_loop.py')
        for src in (overlay/'scripts/research/historical_replay/task12').glob('task12*'):
            shutil.copy2(src,temp/'scripts/research/historical_replay/task12'/src.name)
        patch_source(temp,overlay)
        import ast
        for file in source_files(temp):
            if file.suffix=='.py':ast.parse(file.read_text(),filename=str(file))
        digest=source_hashes(temp)
        temp.rename(JOB/'source')
        (JOB/'registry.toml').write_text(routed_registry(reg.read_text()))
        dump(JOB/'scientific_plan.json',plan())
        dump(JOB/'frozen_manifest.json',digest)
    except Exception:
        if temp.exists():shutil.rmtree(temp)
        raise
    verify();print(f'Task12 frozen source ready: {len(digest)} files')


def verify(job=JOB):
    job=Path(job)
    if load(job/'scientific_plan.json')!=plan():raise ValueError('Scientific plan changed')
    manifest=load(job/'frozen_manifest.json')
    current=source_hashes(job/'source')
    if manifest!=current:
        raise ValueError('Frozen Task12 files differ from the manifest: '+str(set(manifest.items())^set(current.items())))
    if not (job/'registry.toml').is_file():raise FileNotFoundError(job/'registry.toml')
    print(f'Task12 frozen source verified ({len(manifest)} files)')


def task4_args(seed):
    if seed not in SEEDS:raise ValueError('Unknown seed')
    return ['--registry',str(JOB/'registry.toml'),'--suite-id',SUITE,
            '--bootstrap-source-suite-id',BASE,'--require-existing-bootstrap',
            '--training-initialization','warm_start','--strategies',POLICY,
            '--seed',str(seed),'--split-seed','135','--initial-fraction','0.10',
            '--minimum-initial-per-class','5','--acquisition-fraction','0.10',
            '--max-cycles','9','--complete-pool','--max-epochs','50',
            '--patience','10','--batch-size','512','--validation-batch-size','2048',
            '--learning-rate','0.0001','--train-workers','0','--validation-workers','0',
            '--inference-workers','0','--evaluation-workers','0',
            '--assert-excluded-manifest',str(EXCLUDE)]


def preflight(seed=135):
    verify()
    env=dict(os.environ,PYTHONPATH=str(JOB/'source/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    subprocess.run([sys.executable,str(JOB/'source/scripts/run_astrominn_closed_loop.py'),
                    *task4_args(seed)],env=env,check=True)
    print('Task12 preflight passed; no training or stochastic inference')


def audit_matching_split(replay,ds):
    task4=PERSISTENT/'runs'/TASK4/'shared/historical_replay_split'
    def ids(path):
        with open(path,newline='') as f:
            return {row[ds.object_id_column] for row in csv.DictReader(f)}
    if ids(replay.initial_manifest)!=ids(task4/'initial_labeled_manifest.csv'):
        raise ValueError('Task12 L0 does not match Task4 L0')
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    a=CandidatePoolManifest.from_csv(task4/'candidate_pool/candidate_pool.csv')
    b=CandidatePoolManifest.from_csv(replay.candidate_pool_dir/'candidate_pool.csv')
    if tuple(a.object_ids)!=tuple(b.object_ids):
        raise ValueError('Task12 U0 order differs from Task4')


def pilot(seed=135):
    verify()
    if seed not in SEEDS:raise ValueError('Unsupported pilot seed')
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.experiments.historical_replay import materialize_astrominn_historical_replay_split
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import DEFAULT_LEGACY_CODE_DIR
    from al_pipeline.astronomy.analysis.task12_mc_dropout import sample_pool
    from al_pipeline.core.scoring.bald import entropy
    import numpy as np
    out=JOB/'pilot'/f'seed_{seed}'/'result.json'
    if out.exists():print(out.read_text());return
    registry=DatasetRegistry.from_toml(JOB/'registry.toml')
    resolved=registry.resolve_profile('astrominn_ztf');ds=resolved.dataset
    replay=materialize_astrominn_historical_replay_split(
        dataset=ds,destination=PERSISTENT/'runs'/SUITE/'shared/historical_replay_split',
        initial_fraction=.10,minimum_initial_per_class=5,split_seed=135,
        pool_id=SUITE+'_historical_u0',excluded_object_manifests=[EXCLUDE])
    audit_matching_split(replay,ds)
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    manifest=CandidatePoolManifest.from_csv(replay.candidate_pool_dir/'candidate_pool.csv')
    ids=tuple(manifest.object_ids)
    checkpoint=DATA/'runs'/BASE/'shared'/f'seed_{seed}'/'bootstrap_training/best_vit_tower.pth'
    artifact_dir=Path(resolved.profile.options['checkpoint']).parent
    start=time.perf_counter()
    result=sample_pool(pool_path=replay.candidate_pool_dir,object_ids=ids,
                       checkpoint_path=checkpoint,artifact_dir=artifact_dir,
                       device='cuda',samples=20,batch_size=64,seed=seed,cycle=1,
                       legacy_code_dir=DEFAULT_LEGACY_CODE_DIR)
    duration=time.perf_counter()-start
    import math
    k=min(len(ids),math.ceil(load(replay.summary_path)['historical_objects']*.10))
    ix=np.argsort(-result.scores,kind='stable')[:k]
    selected=[ids[int(i)] for i in ix]
    entropy_ref=PERSISTENT/'runs'/TASK4/'entropy'/f'seed_{seed}'/'cycle_001/selection.json'
    if not entropy_ref.is_file():raise FileNotFoundError(entropy_ref)
    reference=load(entropy_ref)['ids']
    if len(reference)!=k:raise ValueError('Entropy reference has different budget')
    overlap=len(set(selected)&set(reference))
    p=result.probabilities.astype(np.float64)
    mean_ent=entropy(p.mean(axis=1));cond_ent=entropy(p).mean(axis=1)
    if not np.allclose(result.scores,mean_ent-cond_ent,atol=1e-7):
        raise RuntimeError('BALD score decomposition failed')
    record={'seed':seed,'selected_ids':selected,'selection_budget':k,
        'candidate_count':len(ids),'posterior_samples':20,
        'dropout_layers':list(result.dropout_layers),
        'max_bald_nats':float(result.scores.max()),
        'mean_bald_nats':float(result.scores.mean()),
        'median_bald_nats':float(np.median(result.scores)),
        'mean_predictive_entropy_nats':float(mean_ent.mean()),
        'mean_conditional_entropy_nats':float(cond_ent.mean()),
        'entropy_overlap_fraction':overlap/k,
        'mc_inference_seconds':duration,
        'nonzero_score_fraction':float(np.mean(result.scores>1e-8)),
        'checkpoint_sha256':sha(checkpoint),
        'initial_manifest_sha256':sha(replay.initial_manifest),
        'task4_L0_U0_equal':True,'all_selected_unique':len(set(selected))==k,
        'posterior_sampling':'coherent_fixed_dropout_masks_across_objects',
        'validation_labels_accessed':False,'test_labels_accessed':False}
    if not record['all_selected_unique']:raise RuntimeError('Repeated acquisition IDs')
    dump(out,record)
    print(json.dumps({key:value for key,value in record.items() if key!='selected_ids'},indent=2))


def pilot_status():
    p=JOB/'pilot/seed_135/result.json'
    if not p.is_file():print('Pilot not complete');return
    d=load(p)
    print(json.dumps({k:v for k,v in d.items() if k!='selected_ids'},indent=2))


def task11_dependency():
    """Only permit Task12 after Task11 success, through explicit SLURM DAG."""
    directory=PERSISTENT/'runs'/TASK11
    all_complete=True
    for policy in ('badge','core_set','fisher_batch'):
        for seed in SEEDS:
            base=directory/policy/f'seed_{seed}'
            state=base/'run_state.json'
            last=load(state).get('last_completed_cycle',0) if state.is_file() else 0
            if int(last)!=9 or not (base/'cycle_009/cycle_summary.json').is_file():
                all_complete=False
    if all_complete:return (), 'Task11 9/9 trajectories already completed'
    task11_job=PERSISTENT/'closed_loop'/TASK11/'submitted.json'
    if not task11_job.is_file():
        raise RuntimeError('Task11 not yet fully completed or submitted; cannot bind Task12 to its final jobs')
    data=load(task11_job)
    entries=data.get('jobs',[])
    finals=[str(v['job_id']) for v in entries if v.get('cycle')==9]
    if len(finals)!=9 or len(set(finals))!=9 or any(not x.isdigit() for x in finals):
        raise ValueError('Task11 submission does not contain all nine final-cycle job IDs')
    return tuple(finals),'afterok on Task11 nine terminal jobs'


def submit(walltime):
    verify()
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d',walltime) or walltime=='00:00:00':
        raise ValueError('Invalid walltime')
    p=JOB/'pilot/seed_135/result.json'
    if not p.is_file():raise RuntimeError('BALD real-model pilot required before 27-fit submission')
    pilot_data=load(p)
    if (not pilot_data['all_selected_unique'] or pilot_data['nonzero_score_fraction']<.01
            or not pilot_data['dropout_layers']):
        raise RuntimeError('Pilot did not establish usable MC dropout variation')
    if (JOB/'submitted.json').exists():raise RuntimeError('Task12 has already been submitted')
    gates, description=task11_dependency()
    dependency='afterok:'+':'.join(gates) if gates else None
    generated=[];jobs=[]
    try:
        for seed in SEEDS:
            prior=None
            for cycle in range(1,10):
                args=['sbatch','--parsable','--hold','--time',walltime,
                      '--job-name',f'T12_bald_{seed}_{cycle}',
                      '--output',str(JOB/'logs/%x_%j.out'),
                      '--error',str(JOB/'logs/%x_%j.err')]
                parent=f'afterok:{prior}' if prior else dependency
                if parent:args+=['--dependency',parent]
                args += [str(JOB/'source/scripts/research/historical_replay/task12/task12_worker.sbatch'),
                         str(JOB),'job',str(seed),str(cycle)]
                response=subprocess.check_output(args,text=True).strip()
                jobid=response.split(';')[0]
                if not jobid.isdigit():raise RuntimeError('Unrecognized sbatch response '+response)
                generated.append(jobid)
                jobs.append({'seed':seed,'cycle':cycle,'job_id':jobid,'dependency':parent})
                prior=jobid
        if len(generated)!=27:raise RuntimeError('Expected 27 fits')
        dump(JOB/'submitted.json',{'jobs':jobs,'dependency':description,'state':'held_awaiting_release'})
        for jobid in reversed(generated):
            subprocess.run(['scontrol','release',jobid],check=True)
        dump(JOB/'submitted.json',{'jobs':jobs,'dependency':description,'state':'released'})
    except Exception:
        if not (JOB/'submitted.json').exists() and generated:
            subprocess.run(['scancel',*generated],check=False)
        raise
    print('Submitted 27 BALD jobs in three 9-cycle chains; '+description)


def status():
    cases=[]
    for seed in SEEDS:
        root=PERSISTENT/'runs'/SUITE/POLICY/f'seed_{seed}'
        state=root/'run_state.json'
        completed=int(load(state).get('last_completed_cycle',0)) if state.exists() else 0
        summaries=sum((root/f'cycle_{i:03d}/cycle_summary.json').is_file() for i in range(1,10))
        cases.append({'seed':seed,'completed_cycles':completed,'cycle_summaries':summaries,'complete':completed==9 and summaries==9})
    print(json.dumps({'suite':SUITE,'completed':sum(c['complete'] for c in cases),
        'planned':3,'cases':cases},indent=2))


def job(seed,cycle,job_dir):
    if seed not in SEEDS or cycle not in range(1,10):raise ValueError('Invalid BALD job')
    verify(job_dir)
    runner=Path(job_dir)/'source/scripts/run_astrominn_closed_loop.py'
    env=dict(os.environ,PYTHONPATH=str(Path(job_dir)/'source/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    subprocess.run([sys.executable,str(runner),*task4_args(seed),
                    '--device','cuda','--cycle',str(cycle),'--run'],env=env,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('plan','prepare','verify','preflight','pilot','pilot-status','submit','status','job'))
    p.add_argument('seed_pos',nargs='?',type=int)
    p.add_argument('cycle_pos',nargs='?',type=int)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--overlay',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--job-dir',type=Path,default=JOB)
    p.add_argument('--seed',type=int,default=135)
    p.add_argument('--walltime',default='01:30:00')
    args=p.parse_args()
    if args.command=='plan':print(json.dumps(plan(),indent=2));return
    if args.command=='prepare':prepare(args.repo,args.overlay);return
    if args.command=='verify':verify(args.job_dir);return
    if args.command=='preflight':preflight(args.seed);return
    if args.command=='pilot':pilot(args.seed);return
    if args.command=='pilot-status':pilot_status();return
    if args.command=='submit':submit(args.walltime);return
    if args.command=='status':status();return
    if args.command=='job':job(args.seed_pos,args.cycle_pos,args.job_dir);return

if __name__=='__main__':main()
