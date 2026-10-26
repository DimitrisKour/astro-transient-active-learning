#!/usr/bin/env python3
"""Task8C.1: CPU Task4 replay until first rejection; GPU retention-only continuations.

prepare, plan, verify, and status submit NO jobs. prefill uses cached Task4
predictions and checkpoints (no training), run executes ONE new GPU cycle.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

P = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
D = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
REFERENCE = P/'closed_loop'/'astrominn_closed_loop_warm_v1'
AUDIT = D/'analysis'/'task8a_retrospective_acceptance_v1'/'first_rejections.csv'
SUITE = 'astrominn_task8c1_retention_macro1pp_v1'
JOB = D/'closed_loop'/SUITE
POLICIES = ('entropy', 'entropy_balanced_explore')
SEEDS = (135, 246, 357)
EXPECTED = {('entropy',135):4, ('entropy',246):4, ('entropy',357):5,
            ('entropy_balanced_explore',135):4, ('entropy_balanced_explore',246):3,
            ('entropy_balanced_explore',357):6}
OVERLAYS = ('src/al_pipeline/core/analysis/__init__.py',
            'src/al_pipeline/core/analysis/retention_gate.py',
            'src/al_pipeline/astronomy/experiments/retention.py',
            'scripts/adaptive/task8c1.py','scripts/adaptive/task8c1_report.py',
            'scripts/adaptive/task8c1_worker_cpu.sbatch',
            'scripts/adaptive/task8c1_worker_gpu.sbatch')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def json_load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+'.new')
    if t.exists():raise RuntimeError('Unresolved prior plan write: '+str(t))
    t.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    t.replace(path)


def cases_from_audit(audit:Path):
    with audit.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    selected=[r for r in rows if r.get('suite_key')=='task4' and r.get('rule')=='macro_tolerance_1pp']
    if len(selected)!=6:raise ValueError(f'Expected six Task4 first rejections; found {len(selected)}')
    out={}
    for r in selected:
        key=(r['policy'],int(r['seed']))
        if key not in EXPECTED or key in out:raise ValueError(f'Unknown or duplicate Task4 trajectory: {key}')
        c=int(r['first_rejected_cycle'])
        if c!=EXPECTED[key] or int(r['last_accepted_cycle'])!=c-1:
            raise ValueError(f'First-rejection audit disagrees for {key}: {c}')
        if str(r.get('fully_accepted_saved_trajectory','')).lower() not in ('false','0'):
            raise ValueError('Unexpected fully-accepted source trajectory')
        out[key]=c
    if set(out)!=set(EXPECTED):raise ValueError('Missing Task4 trajectory')
    return [{'policy':policy,'seed':seed,'first_rejection':out[(policy,seed)],
             'key':f'{policy}_s{seed}'} for policy in POLICIES for seed in SEEDS]


def verify_task4(reference:Path):
    entries=json_load(reference/'source_snapshot.json')
    if not entries:raise ValueError('Empty Task4 source snapshot')
    for rel,expected in entries.items():
        p=reference/'source'/rel
        if not p.is_file() or sha(p)!=expected:raise ValueError('Task4 source changed: '+str(p))
    return sha(reference/'source_snapshot.json')


def routed_registry(reference:Path,data:Path):
    """Route only [storage] roots to HDD, preserving all other TOML entries.

    Dataset and profile tables may legitimately contain their own ``data_root``
    keys. A global regex would rewrite them (or reject a valid registry).
    """
    source=(reference/'registry.toml').read_text(encoding='utf-8')
    original=tomllib.loads(source)
    headers=list(re.finditer(
        r'(?m)^[ \t]*\[[^\]\r\n]+\][ \t]*(?:\#.*)?$', source
    ))
    storage=[i for i,h in enumerate(headers)
             if re.match(r'^[ \t]*\[storage\]',h.group(0))]
    if len(storage)!=1 or 'storage' not in original:
        raise ValueError('Expected exactly one [storage] section in Task4 registry')
    index=storage[0]
    beginning=headers[index].end()
    end=headers[index+1].start() if index+1<len(headers) else len(source)
    section=source[beginning:end]
    for field in ('persistent_root','data_root'):
        pattern=rf'(?m)^([ \t]*{re.escape(field)}[ \t]*=[ \t]*).*$'
        section,count=re.subn(
            pattern,lambda match:match.group(1)+json.dumps(str(data)),section
        )
        if count!=1:
            raise ValueError(f'Unexpected Task4 [storage] field {field} count {count}')
    routed=source[:beginning]+section+source[end:]
    parsed=tomllib.loads(routed)
    # Fail closed if any dataset/profile setting was changed or a root was lost.
    if {k:v for k,v in parsed.items() if k!='storage'} != {
            k:v for k,v in original.items() if k!='storage'}:
        raise ValueError('Registry rewrite modified a non-storage section')
    storage_expected=dict(original['storage'])
    storage_expected.update(persistent_root=str(data),data_root=str(data))
    if parsed['storage']!=storage_expected:
        raise ValueError('Unexpected changes to storage registry fields')
    return routed


def build_plan(reference:Path,audit:Path,data:Path):
    reference_digest=verify_task4(reference)
    cases=cases_from_audit(audit)
    source_split=P/'runs'/'astrominn_closed_loop_warm_v1'/'shared'/'historical_replay_split'
    l0=source_split/'initial_labeled_manifest.csv'
    pool=source_split/'candidate_pool'/'candidate_pool.csv'
    labels=source_split/'hidden_labels.csv'
    for path in (l0,pool,labels):
        if not path.is_file():raise FileNotFoundError(path)
    for row in cases:
        policy,seed,r=row['policy'],row['seed'],row['first_rejection']
        for c in (1,r):
            s=P/'runs'/'astrominn_closed_loop_warm_v1'/policy/f'seed_{seed}'/f'cycle_{c:03d}'/'cycle_summary.json'
            if not s.is_file():raise FileNotFoundError(s)
        boot=D/'runs'/'astrominn_historical_replay_v1'/'shared'/f'seed_{seed}'/'bootstrap_training'/'best_vit_tower.pth'
        if not boot.is_file():raise FileNotFoundError(boot)
        row['bootstrap_checkpoint']=str(boot)
        row['bootstrap_sha256']=sha(boot)
    return {'schema':'task8c1_retention_v1','suite':SUITE,'cases':cases,
        'source_suite':'astrominn_closed_loop_warm_v1',
        'reference':str(reference),'reference_snapshot_sha256':reference_digest,
        'reference_registry_sha256':sha(reference/'registry.toml'),
        'audit':str(audit),'audit_sha256':sha(audit),'data':str(data),
        'source_l0':str(l0),'l0_sha256':sha(l0),
        'source_pool':str(pool),'pool_sha256':sha(pool),
        'source_hidden_labels':str(labels),'hidden_labels_sha256':sha(labels),
        'active_gate':{'macro_f1_tolerance':0.01},
        'shadow_gate':{'balanced_accuracy_loss_tolerance':0.03,
                       'net_correct_losses':{'AGN':35,'TDE':3,'SN I':100,'SN II':30,'CV':10}},
        'training':{'max_epochs':50,'patience':10,'class_weight_exponent':0.85,
                    'weighting':'per_sample','checkpoint_selection':'weighted_loss_best',
                    'expert_towers':'frozen','initialization':'last_accepted_checkpoint',
                    'test_used_for_selection':False},
        'first_rejection_replay':True,'full_cycles':9,
        'planned_new_gpu_cycles':sum(9-c['first_rejection'] for c in cases),
        'submissions':False}


def prepare(repo:Path,job:Path,reference:Path,audit:Path,data:Path):
    repo=repo.resolve()
    if data.resolve() not in job.resolve().parents:raise ValueError('Task8C1 state must be on HDD')
    plan=build_plan(reference,audit,data)
    content={str(p.relative_to(reference/'source')):p.read_bytes()
             for p in (reference/'source'/'src').rglob('*.py')}
    for rel in OVERLAYS:
        p=repo/rel
        if not p.is_file():raise FileNotFoundError(p)
        if rel in content and content[rel]!=p.read_bytes():
            raise ValueError(f'Task8C1 overlay would change frozen Task4 code: {rel}')
        content[rel]=p.read_bytes()
    hashes={rel:hashlib.sha256(raw).hexdigest() for rel,raw in content.items()}
    registry=routed_registry(reference,data)
    if (job/'plan.json').exists():
        if (json_load(job/'plan.json')!=plan or
            json_load(job/'source_snapshot.json')!=hashes or
            (job/'registry.toml').read_text()!=registry):
            raise ValueError('Existing frozen Task8C1 scientific plan differs')
        verify(job,deep=True)
        return plan
    if (job/'source').exists() or (job/'registry.toml').exists():
        raise RuntimeError('Partial frozen Task8C1 snapshot: inspect before retrying')
    for rel,blob in content.items():
        target=job/'source'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(blob)
    (job/'logs').mkdir(parents=True,exist_ok=True)
    (job/'registry.toml').write_text(registry,encoding='utf-8')
    atomic_json(job/'source_snapshot.json',hashes)
    atomic_json(job/'plan.json',plan)
    verify(job,deep=True)
    return plan


def verify(job:Path,deep=False):
    plan=json_load(job/'plan.json')
    if (plan.get('schema')!='task8c1_retention_v1' or plan.get('suite')!=SUITE or
        len(plan.get('cases',()))!=6 or plan.get('planned_new_gpu_cycles')!=28):
        raise ValueError('Invalid Task8C1 planned identity')
    hashes=json_load(job/'source_snapshot.json')
    for rel,expected in hashes.items():
        p=job/'source'/rel
        if not p.is_file() or sha(p)!=expected:raise ValueError('Frozen 8C1 file modified: '+str(p))
    if sha(Path(plan['audit']))!=plan['audit_sha256'] or sha(Path(plan['source_l0']))!=plan['l0_sha256'] or sha(Path(plan['source_pool']))!=plan['pool_sha256'] or sha(Path(plan['source_hidden_labels']))!=plan['hidden_labels_sha256']:
        raise ValueError('Reference input checksum mismatch')
    if sha(Path(plan['reference'])/'registry.toml')!=plan['reference_registry_sha256']:
        raise ValueError('Task4 registry mutated')
    if routed_registry(Path(plan['reference']),Path(plan['data']))!=(job/'registry.toml').read_text():
        raise ValueError('Frozen registry mismatch')
    if deep:
        if verify_task4(Path(plan['reference']))!=plan['reference_snapshot_sha256']:
            raise ValueError('Task4 source snapshot changed')
        rebuilt=build_plan(Path(plan['reference']),Path(plan['audit']),Path(plan['data']))
        if rebuilt!=plan:
            raise ValueError('Frozen scientific plan does not match audited sources')
    return plan


def experiment_for(job:Path,case_idx:int):
    plan=verify(job)
    case=plan['cases'][case_idx]
    # Heavy imports performed only at run time, from the *frozen* Task4 snapshot.
    import torch
    from al_pipeline.astronomy.experiments.astrominn import build_astrominn_simulated_suite
    from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainingConfig
    from al_pipeline.astronomy.astrominn.evaluation import AstroMiNNViTEvaluator
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.core.experiments.batch_policy import BatchAcquisitionConfig
    from al_pipeline.astronomy.experiments.retention import RetentionExperiment
    policy_name,seed=case['policy'],case['seed']
    source_run=D/'runs'/'astrominn_closed_loop_warm_v1'/policy_name/f'seed_{seed}'
    source_training=json_load(source_run/'cycle_001'/'training'/'training_summary.json')
    train_policy=AstroMiNNViTTrainingConfig(**source_training['policy'])
    if (train_policy.loader_seed!=seed or train_policy.class_weight_exponent!=.85 or
        train_policy.new_object_weighting!='per_sample' or
        train_policy.class_weight_source!='training_manifest' or train_policy.overwrite or
        train_policy.max_epochs!=50 or train_policy.patience!=10):
        raise ValueError('Training policy differs from Task4')
    reference_registry=Path(plan['reference'])/'registry.toml'
    reference_ds=DatasetRegistry.from_toml(reference_registry).resolve_profile('astrominn_ztf')
    artifact=Path(reference_ds.profile.options['checkpoint']).parent
    suite=build_astrominn_simulated_suite(
        suite_id=SUITE,registry_path=job/'registry.toml',
        pool_path=Path(plan['source_pool']).parent,
        hidden_labels_path=Path(plan['source_hidden_labels']),
        training_policy=train_policy,profile_name='astrominn_ztf',
        strategies=(policy_name,),acquisition_size=473,max_cycles=9,
        allow_partial_final_cycle=True,
        batch_policy=BatchAcquisitionConfig(shortlist_multiplier=3,
            balance_alpha=1.0,balance_beta=.5,balance_strength=1.0,
            balance_cap=3.0,exploration_fraction=.2),
        seed=seed,training_initialization='warm_start',holdout_ids=(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        inference_num_workers=0,evaluation_num_workers=0,
        baseline_checkpoint=Path(case['bootstrap_checkpoint']),
        training_manifest_override=Path(plan['source_l0']),
        artifact_dir=artifact)
    orig=suite.experiments[policy_name]
    orig.evaluator=AstroMiNNViTEvaluator(dataset=orig.dataset,
        manifests={'validation':orig.dataset.validation_manifest},
        device=orig.trainer.device,batch_size=2048,num_workers=0,
        artifact_dir=artifact)
    return RetentionExperiment.from_experiment(orig,
        reference_persistent=P,reference_data=D)


def prefill(job:Path,index:int):
    plan=verify(job,deep=True)
    case=plan['cases'][index]
    exp=experiment_for(job,index)
    exp.evaluate_baseline()
    for c in range(1,case['first_rejection']+1):
        state=exp.recover_completed_decisions()
        if state.last_completed_cycle >= c:continue
        if state.last_completed_cycle!=c-1:
            raise RuntimeError('Prefill cycle order wrong')
        if not exp._can_reference_replay(c):
            raise RuntimeError('CPU prefill refused: cycle requires non-reference training')
        # Hard safety boundary: even a bug in reference detection cannot start training.
        def prohibit_training(*args, **kwargs):
            raise RuntimeError('CPU prefill must never call the training backend')
        exp.trainer.train = prohibit_training
        exp.run_next_cycle()
        decision=json_load(exp._decision_path(c))
        if decision['accepted'] != (c < case['first_rejection']):
            raise ValueError(f'Frozen Task8A first-rejection mismatch: {case} cycle {c}')
        if not decision['reference_reused']:
            raise RuntimeError('Prefill unexpectedly required new GPU training')
    print(json.dumps({'prefilled':case['key'], 'last_completed_cycle':case['first_rejection'],
                      'first_rejection_verified':True}))


def run_gpu(job:Path,index:int,cycle:int):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Task8C.1 GPU training can run only inside SLURM')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU training fallback')
    plan=verify(job,deep=True)
    case=plan['cases'][index]
    if not case['first_rejection'] < cycle <= 9:
        raise ValueError('GPU cycle must follow read-only first-rejection replay')
    exp=experiment_for(job,index)
    state=exp.recover_completed_decisions()
    if state.last_completed_cycle >= cycle:
        dec=exp._decision_path(cycle)
        if not dec.is_file():raise ValueError('Completed cycle missing retention decision')
        print('Already completed',case['key'],cycle)
        return
    if state.last_completed_cycle != cycle-1 or state.active_cycle not in (None,cycle):
        raise RuntimeError('Previous dependent cycle has not completed')
    exp.run_next_cycle()
    print(json.dumps({'completed':case['key'],'cycle':cycle,
                      'decision':json_load(exp._decision_path(cycle))['accepted']}))


def status(job:Path):
    plan=verify(job)
    rows=[]
    for case in plan['cases']:
        run=Path(plan['data'])/'runs'/SUITE/case['policy']/f"seed_{case['seed']}"
        status_path=run/'run_state.json'
        stat=json_load(status_path) if status_path.is_file() else {}
        completed=int(stat.get('last_completed_cycle',0))
        decisions=[read for c in range(1,completed+1)
                   if (read:=run/f'cycle_{c:03d}'/'retention_decision.json').is_file()]
        rejected=[c for c in range(1,completed+1)
                  if (run/f'cycle_{c:03d}'/'retention_decision.json').is_file() and
                  not json_load(run/f'cycle_{c:03d}'/'retention_decision.json')['accepted']]
        rows.append({'case':case['key'],'completed':completed,
                     'decisions':len(decisions),'rejected_cycles':rejected,'active':stat.get('active_cycle')})
    return {'cases':rows,'completed_trajectories':sum(x['completed']==9 and x['decisions']==9 for x in rows)}


def submit_prefill(job:Path,walltime:str):
    plan=verify(job,deep=True)
    if (job/'prefill_submitted.json').exists():raise RuntimeError('Prefill already submitted')
    command=['sbatch','--parsable',f'--time={walltime}', '--array=0-5',
       f'--output={job}/logs/T8C1_pf_%A_%a.out',f'--error={job}/logs/T8C1_pf_%A_%a.err',
       str(job/'source/scripts/adaptive/task8c1_worker_cpu.sbatch'),str(job)]
    jid=subprocess.check_output(command,text=True).strip().split(';')[0]
    if not jid.isdigit():raise RuntimeError('Unexpected SLURM job id '+jid)
    atomic_json(job/'prefill_submitted.json',{'job_id':jid,'command':command})
    print('Prefill CPU array submitted:',jid)


def submit_gpu(job:Path,walltime:str):
    plan=verify(job,deep=True)
    if (job/'gpu_submitted.json').exists() or (job/'gpu_submission_pending.json').exists():
        raise RuntimeError('GPU experiment already submitted or incomplete submission requires audit')
    checks=status(job)['cases']
    for case,st in zip(plan['cases'],checks):
        if st['completed']!=case['first_rejection'] or st['decisions']!=case['first_rejection'] or st['rejected_cycles']!=[case['first_rejection']]:
            raise RuntimeError(f'Read-only prefill has not passed: {case["key"]} => {st}')
    records=[]
    pending=job/'gpu_submission_pending.json'
    try:
        for index,case in enumerate(plan['cases']):
            parent=None
            for cycle in range(case['first_rejection']+1,10):
                name=f'T8C1_{index}_c{cycle}'
                cmd=['sbatch','--parsable','--hold',f'--time={walltime}',f'--job-name={name}',
                     f'--output={job}/logs/%x_%j.out',f'--error={job}/logs/%x_%j.err']
                if parent:cmd.append(f'--dependency=afterok:{parent}')
                cmd.extend([str(job/'source/scripts/adaptive/task8c1_worker_gpu.sbatch'),
                            str(job),str(index),str(cycle)])
                jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
                if not jid.isdigit():raise RuntimeError(f'Invalid job ID {jid!r}')
                records.append({'case':case['key'],'cycle':cycle,'job_id':jid,'parent':parent})
                pending.parent.mkdir(parents=True,exist_ok=True)
                atomic_json(pending,{'status':'held','jobs':records})
                parent=jid
        # Release reverse dependency order to avoid triggering child jobs early.
        for r in reversed(records):subprocess.run(['scontrol','release',r['job_id']],check=True)
        atomic_json(job/'gpu_submitted.json',{'status':'released','jobs':records})
        pending.unlink()
    except BaseException:
        if records:subprocess.run(['scancel',*[r['job_id'] for r in records]],check=False)
        atomic_json(job/'gpu_submission_failed.json',{'jobs':records,'cancellation_requested':True})
        raise
    print('Submitted',len(records),'GPU jobs in six dependent chains')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan','prepare','verify','prefill','run','status','submit-prefill','submit'))
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--job-dir',type=Path,default=JOB)
    p.add_argument('--reference',type=Path,default=REFERENCE)
    p.add_argument('--audit',type=Path,default=AUDIT)
    p.add_argument('--data',type=Path,default=D)
    p.add_argument('--index',type=int)
    p.add_argument('--cycle',type=int)
    p.add_argument('--walltime',default='00:20:00')
    args=p.parse_args()
    if args.command=='plan':
        plan=build_plan(args.reference,args.audit,args.data)
        print(json.dumps({'cases':plan['cases'],'new_GPU_cycles':plan['planned_new_gpu_cycles']},indent=2))
    elif args.command=='prepare':
        prepare(args.repo,args.job_dir,args.reference,args.audit,args.data)
        print('Frozen 8C.1 prepared; NO jobs submitted')
    elif args.command=='verify':
        verify(args.job_dir,deep=True)
        print('Frozen source, input hashes, and macro-F1 rule verified')
    elif args.command=='prefill':
        if args.index is None:p.error('--index required')
        prefill(args.job_dir,args.index)
    elif args.command=='run':
        if args.index is None or args.cycle is None:p.error('--index and --cycle required')
        run_gpu(args.job_dir,args.index,args.cycle)
    elif args.command=='status':print(json.dumps(status(args.job_dir),indent=2))
    elif args.command in ('submit-prefill','submit'):
        if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d',args.walltime):p.error('Invalid time HH:MM:SS')
        if args.command=='submit-prefill':submit_prefill(args.job_dir,args.walltime)
        else:submit_gpu(args.job_dir,args.walltime)

if __name__=='__main__':main()
