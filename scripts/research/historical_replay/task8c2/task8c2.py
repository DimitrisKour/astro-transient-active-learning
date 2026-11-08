#!/usr/bin/env python3
"""Task8C.2: CPU Task4 replay before first rejection; GPU bounded-retry continuations.

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
SUITE = 'astrominn_task8c2_retry4_macro1pp_v1'
JOB = D/'closed_loop'/SUITE
POLICIES = ('entropy', 'entropy_balanced_explore')
SEEDS = (135, 246, 357)
EXPECTED = {('entropy',135):4, ('entropy',246):4, ('entropy',357):5,
            ('entropy_balanced_explore',135):4, ('entropy_balanced_explore',246):3,
            ('entropy_balanced_explore',357):6}
OVERLAYS = ('src/al_pipeline/core/analysis/__init__.py',
            'src/al_pipeline/core/analysis/retention_gate.py',
            'src/al_pipeline/astronomy/experiments/retention.py',
            'src/al_pipeline/astronomy/experiments/retention_retry.py',
            'scripts/research/historical_replay/task8c2/task8c2.py','scripts/research/historical_replay/task8c2/task8c2_report.py',
            'scripts/research/historical_replay/task8c2/task8c2_worker_cpu.sbatch',
            'scripts/research/historical_replay/task8c2/task8c2_worker_gpu.sbatch')


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
    return {'schema':'task8c2_retry4_v1','suite':SUITE,'cases':cases,
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
                    'retry_initialization':'same_incumbent_checkpoint',
                    'test_used_for_selection':False},
        'first_rejection_replay':True,'full_cycles':9,
        'planned_new_gpu_cycles':sum(9-c['first_rejection'] for c in cases),
        'retry_seeds':[1001,1002,1003,1004],
        'planned_gpu_job_slots':sum(4+5*(9-c['first_rejection']) for c in cases),
        'submissions':False}


def prepare(repo:Path,job:Path,reference:Path,audit:Path,data:Path):
    repo=repo.resolve()
    if data.resolve() not in job.resolve().parents:raise ValueError('Task8C2 state must be on HDD')
    plan=build_plan(reference,audit,data)
    content={str(p.relative_to(reference/'source')):p.read_bytes()
             for p in (reference/'source'/'src').rglob('*.py')}
    for rel in OVERLAYS:
        p=repo/rel
        if not p.is_file():raise FileNotFoundError(p)
        if rel in content and content[rel]!=p.read_bytes():
            raise ValueError(f'Task8C2 overlay would change frozen Task4 code: {rel}')
        content[rel]=p.read_bytes()
    hashes={rel:hashlib.sha256(raw).hexdigest() for rel,raw in content.items()}
    registry=routed_registry(reference,data)
    if (job/'plan.json').exists():
        if (json_load(job/'plan.json')!=plan or
            json_load(job/'source_snapshot.json')!=hashes or
            (job/'registry.toml').read_text()!=registry):
            raise ValueError('Existing frozen Task8C2 scientific plan differs')
        verify(job,deep=True)
        return plan
    if (job/'source').exists() or (job/'registry.toml').exists():
        raise RuntimeError('Partial frozen Task8C2 snapshot: inspect before retrying')
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
    if (plan.get('schema')!='task8c2_retry4_v1' or plan.get('suite')!=SUITE or
        len(plan.get('cases',()))!=6 or plan.get('planned_new_gpu_cycles')!=28):
        raise ValueError('Invalid Task8C2 planned identity')
    hashes=json_load(job/'source_snapshot.json')
    for rel,expected in hashes.items():
        p=job/'source'/rel
        if not p.is_file() or sha(p)!=expected:raise ValueError('Frozen 8C2 file modified: '+str(p))
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
    from al_pipeline.astronomy.experiments.retention_retry import RetryRetentionExperiment
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
    return RetryRetentionExperiment.from_experiment(orig,
        reference_persistent=P,reference_data=D)


def prefill(job:Path,index:int):
    plan=verify(job,deep=True)
    case=plan['cases'][index]
    first=case['first_rejection']
    exp=experiment_for(job,index)
    exp.evaluate_baseline()
    for cycle in range(1,first+1):
        state=exp.initialize()
        if state.last_completed_cycle>=cycle:
            if cycle<first:
                row=exp.check_final(cycle)
                if not row['accepted'] or not row['reference_reused']:
                    raise RuntimeError('Cached CPU prefill decision is not accepted Task4')
            else:
                base=json_load(exp._base_path(cycle))
                if base['comparison']['accepted'] or not base['reference_reused'] or exp._decision_path(cycle).exists():
                    raise RuntimeError('First rejected original checkpoint should remain pending retry')
            continue
        if state.last_completed_cycle!=cycle-1 or not exp._can_reference_replay(cycle):
            raise RuntimeError('CPU prefill out of order or unable to reuse Task4')
        def prohibit_training(*args,**kwargs):
            raise RuntimeError('CPU prefill must never start model training')
        exp.trainer.train=prohibit_training
        base=exp.base_stage(cycle)
        if not base['reference_reused'] or base['comparison']['accepted']!=(cycle<first):
            raise RuntimeError('Task8A first-rejection replay not reproduced')
        if cycle<first:
            if not exp.check_final(cycle)['accepted']:
                raise RuntimeError('Unexpected historical pre-rejection checkpoint rejection')
        elif exp._decision_path(cycle).exists():
            raise RuntimeError('First rejected checkpoint must not be finalized before retries')
    print(json.dumps({'prefilled':case['key'], 'last_completed_cycle':first,
                      'first_rejected_candidate_pending_retries':True}))


def run_gpu(job:Path,index:int,cycle:int,stage:str,retry_number:int|None):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Task8C.2 GPU stages run only inside SLURM')
    plan=verify(job,deep=True)
    case=plan['cases'][index]
    if not case['first_rejection'] <= cycle <= 9:
        raise ValueError('GPU cycle outside divergent closed-loop phase')
    if stage not in ('base','retry') or (stage=='base' and retry_number is not None) or (stage=='retry' and retry_number not in (1,2,3,4)):
        raise ValueError('Invalid GPU stage/index combination')
    if stage=='retry':
        final=Path(plan['data'])/'runs'/SUITE/case['policy']/f"seed_{case['seed']}"/f'cycle_{cycle:03d}'/'retention_decision.json'
        if final.is_file():
            # A finalized decision guarantees no further training is required.
            # Verify its identity and checkpoint content without loading PyTorch.
            row=json_load(final)
            if row['schema']!='task8c2_bounded_retry_v1' or row['policy']!=case['policy'] or row['seed']!=case['seed'] or row['cycle']!=cycle:
                raise ValueError('Existing final decision has the wrong identity')
            if sha(Path(row['incumbent_checkpoint'])) != row['incumbent_sha256']:
                raise ValueError('Final incumbent checkpoint hash changed')
            if sha(Path(row['training_manifest'])) != row['training_manifest_sha256']:
                raise ValueError('Final training manifest changed')
            print(json.dumps({'case':case['key'],'cycle':cycle,'stage':'retry',
                              'attempt':retry_number,'outcome':'skipped_already_finalized'}))
            return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; cannot use CPU fallback')
    exp=experiment_for(job,index)
    if stage=='base':
        result=exp.base_stage(cycle)
        print(json.dumps({'cycle':cycle,'case':case['key'],'stage':'base',
                          'accepted':result['comparison']['accepted'],
                          'reference_reused':result['reference_reused']}))
    else:
        result=exp.retry_stage(cycle,retry_number)
        print(json.dumps({'cycle':cycle,'case':case['key'],'stage':'retry',
                          'attempt':retry_number,'outcome':result.get('skipped','trained'),
                          'accepted':result.get('comparison',{}).get('accepted')}))


def status(job:Path):
    plan=verify(job)
    rows=[]
    for case in plan['cases']:
        run=Path(plan['data'])/'runs'/SUITE/case['policy']/f"seed_{case['seed']}"
        state_path=run/'run_state.json'
        stat=json_load(state_path) if state_path.is_file() else {}
        completed=int(stat.get('last_completed_cycle',0))
        finalized=[c for c in range(1,completed+1)
                   if (run/f'cycle_{c:03d}'/'retention_decision.json').is_file()]
        rejected=[c for c in finalized if not json_load(run/f'cycle_{c:03d}'/'retention_decision.json')['accepted']]
        retry_accepted=[c for c in finalized if json_load(run/f'cycle_{c:03d}'/'retention_decision.json')['accepted_retry_seed'] is not None]
        executed_retries=sum(sum((run/f'cycle_{c:03d}'/'retry_attempts'/f'retry_{a}.json').is_file()
                                 for a in (1,2,3,4)) for c in range(1,completed+1))
        rows.append({'case':case['key'],'completed':completed,'decisions':len(finalized),
                     'rejected_cycles':rejected,'retry_accepted_cycles':retry_accepted,
                     'trained_retries':executed_retries,'active':stat.get('active_cycle'),
                     'incomplete_decision_cycles':[c for c in range(1,completed+1) if c not in finalized]})
    return {'cases':rows,'completed_trajectories':sum(x['completed']==9 and x['decisions']==9 for x in rows)}


def submit_prefill(job:Path,walltime:str):
    verify(job,deep=True)
    if (job/'prefill_submitted.json').exists():raise RuntimeError('CPU prefill already submitted')
    cmd=['sbatch','--parsable',f'--time={walltime}', '--array=0-5',
         f'--output={job}/logs/T8C2_pf_%A_%a.out',
         f'--error={job}/logs/T8C2_pf_%A_%a.err',
         str(job/'source/scripts/research/historical_replay/task8c2/task8c2_worker_cpu.sbatch'),str(job)]
    jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not jid.isdigit():raise RuntimeError('Unexpected CPU Slurm job ID '+jid)
    atomic_json(job/'prefill_submitted.json',{'job_id':jid,'command':cmd})
    print('Submitted CPU prefill array:',jid)


def stage_plan(cases, indices):
    """Build all attempts in six deterministic chains; no speculative retry fits."""
    jobs=[]
    for index in indices:
        case=cases[index]
        for cycle in range(case['first_rejection'],10):
            if cycle>case['first_rejection']:
                jobs.append({'case_index':index,'case':case['key'],'cycle':cycle,
                             'stage':'base','retry_number':None})
            for number in (1,2,3,4):
                jobs.append({'case_index':index,'case':case['key'],'cycle':cycle,
                             'stage':'retry','retry_number':number})
    return jobs


def submit_gpu(job:Path,walltime:str,case_index:int|None=None):
    plan=verify(job,deep=True)
    check=status(job)['cases']
    indices=[case_index] if case_index is not None else list(range(6))
    if any(i not in range(6) for i in indices):raise ValueError('Invalid case-index')
    for i in indices:
        row,st=plan['cases'][i],check[i]
        first=row['first_rejection']
        if (st['completed']!=first or st['decisions']!=first-1 or
            st['rejected_cycles'] or st['incomplete_decision_cycles']!=[first]):
            raise RuntimeError(f'CPU first rejection has not been verified: {row["key"]}: {st}')
        basedir=Path(plan['data'])/'runs'/SUITE/row['policy']/f"seed_{row['seed']}"/f'cycle_{first:03d}'
        base=json_load(basedir/'base_attempt.json')
        if base['comparison']['accepted'] or not base['reference_reused']:
            raise RuntimeError('Original first rejected candidate not reproduced')
    for i in indices:
        if (job/f'gpu_submitted_case_{i}.json').exists() or (job/f'gpu_submission_pending_case_{i}.json').exists() or (job/f'gpu_submission_failed_case_{i}.json').exists():
            raise RuntimeError(f'GPU case {i} has an existing submission marker; do not duplicate')
    for i in indices:
        records=[]
        pending=job/f'gpu_submission_pending_case_{i}.json'
        parent=None
        try:
            for row in stage_plan(plan['cases'],[i]):
                label='b' if row['stage']=='base' else f'r{row["retry_number"]}'
                name=f'T8C2_{i}_c{row["cycle"]}_{label}'
                cmd=['sbatch','--parsable','--hold',f'--time={walltime}',f'--job-name={name}',
                     f'--output={job}/logs/%x_%j.out',f'--error={job}/logs/%x_%j.err']
                if parent:cmd.append(f'--dependency=afterok:{parent}')
                cmd.extend([str(job/'source/scripts/research/historical_replay/task8c2/task8c2_worker_gpu.sbatch'),
                            str(job),str(i),str(row['cycle']),row['stage'],
                            str(row['retry_number'] or 0)])
                jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
                if not jid.isdigit():raise RuntimeError(f'Invalid Slurm job ID {jid!r}')
                records.append({**row,'job_id':jid,'parent':parent})
                atomic_json(pending,{'status':'held','jobs':records})
                parent=jid
            for row in reversed(records):
                subprocess.run(['scontrol','release',row['job_id']],check=True)
            atomic_json(job/f'gpu_submitted_case_{i}.json',{'status':'released','jobs':records})
            pending.unlink()
        except BaseException:
            if records:subprocess.run(['scancel',*[x['job_id'] for x in records]],check=False)
            atomic_json(job/f'gpu_submission_failed_case_{i}.json',
                        {'jobs':records,'cancellation_requested':True})
            raise
        print('Submitted',len(records),'staged jobs for',plan['cases'][i]['key'],
              '(skipped retry stages consume no training GPU time)')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','prepare','verify','prefill','run','status','submit-prefill','submit'))
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--job-dir',type=Path,default=JOB)
    p.add_argument('--reference',type=Path,default=REFERENCE)
    p.add_argument('--audit',type=Path,default=AUDIT)
    p.add_argument('--data',type=Path,default=D)
    p.add_argument('--index',type=int)
    p.add_argument('--cycle',type=int)
    p.add_argument('--stage',choices=('base','retry'))
    p.add_argument('--retry-number',type=int)
    p.add_argument('--case-index',type=int)
    p.add_argument('--walltime',default='00:20:00')
    args=p.parse_args()
    if args.command=='plan':
        plan=build_plan(args.reference,args.audit,args.data)
        print(json.dumps({'cases':plan['cases'],
          'base_cycles_to_execute':plan['planned_new_gpu_cycles'],
          'conditional_retry_stage_slots':plan['planned_gpu_job_slots']-plan['planned_new_gpu_cycles'],
          'total_slurm_job_slots':plan['planned_gpu_job_slots']},indent=2))
    elif args.command=='prepare':
        prepare(args.repo,args.job_dir,args.reference,args.audit,args.data)
        print('Frozen Task8C.2 prepared; NO Slurm jobs submitted')
    elif args.command=='verify':
        verify(args.job_dir,deep=True)
        print('Frozen Task8C.2 sources and inputs verified')
    elif args.command=='prefill':
        if args.index is None:p.error('--index required')
        prefill(args.job_dir,args.index)
    elif args.command=='run':
        if args.index is None or args.cycle is None or args.stage is None:
            p.error('--index, --cycle and --stage required')
        run_gpu(args.job_dir,args.index,args.cycle,args.stage,args.retry_number)
    elif args.command=='status':print(json.dumps(status(args.job_dir),indent=2))
    else:
        if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d',args.walltime):
            p.error('Invalid time HH:MM:SS')
        if args.command=='submit-prefill':submit_prefill(args.job_dir,args.walltime)
        else:submit_gpu(args.job_dir,args.walltime,args.case_index)

if __name__=='__main__':main()
