#!/usr/bin/env python3
"""Task9B isolated ten-epoch SWA continuations on six original Task4 first rejections.

plan/prepare/verify/status/report do not submit SLURM or train. submit is explicit.
Training is validation-only and never changes any active-learning trajectory.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

B=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/recovery/astrominn_task8b_first_rejection_recovery_v1')
ROOT=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task9b_swa_continuations_v1')
OVERLAYS=('src/al_pipeline/astronomy/analysis/swa_continuation.py',
          'scripts/research/historical_replay/task9/task9b_swa.py','scripts/research/historical_replay/task9/task9b_swa_worker.sbatch')


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def atomic_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.new')
    if temp.exists():raise RuntimeError(f'Unresolved earlier temp write: {temp}')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8');temp.replace(p)


def check_task8b(b:Path,full=False):
    plan=read_json(b/'plan.json')
    if (plan.get('schema')!='task8b_first_rejection_v1' or len(plan.get('cases',{}))!=6 or
        len(plan.get('jobs',[]))!=30):raise ValueError('Invalid Task8B research plan')
    if hash_file(Path(plan['source_audit']))!=plan['source_audit_sha256']:
        raise ValueError('Task8A first-rejection audit changed')
    if full:
        snap=read_json(b/'source_snapshot.json')
        for rel,digest in snap.items():
            if hash_file(b/'source'/rel)!=digest:raise ValueError('Task8B frozen source changed: '+rel)
        if hash_file(b/'registry.toml')!=plan['reference_registry_sha256']:
            raise ValueError('Task8B registry changed')
    for inp in plan['cases'].values():
        for path,digest in ((inp['candidate_checkpoint'],inp['candidate_checkpoint_sha256']),
                            (inp['training_manifest'],inp['manifest_sha256']),
                            (inp['validation_manifest'],inp['validation_sha256']),
                            (inp['source_train_summary'],inp['source_training_summary_sha256'])):
            if hash_file(path)!=digest:raise ValueError('Source artifact changed: '+str(path))
    return plan


def make_plan(b:Path,root:Path):
    source=check_task8b(b,full=True)
    if root.resolve()==b.resolve() or root.resolve().is_relative_to(b.resolve()):
        raise ValueError('New SWA output must be independent from Task8B artifacts')
    cases=[]
    for key,inp in sorted(source['cases'].items()):
        cc=inp['case']
        cases.append({'key':key,'policy':cc['policy'],'seed':cc['seed'],'rejected_cycle':cc['rejected_cycle'],
                      'source':inp,'output_dir':str(root/'fits'/key)})
    return {'schema':'task9b_swa_v1','suite':'task9b_swa_continuations_v1','cases':cases,
        'task8b_root':str(b),'task8b_plan_sha256':hash_file(b/'plan.json'),
        'task8b_snapshot_sha256':hash_file(b/'source_snapshot.json'),
        'task8b_registry_sha256':hash_file(b/'registry.toml'),
        'experimental_policy':{'epochs':10,'average_epochs':[6,7,8,9,10],
                               'constant_learning_rate':2e-5,'training_seed':'original_trajectory_seed',
                               'optimizer':'fresh AdamW with original Task4 beta, eps, decay, clip',
                               'new_labels':False,'reacquisition':False,'test_data':False},
        'gpu_fits':6,'submitted':False}


def prepare(repo:Path,b:Path,root:Path):
    plan=make_plan(b,root)
    src=b/'source'/'src'
    content={str(p.relative_to(b/'source')):p.read_bytes() for p in src.rglob('*.py')}
    for rel in OVERLAYS:
        file=repo/rel
        if not file.is_file():raise FileNotFoundError(file)
        if rel in content and content[rel]!=file.read_bytes():
            raise ValueError('SWA overlay would modify frozen Task4 source: '+rel)
        content[rel]=file.read_bytes()
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in content.items()}
    if (root/'plan.json').exists():
        if read_json(root/'plan.json')!=plan or read_json(root/'source_snapshot.json')!=hashes:
            raise ValueError('Existing SWA plan/source mismatch')
        verify(root,deep=True);return plan
    if (root/'source').exists() or (root/'source_snapshot.json').exists():
        raise RuntimeError('Partial SWA frozen source exists; inspect manually')
    for rel,raw in content.items():
        path=root/'source'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (root/'logs').mkdir(parents=True,exist_ok=True)
    shutil.copy2(b/'registry.toml',root/'registry.toml')
    atomic_json(root/'source_snapshot.json',hashes)
    atomic_json(root/'plan.json',plan)
    verify(root,deep=True)
    return plan


def verify(root:Path,deep=False):
    plan=read_json(root/'plan.json')
    if plan['schema']!='task9b_swa_v1' or len(plan['cases'])!=6 or plan['gpu_fits']!=6:
        raise ValueError('Invalid SWA experimental plan')
    if (hash_file(root/'registry.toml')!=plan['task8b_registry_sha256'] or
        hash_file(Path(plan['task8b_root'])/'plan.json')!=plan['task8b_plan_sha256'] or
        hash_file(Path(plan['task8b_root'])/'source_snapshot.json')!=plan['task8b_snapshot_sha256']):
        raise ValueError('Task8B reference changed')
    for rel,digest in read_json(root/'source_snapshot.json').items():
        if hash_file(root/'source'/rel)!=digest:
            raise ValueError('SWA frozen source modified: '+rel)
    if deep:
        source=check_task8b(Path(plan['task8b_root']),full=True)
        if source['cases']!={c['key']:c['source'] for c in plan['cases']}:
            raise ValueError('SWA source cases no longer match Task8B')
        if (plan['experimental_policy']['average_epochs']!=[6,7,8,9,10] or
            plan['experimental_policy']['constant_learning_rate']!=2e-5):
            raise ValueError('Scientific continuation policy changed')
    return plan


def one(root:Path,index:int):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('SWA training requires an explicit Slurm GPU allocation')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('SWA training requires CUDA')
    plan=verify(root,deep=True)
    if index not in range(6):raise ValueError('Case index must be 0..5')
    case=plan['cases'][index];out=Path(case['output_dir']);marker=out/'complete.json'
    if marker.is_file():
        complete=read_json(marker)
        if complete['case']!=case or any(hash_file(out/f'{k}/evaluation_summary.json')!=complete[f'{k}_evaluation_summary_sha256'] for k in ('single','swa')):
            raise ValueError('Existing completed SWA output differs')
        print('Already complete:',case['key']);return
    from al_pipeline.astronomy.analysis.swa_continuation import run_ten_epochs,atomic_json
    from al_pipeline.astronomy.astrominn.evaluation import AstroMiNNViTEvaluator
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from dataclasses import replace
    inp=case['source']
    result=run_ten_epochs(inp=inp,case=case,output=out/'training',registry=root/'registry.toml')
    original=DatasetRegistry.from_toml(root/'registry.toml').resolve_profile('astrominn_ztf').dataset
    dataset=replace(original,train_manifest=Path(inp['l0_manifest']))
    evaluator=AstroMiNNViTEvaluator(dataset=dataset,
        manifests={'validation':Path(inp['validation_manifest'])},device='cuda',num_workers=0,
        artifact_dir=Path(inp['model_artifact_dir']))
    checks={}
    for name in ('single','swa'):
        ck=Path(result['final_checkpoint'] if name=='single' else result['swa_checkpoint'])
        evaluation=evaluator.evaluate(checkpoint_path=ck,output_dir=out/name)
        checks[f'{name}_evaluation_summary_sha256']=hash_file(evaluation.summary_path)
    atomic_json(marker,{'case':case,'training_result':result,**checks,'validation_only':True,
                        'test_used':False,'completed_job_id':os.environ['SLURM_JOB_ID']})
    print(json.dumps({'case':case['key'],'training':result,'evaluation':checks},indent=2))


def status(root:Path):
    plan=verify(root)
    output=[]
    for i,c in enumerate(plan['cases']):
        path=Path(c['output_dir']);marker=path/'complete.json';last=path/'training'/'continuation_state.pth'
        if marker.exists():
            m=read_json(marker)
            if m['case']!=c:raise ValueError('SWA completion marker belongs to different case')
        output.append({'index':i,'case':c['key'],'completed':marker.exists(),
                       'has_restart_state':last.exists(),'submitted':(root/f'submitted_case_{i}.json').exists()})
    return {'cases':output,'complete':sum(x['completed'] for x in output),'planned':6}


def submit(root:Path,indices:list[int],walltime:str):
    plan=verify(root,deep=True)
    if len(set(indices))!=len(indices) or not indices or any(i not in range(6) for i in indices):
        raise ValueError('Supply distinct --indices in range 0..5')
    if not re.fullmatch(r'\d{2}:\d{2}:\d{2}',walltime):raise ValueError('Use HH:MM:SS walltime')
    hr,mn,sec=map(int,walltime.split(':'))
    if not 0 < hr*3600+mn*60+sec <= 20*60:raise ValueError('Walltime must be <=20 minutes')
    for idx in indices:
        if (root/f'submitted_case_{idx}.json').exists() or (Path(plan['cases'][idx]['output_dir'])/'complete.json').exists():
            raise RuntimeError(f'Case {idx} already queued or completed')
    worker=root/'source/scripts/research/historical_replay/task9/task9b_swa_worker.sbatch'
    for idx in indices:
        cmd=['sbatch','--parsable',f'--time={walltime}',
             f'--job-name=T9B_swa_{idx}',f'--output={root}/logs/T9B_%j.out',
             f'--error={root}/logs/T9B_%j.err',str(worker),str(root),str(idx)]
        response=subprocess.run(cmd,check=True,capture_output=True,text=True).stdout.strip()
        jobid=response.split(';')[0]
        if not jobid.isdigit():raise RuntimeError(f'Invalid Slurm ID: {response}')
        atomic_json(root/f'submitted_case_{idx}.json',{'case':plan['cases'][idx],
            'job_id':jobid,'command':cmd})
        print('Submitted',idx,plan['cases'][idx]['key'],jobid,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','prepare','verify','submit','run','status'))
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--task8b',type=Path,default=B)
    parser.add_argument('--job-dir',type=Path,default=ROOT)
    parser.add_argument('--index',type=int)
    parser.add_argument('--indices',default='0,3')
    parser.add_argument('--walltime',default='00:20:00')
    a=parser.parse_args()
    if a.command=='plan':
        plan=make_plan(a.task8b,a.job_dir)
        print(json.dumps({'cases':[{'index':i,'key':x['key'],'cycle':x['rejected_cycle']} for i,x in enumerate(plan['cases'])],
                'fits':6,'gpu_submission':False},indent=2))
    elif a.command=='prepare':
        x=prepare(a.repo,a.task8b,a.job_dir);print('PREPARED; no jobs submitted:',a.job_dir)
    elif a.command=='verify':
        verify(a.job_dir,deep=True);print('Frozen SWA source and all six source cases verified')
    elif a.command=='status':print(json.dumps(status(a.job_dir),indent=2))
    elif a.command=='submit':
        submit(a.job_dir,[int(x) for x in a.indices.split(',')],a.walltime)
    elif a.command=='run':
        if a.index is None:parser.error('--index required for run')
        one(a.job_dir,a.index)

if __name__=='__main__':main()
