#!/usr/bin/env python3
"""Task7A warm-start temperature-scaled entropy: prepare, inspect, verify, submit.

A five-cycle PILOT only: exactly 15 single-A40 fits in 3 independent seed chains.
The frozen scientific contract permits future cycles 6-9, but DOES NOT queue them.
Only `submit --submit` calls sbatch. No model training on login nodes.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from task7a_prepare import freeze,verify as verify_frozen,check_splits,SEEDS,SUITE,REFERENCE

P=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
D=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
ROOT=P/'closed_loop'/SUITE
POLICY='entropy'


def run(args:list[str],**kwargs):
    return subprocess.run(args,check=True,text=True,**kwargs)


def command_args(source:Path, seed:int)->list[str]:
    proc=run(['bash','-c','source "$1"; task7a_arguments "$2"; printf "%s\\n" "${TASK7A_ARGS[@]}"',
              '_',str(source/'scripts/research/historical_replay/task7a/task7a_common.sh'),str(seed)],capture_output=True)
    return proc.stdout.splitlines()


def scientific_env(source:Path)->dict[str,str]:
    env=os.environ.copy()
    env['PYTHONPATH']=str(source/'src')
    for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[var]='1'
    prefix=Path(env.get('CONDA_PREFIX',str(Path(sys.executable).parents[1])))
    stdcpp=prefix/'lib/libstdc++.so.6'
    if stdcpp.exists():
        env['LD_LIBRARY_PATH']=str(prefix/'lib')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        env['LD_PRELOAD']=str(stdcpp)+(':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    return env


def preflight(job:Path)->None:
    source=job/'source'
    verify_frozen(job)
    # The runner CREATES the replay split before check_splits() reads it.
    # --run is never passed. Reused .85 L0 records are required, not retrained.
    for seed in SEEDS:
        args=command_args(source,seed)
        with (job/f'preflight_seed_{seed}.txt').open('w') as output:
            run([sys.executable,str(source/'scripts/run_task7a_closed_loop.py'),
                *args,'--device','cpu'],env=scientific_env(source),stdout=output)
    check_splits(P,D)
    print('Task7A preflight completed: original .85 L0 and original U0, no GPU work')


def validate_walltime(value:str)->None:
    if not re.fullmatch(r'[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]',value):
        raise ValueError('Walltime must have HH:MM:SS format')
    h,m,s=map(int,value.split(':'))
    if not 0 < h*3600+m*60+s < 25*60:raise ValueError('Use 0 < walltime < 25 minutes')


def plan()->list[dict]:
    return [{'seed':seed,'cycle':c,'policy':POLICY,
             'key':f'{seed}:entropy:{c}',
             'parent':None if c==1 else f'{seed}:entropy:{c-1}'}
            for seed in SEEDS for c in range(1,6)]


def check_submission(job:Path)->None:
    if job.resolve()!=ROOT.resolve():raise ValueError('Task7A only supports its dedicated suite')
    verify_frozen(job)
    check_splits(P,D)
    for name in ('submitted.json','submission_pending.json','submission_failed.json'):
        if (job/name).exists():raise RuntimeError(f'Submission marker already exists: {job/name}')
    for seed in SEEDS:
        state=P/'runs'/SUITE/POLICY/f'seed_{seed}'/'run_state.json'
        if state.exists():raise RuntimeError(f'Task7A has already started; do not resubmit: {state}')
        # Original benchmark should be complete to provide a full reference.
        ref=P/'runs'/REFERENCE/POLICY/f'seed_{seed}'/'run_state.json'
        if not ref.is_file() or int(json.loads(ref.read_text())['last_completed_cycle'])<9:
            raise RuntimeError(f'Original entropy benchmark incomplete: {ref}')
        bootstrap=D/'runs'/'astrominn_historical_replay_v1'/'shared'/f'seed_{seed}'/'bootstrap_training'/'best_vit_tower.pth'
        if not bootstrap.is_file():raise FileNotFoundError(bootstrap)


def sbatch_command(job:Path,row:dict,parent_job:str|None,walltime:str)->list[str]:
    if row['cycle']==1 and parent_job is not None:raise ValueError('Cycle 1 has no dependency')
    if row['cycle']>1 and parent_job is None:raise ValueError('Later cycles need their own prior fit')
    name=f'T7A_s{row["seed"]}_c{row["cycle"]}'
    result=['sbatch','--parsable','--hold',f'--time={walltime}',
            f'--job-name={name}',f'--output={job}/logs/%x_%j.out',
            f'--error={job}/logs/%x_%j.err']
    if parent_job:result.append(f'--dependency=afterok:{parent_job}')
    return result+[str(job/'source/scripts/research/historical_replay/task7a/task7a_worker.sbatch'),
                   str(job/'source'),str(row['seed']),str(row['cycle'])]


def submit(job:Path,walltime:str='00:20:00',actually:bool=False)->None:
    validate_walltime(walltime)
    check_submission(job)
    rows=plan()
    if not actually:
        print('DRY RUN: 15 held one-A40 jobs in THREE independent chains (5 cycles per seed).')
        for x in rows:print(f"  {x['key']:>20} depends on {x['parent'] or 'nothing'}")
        print('No SLURM jobs submitted; add --submit when ready.')
        return
    jobids={};done=[];pending=job/'submission_pending.json'
    try:
        for x in rows:
            parent=jobids.get(x['parent']) if x['parent'] else None
            raw=run(sbatch_command(job,x,parent,walltime),capture_output=True).stdout.strip()
            jid=raw.split(';')[0]
            if not jid.isdigit():raise RuntimeError(f'Unexpected SLURM job id: {raw!r}')
            jobids[x['key']]=jid
            done.append({**x,'job_id':jid,'dependency_job_id':parent})
            pending.write_text(json.dumps({'status':'held','jobs':done},indent=2)+'\n')
        for x in reversed(done):run(['scontrol','release',x['job_id']],capture_output=True)
        pending.write_text(json.dumps({'status':'released','jobs':done},indent=2)+'\n')
        pending.replace(job/'submitted.json')
    except BaseException:
        if done:subprocess.run(['scancel',*[x['job_id'] for x in done]],check=False)
        (job/'submission_failed.json').write_text(json.dumps({'jobs':done,'cancellation_requested':True},indent=2)+'\n')
        raise
    print('SUBMITTED 15 fits, 3 independent entropy-only chains of 5 cycles.')
    print('Queue manifest:',job/'submitted.json')


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    for cmd in ('prepare','verify','plan','submit'):
        q=sub.add_parser(cmd)
        q.add_argument('--job-dir',type=Path,default=ROOT)
        if cmd=='prepare':q.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
        if cmd=='verify':q.add_argument('--source-only',action='store_true')
        if cmd=='submit':
            q.add_argument('--walltime',default='00:20:00');q.add_argument('--submit',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':
        registry=P/'closed_loop'/REFERENCE/'registry.toml'
        freeze(a.repo.resolve(),a.job_dir,registry)
        preflight(a.job_dir)
        print('Task7A source and input data prepared; no jobs queued')
    elif a.action=='verify':
        verify_frozen(a.job_dir)
        if not a.source_only:check_splits(P,D)
        print('Task7A source verified'+('' if a.source_only else ', split and bootstraps verified'))
    elif a.action=='plan':
        for x in plan():print(f"{x['key']} <- {x['parent'] or 'none'}")
        print('Exactly 15 jobs; NO submission')
    elif a.action=='submit':submit(a.job_dir,a.walltime,a.submit)

if __name__=='__main__':main()
