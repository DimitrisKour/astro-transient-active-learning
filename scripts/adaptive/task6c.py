#!/usr/bin/env python3
"""Task6C equal-object loss closed loop; choose checkpoint mode at submission.

`prepare --selection loss` and `prepare --selection macro_f1` are safe to run
before Task6B finishes. `submit --selection ... --submit` requires matching
completed seed-specific L0 checkpoints and full preflight verification.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from task6c_build import MODES
from task6c_prepare import (PERSISTENT,DATA,SEEDS,POLICIES,freeze,verify,materialize,
                             check_splits,bootstrap_check,job_dir,_environment)


def plan(through:int=9)->list[dict]:
    if not 1<=through<=9:raise ValueError('through must be 1..9')
    out=[]
    for seed in SEEDS:
        for policy in POLICIES:
            parent=None
            for cycle in range(1,through+1):
                key=f'{seed}:{policy}:{cycle}'
                out.append(dict(seed=seed,policy=policy,cycle=cycle,key=key,parent=parent))
                parent=key
    assert len(out)==6*through
    return out

def validate_walltime(t:str)->None:
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d',t): raise ValueError('Walltime must be HH:MM:SS')
    h,m,s=map(int,t.split(':'))
    if not 0<h*3600+m*60+s<25*60:raise ValueError('Walltime must be positive and under 25 minutes')

def _arguments(source:Path,mode:str,seed:int,policy:str)->list[str]:
    script=source/'scripts/adaptive/task6c_common.sh'
    code='source "$1"; task6c_arguments "$2" "$3" "$4"; printf "%s\\n" "${TASK6C_ARGS[@]}"'
    r=subprocess.run(['bash','-c',code,'_',str(script),mode,str(seed),policy],
                     capture_output=True,text=True,check=True)
    return r.stdout.splitlines()

def preflight(mode:str)->None:
    dest=job_dir(mode);source=dest/'source'
    verify(mode,dest);check_splits(mode);bootstrap_check(mode,strict=True)
    for seed in SEEDS:
        arguments=_arguments(source,mode,seed,'entropy')
        logfile=dest/f'preflight_seed_{seed}.txt'
        env=_environment();env['PYTHONPATH']=str(source/'src')
        with logfile.open('w') as f:
            subprocess.run([sys.executable,str(source/'scripts/run_task6c_closed_loop.py'),
                *arguments,'--device','cpu'],check=True,env=env,stdout=f)
    print('Full CPU preflight succeeded: 3 matching L0 models; no training performed')

def _check_submit(mode:str,through:int)->None:
    dest=job_dir(mode)
    if through!=9: raise ValueError('Task6C v1 submission is a full nine-cycle experiment')
    for marker in ('submitted.json','submission_pending.json','submission_failed.json'):
        if (dest/marker).exists():raise RuntimeError(f'Existing submission marker: {dest/marker}')
    for seed in SEEDS:
        for policy in POLICIES:
            state=PERSISTENT/'runs'/MODES[mode]['suite']/policy/f'seed_{seed}'/'run_state.json'
            if state.exists(): raise RuntimeError(f'Task6C has existing trajectory state: {state}')
            source=DATA/'runs'/MODES[mode]['suite']/policy/f'seed_{seed}'
            if source.exists() and any(source.iterdir()):raise RuntimeError(f'Task6C existing artifacts: {source}')
    preflight(mode)

def sbatch_command(row:dict,mode:str,source:Path,dest:Path,parent_job:str|None,walltime:str)->list[str]:
    if bool(row['parent']) != (parent_job is not None):raise ValueError('Incorrect SLURM dependency')
    seed,cycle,policy=row['seed'],row['cycle'],row['policy']
    if mode not in MODES or seed not in SEEDS or cycle not in range(1,10) or policy not in POLICIES:
        raise ValueError('Task6C SLURM request outside approved scope')
    name=f'T6C_{mode}_{seed}_c{cycle}_{"ent" if policy=="entropy" else "bal"}'
    cmd=['sbatch','--parsable','--hold',f'--time={walltime}',f'--job-name={name}',
         f'--output={dest}/logs/%x_%j.out',f'--error={dest}/logs/%x_%j.err']
    if parent_job is not None:cmd.append('--dependency=afterok:'+parent_job)
    return cmd+[str(source/'scripts/adaptive/task6c_worker.sbatch'),
                str(source),mode,str(seed),str(cycle),policy]

def submit(mode:str,walltime:str='00:20:00',really:bool=False)->None:
    validate_walltime(walltime);dest=job_dir(mode)
    _check_submit(mode,through=9)
    rows=plan()
    if not really:
        print(f'DRY RUN: mode={mode}, suite={MODES[mode]["suite"]}; 54 independent 1-GPU jobs')
        for r in rows: print(f' {r["key"]:38} <- {r["parent"] or "matching L0 checkpoint"}')
        return
    source=dest/'source'
    jobids={};entries=[]
    pending=dest/'submission_pending.json';output=dest/'submitted.json'
    try:
        for r in rows:
            parent=jobids[r['parent']] if r['parent'] else None
            cmd=sbatch_command(r,mode,source,dest,parent,walltime)
            reply=subprocess.run(cmd,check=True,capture_output=True,text=True).stdout.strip()
            jobid=reply.split(';',1)[0]
            if not jobid.isdigit():raise RuntimeError(f'Unexpected sbatch ID: {reply!r}')
            jobids[r['key']]=jobid
            entries.append({**r,'job_id':jobid,'dependency_job_id':parent})
            pending.write_text(json.dumps({'status':'held','selection':mode,'jobs':entries},indent=2)+'\n')
        for entry in reversed(entries):
            subprocess.run(['scontrol','release',entry['job_id']],check=True,capture_output=True,text=True)
        pending.write_text(json.dumps({'status':'released','selection':mode,'jobs':entries},indent=2)+'\n')
        pending.replace(output)
    except BaseException:
        if entries:subprocess.run(['scancel',*[e['job_id'] for e in entries]],check=False)
        (dest/'submission_failed.json').write_text(json.dumps({'submitted':entries,'cancellation_requested':True},indent=2)+'\n')
        raise
    print(f'SUBMITTED 54 jobs in six independent seed/policy dependency chains; selection={mode}')
    print('Manifest:',output)

def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    for cmd in ('prepare','verify','plan','submit','preflight'):
        x=sub.add_parser(cmd)
        x.add_argument('--selection',choices=list(MODES),required=True)
        if cmd=='prepare':x.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
        if cmd in ('plan','submit'):x.add_argument('--walltime',default='00:20:00')
        if cmd=='submit':x.add_argument('--submit',action='store_true')
    a=p.parse_args();mode=a.selection
    if a.cmd=='prepare':
        freeze(a.repo.resolve(),mode)
        materialize(mode)
        print(f'PREPARED mode={mode}; source and L0/U0 split verified, no training or submission')
    elif a.cmd=='verify':
        verify(mode);check_splits(mode)
        print(f'VERIFIED {mode}; bootstrap ready={bootstrap_check(mode,strict=False)}')
    elif a.cmd=='plan':
        validate_walltime(a.walltime)
        print(f'PLAN: mode={mode} suite={MODES[mode]["suite"]}; 54 jobs; six independent chains')
        for r in plan():print(f' {r["key"]} <- {r["parent"] or "seed-specific L0"}')
    elif a.cmd=='preflight':preflight(mode)
    elif a.cmd=='submit':submit(mode,walltime=a.walltime,really=a.submit)

if __name__=='__main__':main()
