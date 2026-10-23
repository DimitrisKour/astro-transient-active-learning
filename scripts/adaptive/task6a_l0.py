#!/usr/bin/env python3
"""Task6A new-L0 variant: prepare/verify/plan/submit independently.

Only `submit --submit` invokes sbatch. All queue jobs are held until the full
per-trajectory DAG is submitted. Each L0 seed has exactly two children (entropy
and balanced exploration); every cN>1 depends ONLY on cN-1 of its own chain.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from task6a_l0_prepare import freeze, verify as verify_source, check_splits, SEEDS, SUITE

PERSISTENT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
JOB_DIR = PERSISTENT/'closed_loop'/SUITE
REFERENCE = 'astrominn_closed_loop_warm_v1'
POLICIES = ('entropy', 'entropy_balanced_explore')


def _run(args: list[str], **kwargs):
    return subprocess.run(args, text=True, check=True, **kwargs)


def _arguments(source: Path, seed: int, policy: str) -> list[str]:
    script = source/'scripts/adaptive/task6a_l0_common.sh'
    code = 'source "$1"; task6a_l0_arguments "$2" "$3"; printf "%s\\n" "${TASK6A_L0_ARGS[@]}"'
    proc = _run(['bash','-c',code,'_',str(script),str(seed),policy],capture_output=True)
    return proc.stdout.splitlines()


def scientific_env(source: Path) -> dict[str,str]:
    env = dict(os.environ)
    env['PYTHONPATH']=str(source/'src')
    env['OMP_NUM_THREADS']='1'
    env['OPENBLAS_NUM_THREADS']='1'
    env['MKL_NUM_THREADS']='1'
    prefix = Path(env.get('CONDA_PREFIX',str(Path(sys.executable).parents[1])))
    stdcpp = prefix/'lib/libstdc++.so.6'
    if stdcpp.exists():
        env['LD_PRELOAD']=str(stdcpp)+((':'+env['LD_PRELOAD']) if env.get('LD_PRELOAD') else '')
        env['LD_LIBRARY_PATH']=str(prefix/'lib')+((':'+env['LD_LIBRARY_PATH']) if env.get('LD_LIBRARY_PATH') else '')
    return env


def source_preflight(repo: Path, job_dir: Path) -> None:
    source=job_dir/'source'
    check_splits(PERSISTENT,DATA)
    for seed in SEEDS:
        args=_arguments(source,seed,'entropy')
        output=job_dir/f'plan_seed_{seed}.txt'
        with output.open('w') as f:
            _run([sys.executable,str(source/'scripts/run_task6a_l0_closed_loop.py'),
                  *args,'--device','cpu'],env=scientific_env(source),stdout=f)
    print('Scientific preflight passed for seeds 135, 246, 357; no training or submission')


def _bootstrap_dir(seed: int) -> Path:
    return DATA/'runs'/SUITE/'shared'/f'seed_{seed}'/'bootstrap_training'


def bootstrap_status(seed: int) -> bool:
    folder=_bootstrap_dir(seed)
    summary=folder/'training_summary.json'
    checkpoint=folder/'best_vit_tower.pth'
    if not (summary.exists() or checkpoint.exists()):
        return False
    if not summary.is_file() or not checkpoint.is_file():
        raise RuntimeError(f'Incomplete bootstrap output at {folder}')
    rec=json.loads(summary.read_text(encoding='utf-8'))
    p=rec.get('policy',{})
    if (rec.get('seed')!=seed or rec.get('cycle')!=0 or
        rec.get('run_id')!=f'{SUITE}_bootstrap_seed{seed}' or
        rec.get('initialization')!='random_vit' or
        rec.get('initial_checkpoint') is not None or
        p.get('class_weight_exponent')!=0.5 or
        p.get('class_weight_source')!='training_manifest' or
        p.get('new_object_weighting')!='per_sample' or
        Path(rec.get('checkpoint_path','')) != checkpoint):
        raise RuntimeError(f'Incompatible bootstrap for seed {seed}: {summary}')
    if rec.get('train_manifest_sha256') is None:
        raise RuntimeError(f'Bootstrap missing manifest provenance: {summary}')
    return True


def _trajectory_state(seed: int,policy: str) -> int:
    state=PERSISTENT/'runs'/SUITE/policy/f'seed_{seed}'/'run_state.json'
    if not state.exists():
        return 0
    rec=json.loads(state.read_text())
    if rec.get('active_cycle') is not None:
        raise RuntimeError(f'Active/incomplete trajectory must not be requeued: {state}')
    cycle=int(rec.get('last_completed_cycle',0))
    if not 0<=cycle<=9:
        raise RuntimeError(f'Invalid state {state}')
    return cycle


def validate_walltime(s: str) -> None:
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d',s):
        raise ValueError('Expected HH:MM:SS walltime')
    h,m,sec=map(int,s.split(':'))
    if not 0<h*3600+m*60+sec<25*60:
        raise ValueError('Walltime must be greater than zero and under 25 minutes')


def plan(*,through:int=9) -> list[dict]:
    if not 1<=through<=9:
        raise ValueError('Cycles must be between 1 and 9')
    rows=[]
    for seed in SEEDS:
        rows.append({'seed':seed,'cycle':0,'policy':'bootstrap','key':f'l0:{seed}','parent':None})
    for seed in SEEDS:
        for policy in POLICIES:
            previous=f'l0:{seed}'
            for cycle in range(1,through+1):
                key=f'{seed}:{policy}:{cycle}'
                rows.append({'seed':seed,'cycle':cycle,'policy':policy,'key':key,'parent':previous})
                previous=key
    assert len(rows)==3+6*through
    return rows


def _check_pre_submission(job_dir:Path) -> None:
    if job_dir.resolve()!=JOB_DIR.resolve():
        raise ValueError('Only the dedicated new-bootstrap suite is supported')
    verify_source(job_dir)
    check_splits(PERSISTENT,DATA)
    for marker in ('submitted.json','submission_pending.json','submission_failed.json'):
        if (job_dir/marker).exists():
            raise RuntimeError(f'Existing submission marker: {job_dir/marker}')
    for seed in SEEDS:
        if bootstrap_status(seed):
            raise RuntimeError(f'L0 seed {seed} already completed: this is a NEW full-57 submission; use recovery procedure')
    for seed in SEEDS:
        for policy in POLICIES:
            if _trajectory_state(seed,policy):
                raise RuntimeError(f'Trajectory already progressed: {seed} {policy}')
    for seed in SEEDS:
        if _bootstrap_dir(seed).exists() and any(_bootstrap_dir(seed).iterdir()):
            raise RuntimeError(f'Partial bootstrap artifacts exist: {_bootstrap_dir(seed)}')


def sbatch_command(job_dir:Path, source:Path, row:dict, *, parent_job:str|None,walltime:str,l0_walltime:str)->list[str]:
    seed,policy,cycle=row['seed'],row['policy'],row['cycle']
    if cycle==0 and parent_job is not None:
        raise ValueError('A bootstrap must have no dependency')
    if cycle>0 and parent_job is None:
        raise ValueError('An AL cycle requires its preceding job')
    label='l0' if cycle==0 else f'c{cycle}_{policy}'
    name=f'T6A_{seed}_{label}'
    cmd=['sbatch','--parsable','--hold',f'--time={l0_walltime if cycle==0 else walltime}',
         f'--job-name={name}',f'--output={job_dir}/logs/%x_%j.out',
         f'--error={job_dir}/logs/%x_%j.err']
    if parent_job is not None:
        cmd.append('--dependency=afterok:'+parent_job)
    return cmd+[str(source/'scripts/adaptive/task6a_l0_worker.sbatch'), str(source),str(seed),str(cycle),policy]


def submit(job_dir: Path, *, through:int=9,walltime:str='00:20:00',l0_walltime:str='00:20:00',actually_submit:bool=False):
    validate_walltime(walltime);validate_walltime(l0_walltime)
    _check_pre_submission(job_dir)
    rows=plan(through=through)
    if not actually_submit:
        print(f'DRY RUN: {len(rows)} independent single-A40 jobs; 3 L0 bootstraps + {6*through} closed-loop fits')
        for row in rows:
            print(f"  {row['key']:<40} depends on {row['parent'] or 'nothing'}")
        print('No jobs queued. Explicit --submit required.')
        return
    source=job_dir/'source'
    jobids:dict[str,str]={}
    submitted=[]
    pending=job_dir/'submission_pending.json'
    output=job_dir/'submitted.json'
    try:
        for row in rows:
            parent=jobids.get(row['parent']) if row['parent'] else None
            response=_run(sbatch_command(job_dir,source,row,parent_job=parent,walltime=walltime,l0_walltime=l0_walltime),capture_output=True).stdout.strip()
            jobid=response.split(';',1)[0]
            if not jobid.isdigit():
                raise RuntimeError(f'Unexpected sbatch response {response!r}')
            jobids[row['key']]=jobid
            submitted.append({**row,'job_id':jobid,'dependency_job_id':parent})
            pending.write_text(json.dumps({'status':'held','jobs':submitted},indent=2)+'\n')
        for row in reversed(submitted):
            _run(['scontrol','release',row['job_id']],capture_output=True)
        pending.replace(output)
    except BaseException:
        if submitted:
            subprocess.run(['scancel',*[x['job_id'] for x in submitted]],check=False)
        (job_dir/'submission_failed.json').write_text(json.dumps({'submitted':submitted,'cancellation_requested':True},indent=2)+'\n')
        raise
    print(f'SUBMITTED {len(submitted)} jobs; 3 initial L0 seed jobs, 6 independent policy trajectories.')
    print('Each c1 depends ONLY on its own seed bootstrap; each cN>1 depends ONLY on same seed/policy c(N-1).')
    print('At most 6 parallel A40 fits; actual concurrency depends on Delta scheduler.\nManifest:',output)


def prepare(repo:Path, job_dir:Path)->None:
    registry=PERSISTENT/'closed_loop'/REFERENCE/'registry.toml'
    freeze(repo,job_dir,registry)
    source_preflight(repo,job_dir)
    print('PREPARED: dedicated new-L0 Task6A source; 57 GPU jobs planned; nothing queued')


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    for cmd in ('prepare','verify','plan','submit'):
        p=sub.add_parser(cmd)
        p.add_argument('--job-dir',type=Path,default=JOB_DIR)
        if cmd=='prepare': p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
        if cmd in ('plan','submit'):
            p.add_argument('--through',type=int,default=9)
            p.add_argument('--walltime',default='00:20:00')
            p.add_argument('--l0-walltime',default='00:20:00')
        if cmd=='submit':p.add_argument('--submit',action='store_true')
    a=ap.parse_args()
    if a.cmd=='prepare':prepare(a.repo.resolve(),a.job_dir)
    elif a.cmd=='verify':
        verify_source(a.job_dir);check_splits(PERSISTENT,DATA)
        print('Verified Task6A frozen source and L0/U0 split; no job submitted')
    elif a.cmd in ('plan','submit'):
        submit(a.job_dir,through=a.through,walltime=a.walltime,l0_walltime=a.l0_walltime,actually_submit=(a.cmd=='submit' and a.submit))

if __name__=='__main__':main()
