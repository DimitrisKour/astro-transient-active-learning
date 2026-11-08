#!/usr/bin/env python3
"""Task6A prepare, read-only audit, and opt-in DAG SLURM submission.

prepare and report NEVER submit; probe/submit require --submit explicitly.
Jobs use SLURM arrays limited to at most three concurrent A40 GPUs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from task6a_prepare import check_splits, verify

PERSISTENT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
SUITE = 'astrominn_closed_loop_warm_weight05_v1'
REFERENCE = 'astrominn_closed_loop_warm_v1'
BASE = 'astrominn_historical_replay_v1'
JOB_DIR = PERSISTENT / 'closed_loop' / SUITE
POLICIES = ('entropy', 'entropy_balanced_explore')
SEEDS = (135, 246, 357)


def _run(command: list[str], **opts) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, **opts)


def _source_arg_lines(source: Path, seed: int, policy: str) -> list[str]:
    q = ['bash', '-lc', 'source ' + str(source/'scripts/research/historical_replay/task6a/task6a_common.sh') +
         f'; task6a_arguments {seed} {policy}; '
         'printf "%s\\n" "${TASK6A_ARGS[@]}"']
    output = _run(q, capture_output=True)
    return output.stdout.splitlines()


def _audit_state(job_dir: Path, *, expect_at_least: int = 0) -> None:
    for policy in POLICIES:
        for seed in SEEDS:
            root = PERSISTENT/'runs'/SUITE/policy/f'seed_{seed}'
            state = root/'run_state.json'
            if not state.exists():
                if expect_at_least:
                    raise RuntimeError(f'No state for {policy} seed {seed}')
                continue
            record = json.loads(state.read_text())
            if record.get('active_cycle') is not None:
                raise RuntimeError(f'Active/incomplete cycle: {state}; inspect before submitting')
            if int(record.get('last_completed_cycle', 0)) < expect_at_least:
                raise RuntimeError(f'Expected cycle {expect_at_least}, found {record}: {state}')
            if int(record.get('last_completed_cycle', 0)) > 9:
                raise RuntimeError(f'Invalid cycle in {state}')


def prepare(repo: Path, job_dir: Path) -> None:
    registry = PERSISTENT/'closed_loop'/REFERENCE/'registry.toml'
    _run([sys.executable, str(repo/'scripts/research/historical_replay/task6a/task6a_prepare.py'),
          '--repo', str(repo), '--job-dir', str(job_dir),
          '--source-registry', str(registry),
          '--persistent', str(PERSISTENT), '--data', str(DATA)])
    frozen = job_dir/'source'
    for seed in SEEDS:
        args = _source_arg_lines(frozen, seed, 'entropy')
        plan = job_dir/f'plan_seed_{seed}.txt'
        with plan.open('w') as stream:
            _run([sys.executable, str(frozen/'scripts/run_task6a_closed_loop.py'),
                  *args, '--device', 'cpu'], env={**os.environ,'PYTHONPATH':str(frozen/'src')}, stdout=stream)
    check_splits(PERSISTENT, DATA)
    _audit_state(job_dir)
    print('TASK6A prepared: 54 fits, 6 seed/policy trajectories, 9 cycles. NO jobs queued.')


def _validate_time(value: str) -> None:
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d', value):
        raise ValueError('Expected HH:MM:SS walltime')
    h,m,s = map(int,value.split(':'))
    if not 0 < h*3600+m*60+s <= 24*60:
        raise ValueError('Walltime must be 1 second to 24 minutes')


def _status(root: Path, policy: str, seed: int) -> int:
    file = PERSISTENT/'runs'/SUITE/policy/f'seed_{seed}'/'run_state.json'
    if not file.exists():
        return 0
    state = json.loads(file.read_text())
    if state.get('active_cycle') is not None:
        raise RuntimeError(f'Active cycle at {file}')
    return int(state.get('last_completed_cycle',0))


def probe(job_dir: Path, *, submit: bool, walltime: str) -> None:
    verify(job_dir)
    _validate_time(walltime)
    if (job_dir/'submission.json').exists():
        raise RuntimeError('Full task already submitted')
    if (job_dir/'probe_submission.json').exists():
        raise RuntimeError('Probe already submitted; inspect existing job')
    if _status(job_dir, 'entropy', 135) != 0:
        raise RuntimeError('Probe trajectory already progressed')
    source = job_dir/'source'
    cmd = ['sbatch','--parsable','--array=0','--time='+walltime,
           '--job-name=T6A_probe','--output='+str(job_dir/'logs'/'probe_%A_%a.out'),
           str(source/'scripts/research/historical_replay/task6a/task6a_cycle.sbatch'),str(source),'1']
    print('PROBE:', ' '.join(cmd))
    if not submit:
        print('DRY RUN -- no jobs queued')
        return
    response = _run(cmd,capture_output=True).stdout.strip().split(';')[0]
    if not response.isdigit():
        raise RuntimeError(f'Unrecognized sbatch response: {response}')
    (job_dir/'probe_submission.json').write_text(json.dumps({'job_id':response,'array_index':0,'cycle':1,'time':walltime},indent=2)+'\n')
    print('Submitted single seed135 entropy c1 probe:', response)


def submit_all(job_dir: Path, *, submit: bool, walltime: str, through: int) -> None:
    verify(job_dir)
    _validate_time(walltime)
    if not 1 <= through <= 9:
        raise ValueError('through must be in 1..9')
    if (job_dir/'submission.json').exists() or (job_dir/'submission_pending.json').exists():
        raise RuntimeError('Task6A submission exists. Do not resubmit blindly.')
    if (job_dir/'probe_submission.json').exists():
        if _status(job_dir,'entropy',135) < 1:
            raise RuntimeError('Probe submitted but not completed; do not overlap with main submission')
    _audit_state(job_dir)
    for policy in POLICIES:
        for seed in SEEDS:
            completed = _status(job_dir, policy, seed)
            if completed > through:
                raise RuntimeError(f'{policy} seed{seed} already past requested {through}')
            if completed and not (policy=='entropy' and seed==135 and completed==1):
                raise RuntimeError(f'{policy} seed{seed} progressed beyond expected probe; recovery needs separate plan')
    source = job_dir/'source'
    print('TASK6A: 6 fits/cycle, max 3 concurrent A40 jobs; 1..'+str(through))
    if not submit:
        for c in range(1,through+1):
            print(f'  Cycle {c}: --array=0-5%3 --time={walltime}; depends on cycle {c-1 if c>1 else "none"}')
        print('DRY RUN -- no jobs queued')
        return
    ids=[]
    try:
        # Submit held to prevent execution until all dependencies are installed.
        for c in range(1,through+1):
            cmd=['sbatch','--parsable','--hold','--array=0-5%3','--time='+walltime,
                 '--job-name='+f'T6A_c{c}',
                 '--output='+str(job_dir/'logs'/f'c{c}_%A_%a.out')]
            if ids:
                cmd+=['--dependency=afterok:'+ids[-1]]
            cmd += [str(source/'scripts/research/historical_replay/task6a/task6a_cycle.sbatch'),str(source),str(c)]
            answer=_run(cmd,capture_output=True).stdout.strip().split(';')[0]
            if not answer.isdigit():
                raise RuntimeError('Bad sbatch response: '+answer)
            ids.append(answer)
        payload={'schema':'task6a_submission_v1','array_job_ids':ids,'cycles':through,
                 'fits':6*through, 'walltime':walltime,'concurrency_cap':3}
        (job_dir/'submission_pending.json').write_text(json.dumps(payload, indent=2)+'\n')
        for j in reversed(ids):
            _run(['scontrol','release',j])
        (job_dir/'submission_pending.json').replace(job_dir/'submission.json')
    except BaseException:
        if ids:
            subprocess.run(['scancel',*ids],check=False)
            (job_dir/'failed_submission.json').write_text(json.dumps({'cancelled_ids':ids})+'\n')
        raise
    print(f'QUEUED: {6*through} individual GPU tasks in {through} arrays, at most 3 running; IDs:', ' '.join(ids))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    cmd=p.add_subparsers(dest='action',required=True)
    for name in ('prepare','verify','probe','submit'):
        a=cmd.add_parser(name)
        a.add_argument('--job-dir', type=Path,default=JOB_DIR)
        if name=='prepare':
            a.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
        if name in ('probe','submit'):
            a.add_argument('--walltime',default='00:24:00')
            a.add_argument('--submit',action='store_true')
        if name=='submit':
            a.add_argument('--through',type=int,default=9)
    args=p.parse_args()
    if args.action=='prepare': prepare(args.repo.resolve(),args.job_dir)
    elif args.action=='verify': verify(args.job_dir);check_splits(PERSISTENT,DATA);print('Task6A frozen science contract and source splits verified')
    elif args.action=='probe': probe(args.job_dir,submit=args.submit,walltime=args.walltime)
    elif args.action=='submit': submit_all(args.job_dir,submit=args.submit,walltime=args.walltime,through=args.through)

if __name__=='__main__':
    main()
