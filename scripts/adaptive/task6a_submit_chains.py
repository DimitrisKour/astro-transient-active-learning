#!/usr/bin/env python3
"""Submit six independent per-(seed,policy) Task6A SLURM chains.

Scheduler-only patch for an ALREADY frozen Task6A suite. Never changes the
frozen scientific source. One held singleton-array sbatch per seed/policy/cycle.
Each job depends exclusively on its own previous-cycle job. No global GPU cap
is falsely advertised: at most six trajectories may use six GPUs at once.

Usage: python scripts/adaptive/task6a_submit_chains.py plan [--walltime HH:MM:SS]
       python scripts/adaptive/task6a_submit_chains.py submit [--walltime HH:MM:SS]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from task6a import (
    JOB_DIR, PERSISTENT, DATA, POLICIES, SEEDS, SUITE,
    _audit_state, _status, _validate_time,
)
from task6a_prepare import verify, check_splits


def _run(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def _plans(job_dir: Path, through: int) -> list[dict]:
    # C1..C9 per trajectory, rather than creating global cycle barriers.
    # The worker's one-task array index is seed-index*2 + policy-index.
    plans = []
    for seed_index, seed in enumerate(SEEDS):
        for policy_index, policy in enumerate(POLICIES):
            index = seed_index * len(POLICIES) + policy_index
            done = _status(job_dir, policy, seed)
            if done:
                if not (seed == 135 and policy == 'entropy' and done == 1
                        and (job_dir / 'probe_submission.json').exists()):
                    raise RuntimeError(f'Unexpected existing progress: {seed} {policy} c{done}. '
                                       'Recovery requires a separate audited plan.')
            for cycle in range(done + 1, through + 1):
                plans.append({'seed': seed, 'policy': policy, 'index': index, 'cycle': cycle})
    return plans


def validate(job_dir: Path, through: int, walltime: str) -> list[dict]:
    if not 1 <= through <= 9:
        raise ValueError('through must be 1..9')
    _validate_time(walltime)
    if job_dir.resolve() != JOB_DIR.resolve():
        raise ValueError('This patch is limited to the one frozen Task6A suite')
    verify(job_dir)
    check_splits(PERSISTENT, DATA)
    for file in ('submission.json', 'submission_pending.json',
                 'independent_submission.json', 'independent_submission_pending.json'):
        if (job_dir/file).exists():
            raise RuntimeError(f'Existing submission record; do not double-queue: {job_dir/file}')
    if (job_dir/'probe_submission.json').exists():
        if _status(job_dir, 'entropy', 135) != 1:
            raise RuntimeError('Probe exists but is not completed at exactly cycle 1')
    _audit_state(job_dir)
    plans = _plans(job_dir, through)
    if not plans:
        raise RuntimeError('Nothing left to submit')
    return plans


def make_command(*, source: Path, job_dir: Path, plan: dict,
                 walltime: str, predecessor: str | None) -> list[str]:
    seed, policy, index, cycle = (plan[k] for k in ('seed', 'policy', 'index', 'cycle'))
    command = [
        'sbatch', '--parsable', '--hold', f'--array={index}', f'--time={walltime}',
        f'--job-name=T6A_s{seed}_c{cycle}_{policy}',
        f'--output={job_dir}/logs/T6A_s{seed}_c{cycle}_{policy}_%A_%a.out',
    ]
    if predecessor is not None:
        command.append(f'--dependency=afterok:{predecessor}')
    command += [str(source/'scripts/adaptive/task6a_cycle.sbatch'), str(source), str(cycle)]
    return command


def execute(job_dir: Path, *, through: int, walltime: str, submit: bool) -> None:
    plans = validate(job_dir, through, walltime)
    print(f'Independent Task6A chains: {len(plans)} new jobs, six trajectories, '
          'at most six possible concurrent A40 GPUs.')
    if not submit:
        for p in plans:
            print(f"DRY RUN {p['seed']:>3} {p['policy']:<24} c{p['cycle']} "
                  f"array-index={p['index']} depends ONLY on same-trajectory c{p['cycle']-1}")
        print('NO jobs queued. Use `submit` (not `plan`) to queue.')
        return
    source = job_dir/'source'
    previous: dict[tuple[int, str], str] = {}
    submitted: list[dict] = []
    ids: list[str] = []
    pending = job_dir/'independent_submission_pending.json'
    final = job_dir/'independent_submission.json'
    try:
        # Submit all jobs held, then release in reverse. Dependencies remain
        # per-trajectory, never linking unrelated seed/policy trajectories.
        for p in plans:
            key = (p['seed'], p['policy'])
            predecessor = previous.get(key)
            response = _run(make_command(source=source, job_dir=job_dir, plan=p,
                                         walltime=walltime, predecessor=predecessor))
            job_id = response.split(';', 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f'Bad sbatch response: {response}')
            previous[key] = job_id
            record = dict(p, job_id=job_id, predecessor=predecessor)
            submitted.append(record)
            ids.append(job_id)
            pending.write_text(json.dumps({'state': 'held', 'walltime': walltime,
                                           'jobs': submitted}, indent=2) + '\n')
        for job_id in reversed(ids):
            _run(['scontrol', 'release', job_id])
        pending.replace(final)
        print(f'Queued {len(ids)} independent short jobs; each future cycle has '
              'afterok ONLY for its own previous job.')
        print(f'Manifest: {final}')
    except BaseException:
        if ids:
            subprocess.run(['scancel', *ids], check=False)
            (job_dir/'independent_submission_failure.json').write_text(
                json.dumps({'attempted_job_ids': ids, 'cancel_requested': True}, indent=2)+'\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'submit'))
    parser.add_argument('--walltime', default='00:20:00')
    parser.add_argument('--through', type=int, default=9)
    parser.add_argument('--job-dir', type=Path, default=JOB_DIR)
    args = parser.parse_args()
    execute(args.job_dir, through=args.through, walltime=args.walltime,
            submit=(args.action == 'submit'))


if __name__ == '__main__':
    main()
