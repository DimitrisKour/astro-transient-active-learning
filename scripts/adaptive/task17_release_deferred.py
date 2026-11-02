#!/usr/bin/env python3
"""Gate held Task17 full9 first cycles on pilot and bootstrap Slurm jobs.

Safe to use while the acquisition pilot and some bootstraps are pending.
Never edits frozen Task17 source or the experiment's scientific configuration.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import task17_full9 as full


def active_job(job_id: str, expected_name: str) -> str:
    if not str(job_id).isdigit():
        raise ValueError('Invalid Slurm job ID')
    result = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job_id)], text=True)
    name = re.search(r'(?:^|\s)JobName=([^\s]+)', result)
    state = re.search(r'(?:^|\s)JobState=([^\s]+)', result)
    if name is None or name.group(1) != expected_name:
        raise RuntimeError(f'Job {job_id} is not the expected {expected_name}')
    if state is None or state.group(1) not in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
        raise RuntimeError(f'Prerequisite job {job_id} has unexpected state: {state.group(1) if state else "unknown"}')
    return result


def guard_and_release(pilot_job: str):
    full.verify()
    ledger = full.CONTROL / 'submitted.json'
    data = full.read(ledger)
    entries = data['jobs']
    expected = set(full.cases())
    actual = {(x['policy'], x['seed'], x['cycle']) for x in entries}
    if len(entries) != 54 or actual != expected:
        raise RuntimeError('Expected exactly 54 uniquely keyed Task17 full9 jobs')
    if (full.T17 / 'submitted.json').exists() and full.read(full.T17 / 'submitted.json').get('jobs'):
        raise RuntimeError('Unexpected independent three-cycle jobs; refusing duplicate release')
    by_key = {(x['policy'], x['seed'], x['cycle']): x for x in entries}
    for policy, seed, cycle in full.cases():
        entry = by_key[policy, seed, cycle]
        ancestor = by_key.get((policy, seed, cycle - 1))
        if entry['parent'] != (ancestor['job_id'] if ancestor else None):
            raise RuntimeError('Task17 chain dependency ledger mismatch')
    boots = full.read(full.T17 / 'bootstrap_jobs.json')['jobs']
    bootstrap = {int(x['seed']): x for x in boots}
    if set(bootstrap) != set(full.SEEDS) or len(boots) != len(full.SEEDS):
        raise RuntimeError('Expected three distinct existing bootstrap jobs')
    if not all(x['released'] for x in boots):
        raise RuntimeError('Release the three bootstrap jobs before deferred AL release')

    pilot_result = full.T17 / 'pilot/seed_135/result.json'
    if pilot_result.is_file():
        full.pilot_ready()
        pilot_gate = None
        print('Pilot result exists and passed checks; no pending Slurm pilot gate required')
    else:
        active_job(pilot_job, 'T17_pilot_135')
        pilot_gate = str(pilot_job)
        print('Pilot still running/pending: gating every first cycle on', pilot_gate)

    seed_gates = {}
    for seed in full.SEEDS:
        checkpoint = full.parent().bootstrap_checkpoint(seed)
        if checkpoint.is_file():
            seed_gates[seed] = None
            print(f'Seed {seed}: bootstrap checkpoint present')
        else:
            job = bootstrap[seed]
            active_job(str(job['job_id']), f'T17_L0_{seed}')
            seed_gates[seed] = str(job['job_id'])
            print(f'Seed {seed}: gating first cycle on bootstrap', job['job_id'])

    # Install/verify dependencies BEFORE releasing any AL job. A rerun after
    # an interrupted update remains safe because all jobs are initially held.
    for policy in full.POLICIES:
        for seed in full.SEEDS:
            first = by_key[(policy, seed, 1)]
            dependencies = [j for j in (pilot_gate, seed_gates[seed]) if j]
            if first['released'] and not data.get('gates_installed'):
                raise RuntimeError('First-cycle job released before Slurm gating; inspect manually')
            if dependencies:
                value = 'afterok:' + ':'.join(dependencies)
                subprocess.run(['scontrol', 'update', f'JobId={first["job_id"]}', f'Dependency={value}'], check=True)
                status = subprocess.check_output(['scontrol', 'show', 'job', '-o', first['job_id']], text=True)
                match = re.search(r'(?:^|\s)Dependency=([^\s]+)', status)
                if match is None or not all(j in match.group(1) for j in dependencies):
                    raise RuntimeError('Slurm dependency verification failed on ' + first['job_id'])
                print(f'Installed {value} on first-cycle job {first["job_id"]}')
            else:
                print(f'All prerequisites already available for first-cycle job {first["job_id"]}')
    data['gates_installed'] = True
    data['pilot_job_id'] = str(pilot_job)
    data['bootstrap_job_ids'] = {str(s): bootstrap[s]['job_id'] for s in full.SEEDS}
    full.write(ledger, data)

    for entry in reversed(entries):
        if entry['released']:
            continue
        subprocess.run(['scontrol', 'release', str(entry['job_id'])], check=True)
        entry['released'] = True
        data['state'] = 'partially_released'
        full.write(ledger, data)
    data['state'] = 'released_all'
    full.write(ledger, data)
    print('RELEASED: 54 Task17 jobs; first cycles gated by pilot/bootstrap, later cycles by afterok')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-job', required=True)
    args = parser.parse_args()
    guard_and_release(args.pilot_job)


if __name__ == '__main__':
    main()
