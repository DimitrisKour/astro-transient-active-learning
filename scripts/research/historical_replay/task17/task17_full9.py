#!/usr/bin/env python3
"""Task17 full-nine-cycle controller; submit 54 HELD jobs before pilot completion.

Reuse Task17's immutable random-L0 split, fresh per-seed bootstrap and frozen
scientific source. Cycles 1-3 run under original Task17; cycle 4 explicitly
extends the budget to 9; cycles 5-9 continue the extended trajectories.
No original frozen source, existing pilot, or bootstrap is rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
T17 = ROOT / 'closed_loop/astrominn_task17_random_l0_v1'
CONTROL = ROOT / 'closed_loop/astrominn_task17_random_l0_full9_controller_v1'
POLICIES = ('random', 'entropy')
SEEDS = (135, 246, 357)
CYCLES = tuple(range(1, 10))
FILES = ('task17_full9.py', 'task17_full9_worker.sbatch')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parent():
    path = T17 / 'source/scripts/adaptive/task17.py'
    spec = importlib.util.spec_from_file_location('_frozen_task17_full9_parent', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Frozen Task17 manager is missing')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_parent():
    mod = parent()
    mod.verify(T17)
    p = read(T17 / 'scientific_plan.json')
    if (p['acquisition']['cycles'] != 3 or p['acquisition']['regular_batch_objects'] != 473
            or p['split']['l0_objects'] != 476 or p['split']['seed_fixed_across_training_seeds'] != 17017
            or tuple(p['policies']) != POLICIES or tuple(p['seeds']) != SEEDS):
        raise RuntimeError('Unexpected original Task17 protocol; refusing to schedule')
    runner = (T17 / 'source/scripts/run_astrominn_closed_loop.py').read_text(encoding='utf-8')
    experiment = (T17 / 'source/src/al_pipeline/astronomy/experiments/simulated.py').read_text(encoding='utf-8')
    if ('--complete-pool' not in runner or '--extend-budget' not in runner
            or 'validate_budget_extension' not in experiment):
        raise RuntimeError('Frozen Task17 source cannot safely extend the budget')
    return mod


def prepare(controller=CONTROL, repo=None):
    verify_parent()
    controller = Path(controller)
    if (controller / 'manifest.json').exists():
        verify(controller)
        print('Full9 controller already prepared; original Task17 untouched')
        return
    if controller.exists() and any(controller.iterdir()):
        raise RuntimeError('Partial controller found; inspect rather than overwrite')
    src = (Path(repo) / 'scripts/research/historical_replay/task17') if repo is not None else Path(__file__).resolve().parent
    for name in FILES:
        if not (src / name).is_file():
            raise FileNotFoundError(src / name)
    (controller / 'source').mkdir(parents=True, exist_ok=True)
    (controller / 'logs').mkdir(exist_ok=True)
    for name in FILES:
        shutil.copy2(src / name, controller / 'source' / name)
    write(controller / 'manifest.json', {
        'original_suite': T17.name,
        'parent_manifest_sha256': sha(T17 / 'frozen_manifest.json'),
        'parent_plan_sha256': sha(T17 / 'scientific_plan.json'),
        'controller_source_sha256': {name: sha(controller / 'source' / name) for name in FILES},
        'policies': list(POLICIES),
        'seeds': list(SEEDS),
        'cycles': list(CYCLES),
        'bootstrap': 'reuse existing Task17 seed-specific L0 models',
        'split_seed': 17017,
        'initial_labeled': 476,
        'acquisition_batch': 473,
        'final_batch': 469,
        'final_labeled': 4729,
        'job_count': 54,
        'extend_at_cycle': 4,
    })
    verify(controller)
    print('Prepared separate full9 controller; existing frozen Task17 unchanged')


def verify(controller=CONTROL):
    controller = Path(controller)
    info = read(controller / 'manifest.json')
    if (info['original_suite'] != T17.name or info['cycles'] != list(CYCLES)
            or tuple(info['policies']) != POLICIES or tuple(info['seeds']) != SEEDS
            or info['job_count'] != 54 or info['extend_at_cycle'] != 4):
        raise RuntimeError('Unexpected Task17 full9 controller plan')
    if sha(T17 / 'frozen_manifest.json') != info['parent_manifest_sha256']:
        raise RuntimeError('Original frozen Task17 source manifest changed')
    if sha(T17 / 'scientific_plan.json') != info['parent_plan_sha256']:
        raise RuntimeError('Original Task17 scientific plan changed')
    for name in FILES:
        if sha(controller / 'source' / name) != info['controller_source_sha256'][name]:
            raise RuntimeError('Controller source integrity failed: ' + name)
    verify_parent()
    print('Task17 full9 controller verified')


def cases():
    for policy in POLICIES:
        for seed in SEEDS:
            for cycle in CYCLES:
                yield policy, seed, cycle


def runner_args(policy, seed, cycle):
    if policy not in POLICIES or seed not in SEEDS or cycle not in range(4, 10):
        raise ValueError('Only Task17 extension cycles 4-9 belong here')
    args = parent().runner_args(seed, policy, mode='cycle')
    if args.count('--max-cycles') != 1 or args.count('--require-existing-bootstrap') != 1:
        raise RuntimeError('Frozen runner argument layout changed')
    args[args.index('--max-cycles') + 1] = '9'
    args += ['--complete-pool', '--device', 'cuda', '--cycle', str(cycle), '--run']
    if cycle == 4:
        args += ['--extend-budget']
    return args


def job(policy, seed, cycle, controller=CONTROL):
    verify(controller)
    args = runner_args(policy, seed, cycle)
    cmd = [sys.executable, str(T17 / 'source/scripts/run_astrominn_closed_loop.py'), *args]
    env = dict(os.environ, PYTHONPATH=str(T17 / 'source/src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    subprocess.run(cmd, env=env, check=True)


def pilot_ready():
    p = T17 / 'pilot/seed_135/result.json'
    if not p.is_file():
        raise RuntimeError('Task17 GPU acquisition pilot has not completed successfully')
    result = read(p)
    if (result.get('seed') != 135 or result.get('candidate_count') != 4253
            or result.get('acquisition_budget') != 473 or result.get('unique') is not True
            or result.get('validation_labels_accessed') is not False
            or result.get('test_labels_accessed') is not False):
        raise RuntimeError('Task17 pilot result failed scientific invariants')
    selected = result.get('selected_ids')
    if not isinstance(selected, dict) or set(selected) != set(POLICIES):
        raise RuntimeError('Missing random/entropy pilot selections')
    for policy in POLICIES:
        ids = selected[policy]
        if len(ids) != 473 or len(set(ids)) != 473:
            raise RuntimeError('Pilot selection budget/uniqueness failed: ' + policy)
    frozen = parent()
    if result.get('checkpoint_sha256') != sha(frozen.bootstrap_checkpoint(135)):
        raise RuntimeError('Pilot checkpoint does not match bootstrap seed 135')
    print('Task17 pilot verified, both policies, no extra oracle access')


def submit(controller=CONTROL, walltime='00:20:00'):
    """Create every job HELD now; no pilot or bootstrap completion required."""
    verify(controller)
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d', walltime) or walltime == '00:00:00':
        raise ValueError('Invalid walltime')
    if (T17 / 'submitted.json').exists() and read(T17 / 'submitted.json').get('jobs'):
        raise RuntimeError('Original Task17 AL ledger already contains jobs; avoid duplicate submission')
    controller = Path(controller)
    ledger = controller / 'submitted.json'
    data = read(ledger) if ledger.exists() else {'state': 'held_partial', 'jobs': []}
    # Incomplete queues can safely be resumed from the last recorded job,
    # retaining afterok dependencies and never duplicating recorded cycles.
    existing = {(x['policy'], x['seed'], x['cycle']): x for x in data['jobs']}
    for policy in POLICIES:
        for seed in SEEDS:
            for cycle in CYCLES:
                key = (policy, seed, cycle)
                previous = existing.get((policy, seed, cycle-1))
                dependency = previous['job_id'] if previous else None
                if key in existing:
                    if existing[key].get('parent') != dependency:
                        raise RuntimeError(f'Inconsistent dependency in existing ledger: {key}')
                    continue
                if cycle != 1 and dependency is None:
                    raise RuntimeError('Missing predecessor for cycle ' + str(cycle))
                cmd = ['sbatch', '--parsable', '--hold', '--time', walltime,
                       '--job-name', f'T17full9_{policy}_{seed}_{cycle}',
                       '--output', str(controller / 'logs/%x_%j.out'),
                       '--error', str(controller / 'logs/%x_%j.err')]
                if dependency:
                    cmd += ['--dependency', f'afterok:{dependency}']
                if cycle <= 3:
                    cmd += [str(T17 / 'source/scripts/adaptive/task17_worker.sbatch'),
                            str(T17), 'cycle', policy, str(seed), str(cycle)]
                else:
                    cmd += [str(controller / 'source/task17_full9_worker.sbatch'),
                            str(controller), policy, str(seed), str(cycle)]
                jid = subprocess.check_output(cmd, text=True).strip().split(';')[0]
                if not jid.isdigit():
                    raise RuntimeError('Invalid sbatch job ID: ' + jid)
                entry = {'policy': policy, 'seed': seed, 'cycle': cycle,
                         'job_id': jid, 'parent': dependency, 'released': False}
                data['jobs'].append(entry)
                existing[key] = entry
                write(ledger, data)
                print(f'HELD {jid}: {policy}/seed{seed}/cycle{cycle}')
    if len(data['jobs']) != 54:
        raise RuntimeError('Incomplete held submission: ' + str(len(data['jobs'])))
    data['state'] = 'held'
    write(ledger, data)
    print('Task17: 54/54 jobs submitted HELD in six 9-cycle afterok chains')


def release(controller=CONTROL, all_jobs=False):
    """Explicit release; pilot and all three bootstraps must be available."""
    verify(controller)
    if not all_jobs:
        raise ValueError('For full9 use release --all after the pilot completes')
    controller = Path(controller)
    ledger = controller / 'submitted.json'
    data = read(ledger)
    if len(data.get('jobs', [])) != 54:
        raise RuntimeError('Task17 full9 held submission is incomplete')
    pilot_ready()
    for seed in SEEDS:
        if not parent().bootstrap_checkpoint(seed).is_file():
            raise RuntimeError(f'Missing Task17 L0 bootstrap checkpoint for seed {seed}')
    # Release backwards, after all successors are recorded, preserving afterok.
    for entry in reversed(data['jobs']):
        if entry['released']:
            continue
        subprocess.run(['scontrol', 'release', entry['job_id']], check=True)
        entry['released'] = True
        data['state'] = 'partially_released'
        write(ledger, data)
        print('Released', entry['job_id'], entry['policy'], entry['seed'], entry['cycle'])
    data['state'] = 'released_all'
    write(ledger, data)
    print('All 54 Task17 cycles released with dependencies intact')


def status(controller=CONTROL):
    controller = Path(controller)
    ledger = controller / 'submitted.json'
    data = read(ledger) if ledger.is_file() else {'jobs': [], 'state': 'not_submitted'}
    rows = []
    for policy in POLICIES:
        for seed in SEEDS:
            state = ROOT / 'runs' / T17.name / policy / f'seed_{seed}/run_state.json'
            done = int(read(state).get('last_completed_cycle', 0)) if state.is_file() else 0
            rows.append({'policy': policy, 'seed': seed, 'completed_cycles': done, 'complete': done == 9})
    print(json.dumps({'jobs': len(data['jobs']), 'released': sum(x['released'] for x in data['jobs']),
                      'state': data.get('state', 'not_submitted'),
                      'completed_trajectories': sum(row['complete'] for row in rows),
                      'trajectories': rows}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify', 'submit', 'release', 'status', 'job'))
    p.add_argument('positional', nargs='*')
    p.add_argument('--controller', type=Path, default=CONTROL)
    p.add_argument('--repo', type=Path)
    p.add_argument('--walltime', default='00:20:00')
    p.add_argument('--all', action='store_true')
    a = p.parse_args()
    if a.command == 'prepare':
        prepare(a.controller, a.repo)
    elif a.command == 'verify':
        verify(a.controller)
    elif a.command == 'submit':
        submit(a.controller, a.walltime)
    elif a.command == 'release':
        release(a.controller, a.all)
    elif a.command == 'status':
        status(a.controller)
    elif a.command == 'job':
        if len(a.positional) != 3:
            raise ValueError('job needs policy seed cycle')
        job(a.positional[0], int(a.positional[1]), int(a.positional[2]), a.controller)


if __name__ == '__main__':
    main()
