#!/usr/bin/env python3
"""Resume Task13's six completed four-cycle trajectories through cycle nine.

Freeze this controller separately from the original four-cycle experiment.
The original source, submitted.json and first four cycles are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

T13 = Path('/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task13_tailor_pilot_v1')
CONTROL = T13.parent / 'astrominn_task13_tailor_extension_v1'
ROOT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
POLICIES = ('tailor_diversity', 'uniform_mixture')
SEEDS = (135, 246, 357)
CYCLES = range(5, 10)
FILES = ('task13_extension.py', 'task13_extension_worker.sbatch')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def frozen_task13():
    location = T13 / 'source/scripts/adaptive/task13.py'
    spec = importlib.util.spec_from_file_location('_frozen_task13_extension_parent', location)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load frozen Task13 manager')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_parent_source():
    parent = frozen_task13()
    parent.verify(T13)
    plan = read(T13 / 'scientific_plan.json')
    if (plan['acquisition_cycles'] != 4 or plan['training_jobs'] != 24
            or tuple(plan['policies']) != POLICIES or tuple(plan['seeds']) != SEEDS):
        raise ValueError('Frozen Task13 is not the expected four-cycle study')
    runner = (T13 / 'source/scripts/run_astrominn_closed_loop.py').read_text()
    simulated = (T13 / 'source/src/al_pipeline/astronomy/experiments/simulated.py').read_text()
    if '--extend-budget' not in runner or '--complete-pool' not in runner:
        raise RuntimeError('Frozen runner does not support explicit budget extension')
    if 'validate_budget_extension' not in simulated:
        raise RuntimeError('Frozen experiment lacks extension integrity checks')


def prepare(controller=CONTROL, repo=None):
    controller = Path(controller)
    check_parent_source()
    if (controller / 'manifest.json').exists():
        verify(controller)
        print('Extension controller already prepared; no files changed')
        return
    if controller.exists() and any(controller.iterdir()):
        raise RuntimeError('Extension controller exists without a manifest; inspect it')
    origin = Path(repo) / 'scripts/adaptive' if repo else Path(__file__).resolve().parent
    for name in FILES:
        if not (origin / name).is_file():
            raise FileNotFoundError(origin / name)
    before = {}
    for policy in POLICIES:
        for seed in SEEDS:
            base = ROOT / 'runs' / T13.name / policy / f'seed_{seed}'
            state = read(base / 'run_state.json')
            identity = read(base / 'experiment.json')
            if state.get('last_completed_cycle') != 4 or state.get('active_cycle') is not None:
                raise RuntimeError(f'Expected completed, idle Task13 cycle four: {policy}/{seed}')
            config = identity['config']
            if (config['max_cycles'] != 4 or config['allow_partial_final_cycle']
                    or config['strategy'] != policy or config['seed'] != seed
                    or config['training_initialization'] != 'warm_start'):
                raise RuntimeError(f'Unexpected old experiment identity: {policy}/{seed}')
            summary = base / 'cycle_004/cycle_summary.json'
            if not summary.is_file():
                raise FileNotFoundError(summary)
            before[f'{policy}/{seed}'] = {
                'experiment_sha256': digest(base / 'experiment.json'),
                'cycle4_summary_sha256': digest(summary),
                'cycle4_state_sha256': digest(base / 'run_state.json'),
            }
    (controller / 'source').mkdir(parents=True, exist_ok=True)
    (controller / 'logs').mkdir(exist_ok=True)
    for name in FILES:
        shutil.copy2(origin / name, controller / 'source' / name)
    write(controller / 'manifest.json', {
        'parent_suite': T13.name,
        'parent_source_manifest_sha256': digest(T13 / 'frozen_manifest.json'),
        'parent_plan_sha256': digest(T13 / 'scientific_plan.json'),
        'controller_source_sha256': {name: digest(controller / 'source' / name) for name in FILES},
        'extension_cycles': list(CYCLES),
        'policies': list(POLICIES),
        'seeds': list(SEEDS),
        'cycle5_requires_extend_budget': True,
        'final_cycle_partial': True,
        'pre_extension': before,
    })
    verify(controller)
    print('Prepared Task13 extension controller; original checkpoints preserved')


def verify(controller=CONTROL):
    controller = Path(controller)
    meta = read(controller / 'manifest.json')
    if (meta['parent_suite'] != T13.name or meta['extension_cycles'] != list(CYCLES)
            or tuple(meta['policies']) != POLICIES or tuple(meta['seeds']) != SEEDS):
        raise ValueError('Unexpected Task13 extension plan')
    if digest(T13 / 'frozen_manifest.json') != meta['parent_source_manifest_sha256']:
        raise ValueError('Task13 parent frozen manifest changed')
    if digest(T13 / 'scientific_plan.json') != meta['parent_plan_sha256']:
        raise ValueError('Task13 original scientific plan changed')
    for name in FILES:
        if digest(controller / 'source' / name) != meta['controller_source_sha256'][name]:
            raise ValueError(f'Extension controller source changed: {name}')
    check_parent_source()
    print('Task13 extension source and original frozen experiment verified')


def cases():
    for policy in POLICIES:
        for seed in SEEDS:
            for cycle in CYCLES:
                yield policy, seed, cycle


def runner_args(policy, seed, cycle):
    if policy not in POLICIES or seed not in SEEDS or cycle not in CYCLES:
        raise ValueError('Invalid Task13 extension case')
    args = frozen_task13().runner_args(seed, policy, job_dir=T13)
    if args.count('--max-cycles') != 1:
        raise RuntimeError('Frozen Task13 runner arguments changed')
    args[args.index('--max-cycles') + 1] = '9'
    args += ['--complete-pool', '--device', 'cuda', '--cycle', str(cycle), '--run']
    if cycle == 5:
        args.append('--extend-budget')
    return args


def job(policy, seed, cycle, controller=CONTROL):
    verify(controller)
    command = [sys.executable, str(T13 / 'source/scripts/run_astrominn_closed_loop.py')]
    command.extend(runner_args(policy, seed, cycle))
    env = dict(os.environ)
    env['PYTHONPATH'] = str(T13 / 'source/src') + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run(command, env=env, check=True)


def submit(controller=CONTROL, walltime='00:20:00'):
    verify(controller)
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d', walltime) or walltime == '00:00:00':
        raise ValueError('Invalid walltime')
    controller = Path(controller)
    ledger = controller / 'submitted.json'
    if ledger.exists():
        raise RuntimeError('Extension ledger exists; inspect instead of double-submitting')
    data = {'state': 'submitting_held', 'walltime': walltime, 'jobs': []}
    write(ledger, data)
    parent_job = {}
    for policy, seed, cycle in cases():
        ancestor = parent_job.get((policy, seed))
        name = f'T13ext_{"tail" if policy == "tailor_diversity" else "unif"}_{seed}_{cycle}'
        cmd = ['sbatch', '--parsable', '--hold', '--time', walltime,
               '--job-name', name, '--output', str(controller / 'logs/%x_%j.out'),
               '--error', str(controller / 'logs/%x_%j.err')]
        if ancestor:
            cmd += ['--dependency', f'afterok:{ancestor}']
        cmd += [str(controller / 'source/task13_extension_worker.sbatch'),
                str(controller), policy, str(seed), str(cycle)]
        out = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        if not out.isdecimal():
            raise RuntimeError('Unexpected sbatch response: ' + out)
        parent_job[(policy, seed)] = out
        data['jobs'].append({'policy': policy, 'seed': seed, 'cycle': cycle,
                             'job_id': out, 'parent': ancestor, 'released': False})
        write(ledger, data)
    if len(data['jobs']) != 30:
        raise RuntimeError('Incomplete Task13 extension submission')
    data['state'] = 'held'
    write(ledger, data)
    print('Submitted 30 held Task13 extension jobs in six five-cycle afterok chains')


def release(controller=CONTROL, policy=None, seed=None, cycle=None, all_jobs=False):
    verify(controller)
    ledger = Path(controller) / 'submitted.json'
    data = read(ledger)
    if len(data['jobs']) != 30 or data['state'] not in ('held', 'partially_released', 'released_all'):
        raise RuntimeError('Incomplete or invalid extension ledger')
    if all_jobs:
        if any(x is not None for x in (policy, seed, cycle)):
            raise ValueError('Cannot combine --all with individual selection')
        selection = data['jobs']
    else:
        if policy not in POLICIES or seed not in SEEDS or cycle not in CYCLES:
            raise ValueError('Supply --policy, --seed and --cycle, or --all')
        selection = [j for j in data['jobs']
                     if (j['policy'], j['seed'], j['cycle']) == (policy, seed, cycle)]
    for entry in selection:
        if entry['released']:
            continue
        subprocess.run(['scontrol', 'release', entry['job_id']], check=True)
        entry['released'] = True
        data['state'] = 'partially_released'
        write(ledger, data)
        print('Released', entry['job_id'], entry['policy'], entry['seed'], entry['cycle'])
    if all(j['released'] for j in data['jobs']):
        data['state'] = 'released_all'
        write(ledger, data)


def status(controller=CONTROL):
    controller = Path(controller)
    data = read(controller / 'submitted.json') if (controller / 'submitted.json').exists() else {}
    series = []
    for policy in POLICIES:
        for seed in SEEDS:
            state = read(ROOT / 'runs' / T13.name / policy / f'seed_{seed}/run_state.json')
            cycle = state['last_completed_cycle']
            series.append({'policy': policy, 'seed': seed,
                           'completed_cycle': cycle, 'complete': cycle == 9})
    print(json.dumps({'submitted_jobs': len(data.get('jobs', [])),
                      'released_jobs': sum(j['released'] for j in data.get('jobs', [])),
                      'submission': data.get('state', 'not_submitted'),
                      'trajectories_complete': sum(x['complete'] for x in series),
                      'trajectories': series}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('prepare', 'verify', 'submit', 'release', 'status', 'job'))
    ap.add_argument('policy_pos', nargs='?')
    ap.add_argument('seed_pos', nargs='?', type=int)
    ap.add_argument('cycle_pos', nargs='?', type=int)
    ap.add_argument('--controller', type=Path, default=CONTROL)
    ap.add_argument('--repo', type=Path)
    ap.add_argument('--walltime', default='00:20:00')
    ap.add_argument('--policy', choices=POLICIES)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--cycle', type=int)
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.controller, a.repo)
    elif a.command == 'verify':
        verify(a.controller)
    elif a.command == 'submit':
        submit(a.controller, a.walltime)
    elif a.command == 'release':
        release(a.controller, a.policy, a.seed, a.cycle, a.all)
    elif a.command == 'status':
        status(a.controller)
    elif a.command == 'job':
        job(a.policy_pos, a.seed_pos, a.cycle_pos, a.controller)


if __name__ == '__main__':
    main()
