#!/usr/bin/env python3
"""Task16: full-pool 5% vs 20% acquisition, with existing 10% reference.

Two independent policies x three seeds x (18 + 5) cycles = 138 fits.
Preparing, verifying, preflighting and piloting do NOT submit training.
Held Slurm chains are released only by an explicit release command.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

POLICIES = ('random', 'entropy')
SEEDS = (135, 246, 357)
SUITE = 'astrominn_task16_batchsize_sensitivity_v1'
PERSISTENT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
JOB = PERSISTENT / 'closed_loop' / SUITE
TASK11 = PERSISTENT / 'closed_loop/astrominn_task11_advanced_acquisition_v1'
TASK4 = 'astrominn_closed_loop_warm_v1'
BASE = 'astrominn_historical_replay_v1'
EXCLUDE = PERSISTENT / 'runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv'
HISTORICAL_OBJECTS = 4729
L0_OBJECTS = 476
U0_OBJECTS = 4253
SCHEDULES = {'05': 0.05, '20': 0.20}
REFERENCE = {'10': 0.10}


def suite_id(fraction_key):
    if fraction_key not in SCHEDULES:
        raise ValueError(f'Unsupported fraction: {fraction_key}')
    return f'astrominn_task16_batch{fraction_key}_v1'


def schedule(fraction_key, *, historical=HISTORICAL_OBJECTS, unlabeled=U0_OBJECTS):
    if fraction_key not in SCHEDULES:
        raise ValueError(f'Invalid acquisition fraction: {fraction_key}')
    batch = math.ceil(historical * SCHEDULES[fraction_key])
    cycles = math.ceil(unlabeled / batch)
    remainder = unlabeled - batch * (cycles - 1)
    if not 1 <= remainder <= batch:
        raise ValueError('Invalid final remainder')
    return {'fraction': SCHEDULES[fraction_key], 'batch': batch,
            'cycles': cycles, 'last_batch': remainder,
            'total_acquired': unlabeled, 'final_labeled': historical}


def jobs():
    for frac in SCHEDULES:
        for policy in POLICIES:
            for seed in SEEDS:
                for cycle in range(1, schedule(frac)['cycles'] + 1):
                    yield frac, policy, seed, cycle


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for piece in iter(lambda: fh.read(1 << 20), b''):
            hasher.update(piece)
    return hasher.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temp.replace(path)


def source_files(folder):
    return sorted(p for p in Path(folder).rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.sh', '.sbatch', '.toml'))


def source_hashes(folder):
    folder = Path(folder)
    return {str(p.relative_to(folder)): sha(p) for p in source_files(folder)}


def plan():
    schedules = {name: schedule(name) for name in SCHEDULES}
    total_jobs = sum(s['cycles'] for s in schedules.values()) * len(POLICIES) * len(SEEDS)
    return {
        'suite': SUITE,
        'comparison': 'full-pool acquisition cadence 5% vs 10% vs 20%',
        'policies': list(POLICIES), 'seeds': list(SEEDS),
        'schedules': schedules,
        'existing_reference': {
            'suite': TASK4, 'fraction': 0.10, 'batch': 473,
            'cycles': 9, 'last_batch': 469, 'new_fits': 0,
            'comparability': 'reuse complete random/entropy Task4 trajectories per seed'},
        'planned_new_training_jobs': total_jobs,
        'planned_new_trajectories': len(POLICIES) * len(SEEDS) * len(SCHEDULES),
        'historical_objects': HISTORICAL_OBJECTS,
        'initial_objects': L0_OBJECTS,
        'candidate_objects': U0_OBJECTS,
        'same_L0_and_U0_as_Task4': True,
        'split_seed': 135,
        'bootstrap_source_suite': BASE,
        'initialization': 'warm_start_previous_cycle_loss_best',
        'training': {'max_epochs': 50, 'patience': 10, 'batch_size': 512,
                     'validation_batch_size': 2048, 'learning_rate': 0.0001,
                     'sample_weighting': 'per_sample',
                     'class_weight_source': 'current_training_manifest'},
        'acquisition': 'fraction of all 4729 historical objects, including L0',
        'completion': 'all 4253 originally unlabeled objects acquired; partial final batch',
        'evaluation': 'fixed validation only; test never consulted for selection',
        'source_parent': 'frozen and manifest-verified Task11 source',
        'submission': 'held jobs; independent fraction/policy/seed chains; afterok within chains',
        'interpretation': 'acquisition cadence and frequency of warm-start training vary jointly; compare against number of acquired objects, not cycle index',
    }


def routed_registry(text):
    import tomllib
    origin = tomllib.loads(text)
    match = re.search(r'(?ms)^\[storage\]\s*\n(.*?)(?=^\[|\Z)', text)
    if match is None:
        raise ValueError('Missing [storage] section')
    section = match.group(0)
    for key, val in (('persistent_root', PERSISTENT), ('data_root', DATA)):
        section, count = re.subn(rf'(?m)^\s*{key}\s*=.*$', f'{key} = "{val}"', section)
        if count != 1:
            raise ValueError(f'Missing registry key {key}')
    edited = text[:match.start()] + section + text[match.end():]
    parsed = tomllib.loads(edited)
    if {k:v for k,v in origin.items() if k != 'storage'} != {k:v for k,v in parsed.items() if k != 'storage'}:
        raise ValueError('Non-storage registry changed')
    return edited


def prepare(overlay):
    from task16_patch import patch_source
    if (JOB / 'frozen_manifest.json').is_file():
        verify()
        print('Already prepared')
        return
    if (JOB / 'source').exists() or (JOB / 'registry.toml').exists():
        raise RuntimeError('Partial Task16 preparation; inspect rather than overwrite')
    parent_manifest = TASK11 / 'frozen_manifest.json'
    if load(parent_manifest) != source_hashes(TASK11 / 'source'):
        raise RuntimeError('Parent Task11 frozen source differs from its manifest')
    for seed in SEEDS:
        checkpoint = DATA / 'runs' / BASE / 'shared' / f'seed_{seed}/bootstrap_training/best_vit_tower.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if not EXCLUDE.is_file():
        raise FileNotFoundError(EXCLUDE)
    overlay = Path(overlay).resolve()
    JOB.mkdir(parents=True, exist_ok=True)
    (JOB / 'logs').mkdir(exist_ok=True)
    temp = JOB / 'source.tmp'
    if temp.exists():
        raise RuntimeError('Task16 staging directory already exists')
    try:
        shutil.copytree(TASK11 / 'source', temp, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for old in (temp / 'scripts/adaptive').glob('task11*'):
            old.unlink()
        for src in (overlay / 'scripts/adaptive').glob('task16*'):
            shutil.copy2(src, temp / 'scripts/adaptive' / src.name)
        changed = patch_source(temp, overlay)
        import ast
        for p in source_files(temp):
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        manifest = source_hashes(temp)
        temp.rename(JOB / 'source')
        (JOB / 'registry.toml').write_text(
            routed_registry((TASK11 / 'registry.toml').read_text()), encoding='utf-8')
        dump(JOB / 'scientific_plan.json', plan())
        dump(JOB / 'source_parent.json',
             {'suite': 'Task11', 'manifest_sha256': sha(parent_manifest), 'patched_files': changed})
        dump(JOB / 'frozen_manifest.json', manifest)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    verify()


def verify(job=None):
    job = JOB if job is None else Path(job)
    if load(job / 'scientific_plan.json') != plan():
        raise RuntimeError('Scientific plan mismatch')
    if load(job / 'frozen_manifest.json') != source_hashes(job / 'source'):
        raise RuntimeError('Task16 frozen source mismatch')
    if not (job / 'registry.toml').is_file():
        raise FileNotFoundError(job / 'registry.toml')
    print(f'Task16 frozen source verified ({len(source_files(job / "source"))} files)')


def runner_args(fraction_key, policy, seed):
    if fraction_key not in SCHEDULES or policy not in POLICIES or seed not in SEEDS:
        raise ValueError('Invalid Task16 fraction/policy/seed')
    cfg = schedule(fraction_key)
    return ['--registry', str(JOB / 'registry.toml'),
            '--suite-id', suite_id(fraction_key),
            '--bootstrap-source-suite-id', BASE, '--require-existing-bootstrap',
            '--training-initialization', 'warm_start', '--strategies', policy,
            '--seed', str(seed), '--split-seed', '135', '--initial-fraction', '0.10',
            '--minimum-initial-per-class', '5', '--acquisition-fraction', str(cfg['fraction']),
            '--max-cycles', str(cfg['cycles']), '--complete-pool',
            '--max-epochs', '50', '--patience', '10', '--batch-size', '512',
            '--validation-batch-size', '2048', '--learning-rate', '0.0001',
            '--train-workers', '0', '--validation-workers', '0',
            '--inference-workers', '0', '--evaluation-workers', '0',
            '--assert-excluded-manifest', str(EXCLUDE)]


def manifest_ids(csv_path, id_col):
    with Path(csv_path).open(newline='', encoding='utf-8') as stream:
        return {row[id_col] for row in csv.DictReader(stream)}


def validate_replay(fraction_key):
    # Force imports from our immutable source tree, never an editable checkout.
    frozen_src = str(JOB / 'source/src')
    if frozen_src not in sys.path:
        sys.path.insert(0, frozen_src)
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    registry = DatasetRegistry.from_toml(JOB / 'registry.toml')
    ds = registry.resolve_profile('astrominn_ztf').dataset
    task4_root = PERSISTENT / 'runs' / TASK4 / 'shared/historical_replay_split'
    root = PERSISTENT / 'runs' / suite_id(fraction_key) / 'shared/historical_replay_split'
    l0 = root / 'initial_labeled_manifest.csv'
    u0 = root / 'candidate_pool/candidate_pool.csv'
    if not l0.is_file() or not u0.is_file():
        raise FileNotFoundError('Task16 split missing: run preflight first')
    if manifest_ids(l0, ds.object_id_column) != manifest_ids(task4_root / 'initial_labeled_manifest.csv', ds.object_id_column):
        raise ValueError('Task16 L0 differs from Task4')
    current = CandidatePoolManifest.from_csv(u0)
    reference = CandidatePoolManifest.from_csv(task4_root / 'candidate_pool/candidate_pool.csv')
    if tuple(current.object_ids) != tuple(reference.object_ids):
        raise ValueError('Task16 U0 IDs/order differ from Task4')
    if len(current.object_ids) != U0_OBJECTS:
        raise ValueError('Unexpected U0 size')
    if len(manifest_ids(l0, ds.object_id_column)) != L0_OBJECTS:
        raise ValueError('Unexpected L0 size')
    return current


def preflight():
    verify()
    env = dict(os.environ,
               PYTHONPATH=str(JOB / 'source/src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    runner = JOB / 'source/scripts/run_astrominn_closed_loop.py'
    for fraction in SCHEDULES:
        for seed in SEEDS:
            subprocess.run([sys.executable, str(runner), *runner_args(fraction, 'entropy', seed)],
                           env=env, check=True)
        validate_replay(fraction)
    print('Task16 preflight passed: both full-pool schedules and all three seeds, no GPU training')


def pilot(seed=135):
    """Single real GPU inference for a joint four-policy/fraction acquisition pilot."""
    verify()
    if seed not in SEEDS:
        raise ValueError('Unknown seed')
    frozen_src = str(JOB / 'source/src')
    if frozen_src not in sys.path:
        sys.path.insert(0, frozen_src)
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (
        DEFAULT_LEGACY_CODE_DIR, predict_astrominn_vit_pool)
    from al_pipeline.core.pool import Pool
    from al_pipeline.core.scoring.entropy import EntropyScorer
    from al_pipeline.core.scoring.random import RandomScorer
    from al_pipeline.core.selection import TopKSelector
    import numpy as np
    results = JOB / 'pilot' / f'seed_{seed}' / 'result.json'
    if results.is_file():
        print(json.dumps(load(results), indent=2)); return
    for fraction in SCHEDULES:
        validate_replay(fraction)
    registry = DatasetRegistry.from_toml(JOB / 'registry.toml')
    profile = registry.resolve_profile('astrominn_ztf').profile
    artifact = Path(profile.options['checkpoint']).parent
    cp = DATA / 'runs' / BASE / 'shared' / f'seed_{seed}/bootstrap_training/best_vit_tower.pth'
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    root = PERSISTENT / 'runs' / suite_id('05') / 'shared/historical_replay_split/candidate_pool'
    obj_ids = tuple(CandidatePoolManifest.from_csv(root / 'candidate_pool.csv').object_ids)
    start = time.perf_counter()
    prediction = predict_astrominn_vit_pool(
        pool_path=root, object_ids=obj_ids, device='cuda', batch_size=64,
        num_workers=0, legacy_code_dir=DEFAULT_LEGACY_CODE_DIR,
        artifact_dir=artifact, vit_checkpoint=cp)
    inference_seconds = time.perf_counter() - start
    if tuple(prediction.object_ids) != obj_ids:
        raise ValueError('Task16 pilot inference object order mismatch')
    pool = Pool.from_ids(obj_ids)
    data = []
    for fraction in SCHEDULES:
        for policy in POLICIES:
            k = schedule(fraction)['batch']
            start = time.perf_counter()
            if policy == 'random':
                scorer = RandomScorer(np.random.default_rng(np.random.SeedSequence([seed, 1])))
                scores = scorer.score(len(obj_ids))
            else:
                scores = EntropyScorer().score(prediction.predictions)
            selected = TopKSelector().select(ids=obj_ids, scores=scores, k=k, pool=pool)
            seconds = time.perf_counter() - start
            if len(selected.ids) != k or len(set(selected.ids)) != k:
                raise RuntimeError('Task16 acquisition pilot budget/uniqueness mismatch')
            data.append({'fraction': fraction, 'policy': policy, 'seed': seed,
                         'budget': k, 'unique_selected': True,
                         'selection_seconds': seconds, 'selected_ids': selected.ids})
    dump(results, {'seed': seed, 'inference_seconds': inference_seconds,
                   'checkpoint_sha256': sha(cp), 'valid_replay': True,
                   'validation_labels_accessed': False, 'test_labels_accessed': False,
                   'cases': data})
    print(json.dumps({'seed': seed, 'inference_seconds': inference_seconds,
                      'cases': [{k:v for k,v in d.items() if k != 'selected_ids'} for d in data]}, indent=2))


def pilot_status():
    path = JOB / 'pilot/seed_135/result.json'
    print(json.dumps({'found': path.is_file(), 'path': str(path)}, indent=2))


def filesystem_probe():
    root = DATA / 'runs' / SUITE / 'shared/probe'
    root.mkdir(parents=True, exist_ok=True)
    temp = root / f'{os.getpid()}_probe.tmp'
    payload = os.urandom(16 * 1024 * 1024)
    try:
        with temp.open('wb') as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if temp.read_bytes() != payload:
            raise RuntimeError('Task16 /work/hdd readback mismatch')
        print('Task16 /work/hdd readback passed (16 MiB)')
    finally:
        temp.unlink(missing_ok=True)


def submit(walltime, fraction_filter=None, policy_filter=None, seed_filter=None):
    """Submit complete independent chains as HELD jobs, optionally in stages.

    Slurm submission limits can make 138 simultaneous pending jobs impractical.
    Successfully submitted chains remain recorded, and a failed chain is
    canceled without touching previously recorded chains.
    """
    verify()
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d', walltime) or walltime == '00:00:00':
        raise ValueError('Invalid walltime')
    if fraction_filter is not None and fraction_filter not in SCHEDULES:
        raise ValueError('Invalid submission fraction')
    if policy_filter is not None and policy_filter not in POLICIES:
        raise ValueError('Invalid submission policy')
    if seed_filter is not None and seed_filter not in SEEDS:
        raise ValueError('Invalid submission seed')
    piloted = JOB / 'pilot/seed_135/result.json'
    if not piloted.is_file():
        raise RuntimeError('Task16 GPU acquisition pilot required before submission')
    record = load(piloted)
    expected_cases = {(f, p) for f in SCHEDULES for p in POLICIES}
    if not record.get('valid_replay') or {(x['fraction'], x['policy']) for x in record['cases']} != expected_cases:
        raise ValueError('Task16 pilot incomplete or invalid')
    ledger_path = JOB / 'submitted.json'
    existing = load(ledger_path) if ledger_path.is_file() else {'jobs': [], 'state': 'held_partial'}
    known_cases = {}
    for job in existing['jobs']:
        key = (job['fraction'], job['policy'], job['seed'])
        known_cases.setdefault(key, []).append(job)
    cases = [(f,p,s) for f in SCHEDULES for p in POLICIES for s in SEEDS
             if (fraction_filter is None or f == fraction_filter)
             and (policy_filter is None or p == policy_filter)
             and (seed_filter is None or s == seed_filter)]
    if not cases:
        raise ValueError('No matching Task16 trajectories')
    added = 0
    for fraction, policy, seed in cases:
        key = (fraction, policy, seed)
        if key in known_cases:
            prior = known_cases[key]
            if len(prior) != schedule(fraction)['cycles']:
                raise RuntimeError(f'Incomplete prior submission ledger for {key}; inspect manually')
            print(f'Already submitted, skipping {key}')
            continue
        generated, entries = [], []
        try:
            last = None
            for cycle in range(1, schedule(fraction)['cycles']+1):
                cmd = ['sbatch', '--parsable', '--hold', '--time', walltime,
                       '--job-name', f'T16_{fraction}_{policy}_{seed}_{cycle}',
                       '--output', str(JOB/'logs/%x_%j.out'),
                       '--error', str(JOB/'logs/%x_%j.err')]
                if last:
                    cmd += ['--dependency', f'afterok:{last}']
                cmd += [str(JOB/'source/scripts/adaptive/task16_worker.sbatch'),
                        str(JOB), 'job', fraction, policy, str(seed), str(cycle)]
                response = subprocess.check_output(cmd, text=True).strip().split(';')[0]
                if not response.isdigit():
                    raise RuntimeError('Invalid sbatch response ' + response)
                generated.append(response)
                entries.append({'fraction': fraction, 'policy': policy, 'seed': seed,
                                'cycle': cycle, 'job_id': response,
                                'dependency': f'afterok:{last}' if last else None,
                                'walltime': walltime, 'released': False})
                last = response
        except Exception:
            if generated:
                subprocess.run(['scancel', *generated], check=False)
            raise
        existing['jobs'].extend(entries)
        existing['state'] = ('partially_released' if any(j['released'] for j in existing['jobs'])
                             else 'held' if len(existing['jobs']) == 138 else 'held_partial')
        dump(ledger_path, existing)
        known_cases[key] = entries
        added += len(entries)
        print(f'Submitted HELD chain {key}: {len(entries)} jobs')
    print(f'Task16 submission: {added} new HELD jobs; {len(existing["jobs"])} / 138 total recorded; none released automatically')


def release(fraction=None, policy=None, seed=None, cycle=None, all_jobs=False):
    data = load(JOB / 'submitted.json')
    if all_jobs:
        if any(x is not None for x in (fraction, policy, seed, cycle)):
            raise ValueError('--all cannot be combined with individual filters')
        targets = [j for j in reversed(data['jobs']) if not j['released']]
    else:
        if fraction not in SCHEDULES or policy not in POLICIES or seed not in SEEDS or cycle not in range(1, schedule(fraction)['cycles']+1):
            raise ValueError('Specify --fraction, --policy, --seed, --cycle, or --all')
        targets = [j for j in data['jobs'] if (j['fraction'],j['policy'],j['seed'],j['cycle']) == (fraction,policy,seed,cycle)]
    for j in targets:
        if j['released']:
            continue
        subprocess.run(['scontrol','release',j['job_id']], check=True)
        j['released'] = True
        dump(JOB / 'submitted.json', data)
        print(f"Released {j['job_id']} {j['fraction']} {j['policy']} seed={j['seed']} cycle={j['cycle']}")
    data['state'] = ('released_all' if len(data['jobs']) == 138 and all(x['released'] for x in data['jobs'])
                     else 'partially_released' if any(x['released'] for x in data['jobs'])
                     else data['state'])
    dump(JOB / 'submitted.json', data)


def status():
    trajectories = []
    for fraction in SCHEDULES:
        cycles = schedule(fraction)['cycles']
        for policy in POLICIES:
            for seed in SEEDS:
                path = PERSISTENT / 'runs' / suite_id(fraction) / policy / f'seed_{seed}'
                statefile = path / 'run_state.json'
                done = int(load(statefile).get('last_completed_cycle', 0)) if statefile.is_file() else 0
                count = sum((path/f'cycle_{c:03d}/cycle_summary.json').is_file() for c in range(1, cycles+1))
                trajectories.append({'fraction': fraction, 'policy': policy, 'seed': seed,
                                     'expected_cycles': cycles, 'completed_cycles': done,
                                     'summaries': count, 'complete': done == cycles and count == cycles})
    print(json.dumps({'suite': SUITE,
                      'completed_series': sum(x['complete'] for x in trajectories),
                      'total_series': len(trajectories), 'planned_fits': len(list(jobs())),
                      'submission': load(JOB/'submitted.json')['state'] if (JOB/'submitted.json').exists() else 'not_submitted',
                      'series': trajectories}, indent=2))


def collect(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = [JOB/'scientific_plan.json', JOB/'frozen_manifest.json', JOB/'source_parent.json',
                  JOB/'submitted.json', JOB/'pilot/seed_135/result.json']
    for fraction in SCHEDULES:
        for policy in POLICIES:
            for seed in SEEDS:
                root = PERSISTENT / 'runs' / suite_id(fraction) / policy / f'seed_{seed}'
                artifact = DATA / 'runs' / suite_id(fraction) / policy / f'seed_{seed}'
                candidates += [root/'run_state.json',root/'evaluation_history.json',root/'experiment.json']
                for c in range(0, schedule(fraction)['cycles'] + 1):
                    cy = root / f'cycle_{c:03d}'
                    for name in ('selection.json','revealed_labels.json','cycle_summary.json'):
                        candidates.append(cy/name)
                    evaluation = artifact/f'cycle_{c:03d}/evaluation'
                    for name in ('evaluation_summary.json','validation_object_confusion.csv',
                                 'validation_per_class_metrics.json','validation_sample_confusion.csv'):
                        candidates.append(evaluation/name)
    # Archive metadata for 10% reference too, never overwrite or alter it.
    for policy in POLICIES:
        for seed in SEEDS:
            root = PERSISTENT / 'runs' / TASK4 / policy / f'seed_{seed}'
            candidates += [root/'run_state.json',root/'evaluation_history.json',root/'experiment.json']
            for c in range(0, 10):
                cy = root / f'cycle_{c:03d}'
                for name in ('selection.json', 'cycle_summary.json'):
                    candidates.append(cy/name)
                ev = DATA / 'runs' / TASK4 / policy / f'seed_{seed}' / f'cycle_{c:03d}/evaluation'
                candidates.append(ev/'evaluation_summary.json')
    count = 0
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in dict.fromkeys(candidates):
            if path.is_file():
                if path.is_relative_to(PERSISTENT):
                    name = 'persistent/' + str(path.relative_to(PERSISTENT))
                elif path.is_relative_to(DATA):
                    name = 'data/' + str(path.relative_to(DATA))
                else:
                    raise ValueError('Archive path outside root')
                archive.write(path, name)
                count += 1
    print(json.dumps({'path': str(destination), 'files': count,
                      'bytes': destination.stat().st_size, 'sha256': sha(destination)}, indent=2))


def training_job(fraction, policy, seed, cycle, job):
    if fraction not in SCHEDULES or policy not in POLICIES or seed not in SEEDS or cycle not in range(1,schedule(fraction)['cycles']+1):
        raise ValueError('Invalid Task16 training case')
    verify(job)
    env = dict(os.environ,
               PYTHONPATH=str(Path(job)/'source/src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    runner = Path(job) / 'source/scripts/run_astrominn_closed_loop.py'
    subprocess.run([sys.executable, str(runner), *runner_args(fraction,policy,seed),
                    '--device', 'cuda', '--cycle', str(cycle), '--run'], check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('plan','prepare','verify','preflight','pilot',
                    'pilot-status','filesystem-probe','submit','release','status','collect','job'))
    ap.add_argument('fraction_pos', nargs='?')
    ap.add_argument('policy_pos', nargs='?')
    ap.add_argument('seed_pos', nargs='?', type=int)
    ap.add_argument('cycle_pos', nargs='?', type=int)
    ap.add_argument('--fraction', choices=tuple(SCHEDULES))
    ap.add_argument('--submit-fraction', choices=tuple(SCHEDULES))
    ap.add_argument('--submit-policy', choices=POLICIES)
    ap.add_argument('--submit-seed', type=int, choices=SEEDS)
    ap.add_argument('--policy', choices=POLICIES)
    ap.add_argument('--seed', type=int, default=135)
    ap.add_argument('--cycle', type=int)
    ap.add_argument('--overlay', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--job-dir', type=Path, default=JOB)
    ap.add_argument('--walltime', default='00:35:00')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--destination', type=Path, default=Path('/u/dkourtesis/task16_results.zip'))
    opts = ap.parse_args()
    if opts.command == 'plan': print(json.dumps(plan(), indent=2)); return
    if opts.command == 'prepare': prepare(opts.overlay); return
    if opts.command == 'verify': verify(opts.job_dir); return
    if opts.command == 'preflight': preflight(); return
    if opts.command == 'pilot': pilot(opts.seed); return
    if opts.command == 'pilot-status': pilot_status(); return
    if opts.command == 'filesystem-probe': filesystem_probe(); return
    if opts.command == 'submit': submit(opts.walltime, opts.submit_fraction, opts.submit_policy, opts.submit_seed); return
    if opts.command == 'release': release(opts.fraction,opts.policy,opts.seed,opts.cycle,opts.all); return
    if opts.command == 'status': status(); return
    if opts.command == 'collect': collect(opts.destination); return
    if opts.command == 'job': training_job(opts.fraction_pos,opts.policy_pos,opts.seed_pos,opts.cycle_pos,opts.job_dir); return

if __name__ == '__main__': main()
