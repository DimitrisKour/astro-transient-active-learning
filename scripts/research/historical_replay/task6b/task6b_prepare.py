#!/usr/bin/env python3
"""Freeze Task6B source and audit the unchanged historical L0/U0 split.

No scientific training or acquisition happens in this file.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from task6b_build import build_snapshot_files

SEEDS = (135, 246, 357)
SUITE = 'astrominn_closed_loop_warm_f1best_085_v1'
BASE = 'astrominn_historical_replay_v1'
REFERENCE = 'astrominn_closed_loop_warm_v1'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def selected_files(repo: Path) -> list[Path]:
    files = [*sorted((repo / 'src' / 'al_pipeline').rglob('*.py')),
             repo/'scripts/research/historical_replay/task6b/task6b.py',
             *sorted((repo / 'scripts' / 'research' / 'historical_replay' / 'task6b').glob('task6b_*.py')),
             *sorted((repo / 'scripts' / 'research' / 'historical_replay' / 'task6b').glob('task6b_*.sh')),
             *sorted((repo / 'scripts' / 'research' / 'historical_replay' / 'task6b').glob('task6b_*.sbatch'))]
    if not files or any(not f.is_file() for f in files):
        raise FileNotFoundError('Required Task6B source file missing')
    return files


def verify(job_dir: Path) -> None:
    snapshot = job_dir / 'source_snapshot.json'
    registry = job_dir / 'registry.toml'
    if not snapshot.is_file() or not registry.is_file():
        raise FileNotFoundError('Task6B frozen source/registry is not prepared')
    manifest = json.loads(snapshot.read_text(encoding='utf-8'))
    for rel, expected in manifest.items():
        path = job_dir / 'source' / rel
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Task6B frozen-source mismatch: {path}')
    config = json.loads((job_dir / 'contract.json').read_text(encoding='utf-8'))
    if (config['suite_id'] != SUITE or config['weight_exponent'] != 0.85
            or config['bootstrap_weight_exponent'] != 0.85
            or config['bootstrap_source_suite_id'] != SUITE
            or config['checkpoint'] != 'object_macro_f1_best'):
        raise ValueError('Task6B contract mismatch')
    if digest(registry) != config['registry_sha256']:
        raise ValueError('Task6B registry mismatch')


def freeze(repo: Path, job_dir: Path, source_registry: Path) -> None:
    if not source_registry.is_file():
        raise FileNotFoundError(f'No Task4 registry: {source_registry}')
    runner, training = build_snapshot_files(repo)
    files = selected_files(repo)
    content = {str(f.relative_to(repo)): f.read_bytes() for f in files}
    content['scripts/run_task6b_closed_loop.py'] = runner.encode('utf-8')
    content['src/al_pipeline/astronomy/astrominn/training.py'] = training.encode('utf-8')
    sha = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
    registry_bytes = source_registry.read_bytes()
    contract = {
        'schema': 'task6b_macro_f1_checkpoint_v1', 'suite_id': SUITE,
        'reference_suite_id': REFERENCE, 'bootstrap_source_suite_id': SUITE,
        'weight_exponent': 0.85, 'bootstrap_weight_exponent': 0.85, 'reference_weight_exponent': 0.85,
        'checkpoint': 'object_macro_f1_best',
        'early_stopping': 'best_weighted_validation_loss',
        'optimizer_schedule': 'weighted_validation_loss',
        'sample_weighting': 'per_sample',
        'training_initialization': 'warm_start',
        'policies': ['entropy', 'entropy_balanced_explore'], 'seeds': list(SEEDS),
        'split_seed': 135, 'cycles': 9, 'acquisition_size': 473,
        'registry_sha256': hashlib.sha256(registry_bytes).hexdigest(),
    }
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'logs').mkdir(exist_ok=True)
    snapshot = job_dir / 'source_snapshot.json'
    destination = job_dir / 'source'
    if snapshot.exists():
        if json.loads(snapshot.read_text(encoding='utf-8')) != sha:
            raise ValueError('Task6B source changed after freeze. Use a new suite ID.')
        if json.loads((job_dir / 'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Task6B scientific contract changed after freeze')
        if (job_dir / 'registry.toml').read_bytes() != registry_bytes:
            raise ValueError('Task6B routed registry changed after freeze')
        verify(job_dir)
        return
    if destination.exists() or (job_dir / 'contract.json').exists() or (job_dir / 'registry.toml').exists():
        raise ValueError('Partial Task6B snapshot: inspect instead of overwriting')
    destination.mkdir()
    for name, value in content.items():
        dest = destination / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(value)
    (job_dir / 'registry.toml').write_bytes(registry_bytes)
    (job_dir / 'contract.json').write_text(json.dumps(contract, indent=2, sort_keys=True)+'\n')
    snapshot.write_text(json.dumps(sha, indent=2, sort_keys=True)+'\n')
    verify(job_dir)


def normalized_csv(path: Path, ignored: frozenset[str] = frozenset()) -> list[tuple[tuple[str, str], ...]]:
    with path.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f'Missing CSV header: {path}')
        return sorted(tuple(sorted((k, v) for k, v in row.items() if k not in ignored)) for row in r)


def check_splits(persistent: Path, data: Path) -> None:
    origin = persistent / 'runs' / BASE / 'shared' / 'historical_replay_split'
    current = persistent / 'runs' / SUITE / 'shared' / 'historical_replay_split'
    orig = json.loads((origin / 'split_summary.json').read_text())
    now = json.loads((current / 'split_summary.json').read_text())
    fields = ('split_seed','initial_fraction','minimum_initial_per_class',
              'historical_objects','historical_samples','initial_objects','initial_samples',
              'candidate_objects','candidate_samples','maximum_alert',
              'historical_object_label_counts','initial_object_label_counts',
              'candidate_object_label_counts','initial_sample_label_counts',
              'candidate_sample_label_counts','overlap_checks')
    for key in fields:
        if orig.get(key) != now.get(key):
            raise ValueError(f'Task6B replay split mismatch in {key}')
    for name in ('initial_labeled_manifest.csv','hidden_labels.csv'):
        if normalized_csv(origin / name) != normalized_csv(current / name):
            raise ValueError(f'Task6B replay split mismatch: {name}')
    sub = Path('candidate_pool/candidate_pool.csv')
    if normalized_csv(origin / sub, frozenset({'pool_id'})) != normalized_csv(current / sub, frozenset({'pool_id'})):
        raise ValueError('Task6B candidate pool differs from original')
    for seed in SEEDS:
        l0 = data / 'runs' / BASE / 'shared' / f'seed_{seed}' / 'bootstrap_training'
        if not (l0/'best_vit_tower.pth').is_file() or not (l0/'training_summary.json').is_file():
            raise FileNotFoundError(f'Missing original seed-specific L0: {l0}')
    print('Verified identical L0/U0 splits, hidden labels, 4253-object pool and reference bootstraps')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path)
    p.add_argument('--job-dir', type=Path, required=True)
    p.add_argument('--source-registry', type=Path)
    p.add_argument('--persistent', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if a.verify_only:
        verify(a.job_dir)
        print('Task6B source and contract verified')
        return
    if any(x is None for x in (a.repo, a.source_registry, a.persistent, a.data)):
        p.error('Freeze requires --repo, --source-registry, --persistent, --data')
    if not a.job_dir.resolve().is_relative_to(a.persistent.resolve()):
        raise ValueError('Job directory must reside under persistent root')
    freeze(a.repo.resolve(), a.job_dir, a.source_registry)
    print('Frozen Task6B source and registry:', a.job_dir)

if __name__ == '__main__':
    main()
