#!/usr/bin/env python3
"""Freeze Task7A separately; audit its identical object split and source bootstrap."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from task7a_build import build_snapshot_files, SUITE, BOOT

REFERENCE='astrominn_closed_loop_warm_v1'
SEEDS=(135,246,357)


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda:f.read(1<<20),b''):h.update(piece)
    return h.hexdigest()


def freeze(repo:Path,job_dir:Path,registry:Path)->None:
    repo=repo.resolve();job_dir=job_dir.resolve()
    persistent=Path('/projects/bcrv/dkourtesis/applecider_adaptive').resolve()
    if not job_dir.is_relative_to(persistent):raise ValueError('Experiment must be under persistent root')
    if not registry.is_file():raise FileNotFoundError(registry)
    generated,simulation,science=build_snapshot_files(repo)
    files=[*sorted((repo/'src/al_pipeline').rglob('*.py')),
           repo/'scripts/adaptive/task7a.py',
           *sorted((repo/'scripts/adaptive').glob('task7a_*.py')),
           *sorted((repo/'scripts/adaptive').glob('task7a_*.sh')),
           *sorted((repo/'scripts/adaptive').glob('task7a_*.sbatch'))]
    if not files or any(not x.is_file() for x in files):raise FileNotFoundError('Task7A source incomplete')
    content={str(f.relative_to(repo)):f.read_bytes() for f in files}
    content['scripts/run_task7a_closed_loop.py']=generated.encode('utf-8')
    content['src/al_pipeline/astronomy/experiments/simulated.py']=simulation.encode('utf-8')
    content['src/al_pipeline/astronomy/experiments/temperature_acquisition.py']=science.encode('utf-8')
    # Experimental helper is stored under scripts/adaptive in the editable repo;
    # its copy under src/ in frozen snapshot is the one imported by workers.
    sha={name:hashlib.sha256(value).hexdigest() for name,value in content.items()}
    reg=registry.read_bytes()
    contract={'schema':'task7a_entropy_temperature_v1','suite_id':SUITE,
              'reference_suite':REFERENCE,'bootstrap_suite':BOOT,
              'bootstrap_exponent':0.85,'cycle_exponent':0.85,'checkpoint':'loss_best',
              'training_initialization':'warm_start','object_weighting':'per_sample',
              'policy':'entropy','seeds':list(SEEDS),'split_seed':135,
              'max_cycles_configured':9,'cycles_queued_initially':5,'batch_size':473,
              'temperature_objective':'unweighted validation object-level NLL',
              'temperature_fit_timing':'prior checkpoint before each acquisition',
              'temperature_bounds':[0.05,20.0],
              'temperature_applies_to':'entropy acquisition only; not training/eval',
              'registry_sha256':hashlib.sha256(reg).hexdigest()}
    job_dir.mkdir(parents=True,exist_ok=True);(job_dir/'logs').mkdir(exist_ok=True)
    snapshot=job_dir/'source_snapshot.json'
    if snapshot.exists():
        if json.loads(snapshot.read_text())!=sha:raise ValueError('Task7A source changed after freeze; use a new suite ID')
        if (job_dir/'registry.toml').read_bytes()!=reg:raise ValueError('Task7A registry changed')
        if json.loads((job_dir/'contract.json').read_text())!=contract:raise ValueError('Task7A contract changed')
        verify(job_dir);return
    if (job_dir/'source').exists() or (job_dir/'registry.toml').exists() or (job_dir/'contract.json').exists():
        raise RuntimeError('Partial Task7A snapshot: inspect rather than overwrite')
    for name,value in content.items():
        path=job_dir/'source'/name
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    (job_dir/'registry.toml').write_bytes(reg)
    (job_dir/'contract.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
    snapshot.write_text(json.dumps(sha,indent=2,sort_keys=True)+'\n')
    verify(job_dir)


def verify(job_dir:Path)->None:
    source=job_dir/'source';snapshot=job_dir/'source_snapshot.json'
    if not snapshot.is_file():raise FileNotFoundError('Task7A not prepared')
    manifest=json.loads(snapshot.read_text())
    for name,expected in manifest.items():
        file=source/name
        if not file.is_file() or digest(file)!=expected:raise ValueError(f'Frozen source damaged: {file}')
    c=json.loads((job_dir/'contract.json').read_text())
    if (c['suite_id']!=SUITE or c['bootstrap_suite']!=BOOT or
        c['cycle_exponent']!=0.85 or c['checkpoint']!='loss_best' or
        c['policy']!='entropy' or c['temperature_bounds']!=[.05,20.]):
        raise ValueError('Incorrect Task7A contract')
    if digest(job_dir/'registry.toml')!=c['registry_sha256']:
        raise ValueError('Registry hash mismatch')


def normalized_csv(file:Path,ignore=frozenset())->list:
    with file.open(newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:raise ValueError(f'Missing header {file}')
        return sorted(tuple(sorted((k,v) for k,v in row.items() if k not in ignore)) for row in reader)


def check_splits(persistent:Path,data:Path)->None:
    old=persistent/'runs'/BOOT/'shared'/'historical_replay_split'
    current=persistent/'runs'/SUITE/'shared'/'historical_replay_split'
    left=json.loads((old/'split_summary.json').read_text())
    right=json.loads((current/'split_summary.json').read_text())
    for key in ('split_seed','initial_fraction','minimum_initial_per_class','historical_objects',
        'historical_samples','initial_objects','initial_samples','candidate_objects',
        'candidate_samples','maximum_alert','historical_object_label_counts',
        'initial_object_label_counts','candidate_object_label_counts',
        'initial_sample_label_counts','candidate_sample_label_counts','overlap_checks'):
        if left.get(key)!=right.get(key):raise ValueError(f'Historical split mismatch: {key}')
    for name in ('initial_labeled_manifest.csv','hidden_labels.csv'):
        if normalized_csv(old/name)!=normalized_csv(current/name):raise ValueError(f'Split mismatch: {name}')
    path=Path('candidate_pool')/'candidate_pool.csv'
    if normalized_csv(old/path,frozenset({'pool_id'}))!=normalized_csv(current/path,frozenset({'pool_id'})):
        raise ValueError('Candidate pool differs from original U0')
    for seed in SEEDS:
        folder=data/'runs'/BOOT/'shared'/f'seed_{seed}'/'bootstrap_training'
        ck=folder/'best_vit_tower.pth'; sm=folder/'training_summary.json'
        if not ck.is_file() or not sm.is_file():raise FileNotFoundError(f'Missing Experiment B L0: {folder}')
        record=json.loads(sm.read_text())
        if (record['policy']['class_weight_exponent']!=0.85 or
            record['policy']['class_weight_source']!='training_manifest' or
            record['policy']['new_object_weighting']!='per_sample' or
            record['initial_checkpoint'] is not None or record['seed']!=seed):
            raise ValueError(f'Wrong bootstrap provenance: {sm}')
    print('Task7A L0/U0 matches original; all three seed-specific .85 bootstraps verified')
