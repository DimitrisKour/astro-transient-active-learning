#!/usr/bin/env python3
"""Task17: uniform-random object-level L0, new bootstraps, 3-cycle AL.

Prepare/verify/preflight never train; bootstrap jobs and AL jobs are HELD
until explicitly released. Task17 does not modify any Task4/11-16 run.
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
from pathlib import Path

POLICIES = ('random', 'entropy')
SEEDS = (135, 246, 357)
SPLIT_SEED = 17017
L0_SIZE = 476
HISTORICAL_SIZE = 4729
U0_SIZE = HISTORICAL_SIZE - L0_SIZE
BATCH_SIZE = math.ceil(HISTORICAL_SIZE * 0.10)
CYCLES = 3  # Original scoped sensitivity study: 3 acquisitions, not full-pool exhaustion
SUITE = 'astrominn_task17_random_l0_v1'
BOOTSTRAP = 'astrominn_task17_random_l0_bootstrap_v1'
PERSISTENT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
JOB = PERSISTENT / 'closed_loop' / SUITE
TASK11 = PERSISTENT / 'closed_loop/astrominn_task11_advanced_acquisition_v1'
TASK4 = 'astrominn_closed_loop_warm_v1'
EXCLUDE = PERSISTENT / 'runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(1 << 20), b''):
            h.update(piece)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def source_files(root):
    root = Path(root)
    return sorted(p for p in root.rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.sh', '.sbatch', '.toml'))


def source_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in source_files(root)}


def plan():
    return {
        'suite': SUITE, 'bootstrap_suite': BOOTSTRAP,
        'question': 'How robust are random and entropy active learning to an unstratified L0?',
        'split': {'mode': 'uniform_random_without_rejection_or_class_quotas',
                  'seed_fixed_across_training_seeds': SPLIT_SEED,
                  'l0_objects': L0_SIZE, 'u0_objects': U0_SIZE,
                  'historical_objects': HISTORICAL_SIZE,
                  'class_coverage_guaranteed': False},
        'policies': list(POLICIES), 'seeds': list(SEEDS),
        'acquisition': {'fraction_of_all_historical_objects': .10,
                        'regular_batch_objects': BATCH_SIZE,
                        'cycles': CYCLES,
                        'last_batch_objects': BATCH_SIZE,
                        'total_acquired_objects': CYCLES * BATCH_SIZE,
                        'final_labeled_objects': L0_SIZE + CYCLES * BATCH_SIZE,
                        'full_pool_coverage': False},
        'new_bootstrap_fits': len(SEEDS),
        'new_active_learning_fits': len(POLICIES)*len(SEEDS)*CYCLES,
        'reference': {'suite': TASK4, 'split': 'original_class_stratified_L0',
                      'trajectories': 'existing_random_and_entropy_three_seeds'},
        'class_weighting': 'training_manifest_counts; zero-count classes smoothed to count 1 only in weight formula',
        'initialization': 'fresh_random_ViT_for_new_L0_then_warm_start',
        'training': {'epochs': 50, 'patience': 10, 'batch_size': 512,
                     'validation_batch_size': 2048, 'learning_rate': .0001,
                     'sample_weighting': 'per_sample', 'class_weights': 'training_manifest_only'},
        'evaluation': 'unchanged historical validation; no test for selection',
        'source_parent': 'manifest_verified_Task11_source',
        'slurm': 'three independent bootstrap jobs; six independent three-cycle afterok chains; all held until explicit release',
        'interpretation': 'new L0 changes initial examples, class proportions, bootstrap fit and U0 membership; not just shuffle/order',
    }


def routed_registry(text):
    import tomllib
    original = tomllib.loads(text)
    match = re.search(r'(?ms)^\[storage\]\s*\n(.*?)(?=^\[|\Z)', text)
    if not match:
        raise ValueError('Missing [storage]')
    section = match.group(0)
    for key, value in (('persistent_root', PERSISTENT), ('data_root', DATA)):
        section, count = re.subn(rf'(?m)^\s*{key}\s*=.*$',
                                 f'{key} = "{value}"', section)
        if count != 1:
            raise ValueError('Missing registry key ' + key)
    result = text[:match.start()] + section + text[match.end():]
    parsed = tomllib.loads(result)
    if {k:v for k,v in original.items() if k != 'storage'} != {k:v for k,v in parsed.items() if k != 'storage'}:
        raise ValueError('Registry non-storage changed')
    return result


def prepare(overlay):
    from task17_patch import patch_source
    if (JOB / 'frozen_manifest.json').is_file():
        verify()
        print('Already prepared')
        return
    if (JOB/'source').exists() or (JOB/'registry.toml').exists():
        raise RuntimeError('Partial Task17 preparation exists; inspect manually')
    parent_manifest = TASK11/'frozen_manifest.json'
    if load(parent_manifest) != source_hashes(TASK11/'source'):
        raise RuntimeError('Task11 source parent differs from manifest')
    if not EXCLUDE.is_file():
        raise FileNotFoundError(EXCLUDE)
    overlay = Path(overlay).resolve()
    JOB.mkdir(parents=True, exist_ok=True)
    (JOB/'logs').mkdir(exist_ok=True)
    temp = JOB/'source.tmp'
    if temp.exists():
        raise RuntimeError('Task17 source staging exists')
    try:
        shutil.copytree(TASK11/'source', temp, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for old in (temp/'scripts/adaptive').glob('task11*'):
            old.unlink()
        (temp/'scripts/research/historical_replay/task17').mkdir(parents=True, exist_ok=True)
        for src in (overlay/'scripts/research/historical_replay/task17').glob('task17*'):
            shutil.copy2(src, temp/'scripts/research/historical_replay/task17'/src.name)
        patched = patch_source(temp, overlay)
        import ast
        for f in source_files(temp):
            if f.suffix == '.py':
                ast.parse(f.read_text(encoding='utf-8'), filename=str(f))
        hashes = source_hashes(temp)
        temp.rename(JOB/'source')
        (JOB/'registry.toml').write_text(routed_registry((TASK11/'registry.toml').read_text()), encoding='utf-8')
        dump(JOB/'scientific_plan.json', plan())
        dump(JOB/'source_parent.json', {'task11_manifest_sha256': sha(parent_manifest),
                                        'patched_files': patched})
        dump(JOB/'frozen_manifest.json', hashes)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    verify()


def verify(job=None):
    job = JOB if job is None else Path(job)
    if load(job/'scientific_plan.json') != plan():
        raise RuntimeError('Task17 scientific plan mismatch')
    if load(job/'frozen_manifest.json') != source_hashes(job/'source'):
        raise RuntimeError('Task17 frozen source mismatch')
    if not (job/'registry.toml').is_file():
        raise FileNotFoundError(job/'registry.toml')
    print(f'Task17 frozen source verified ({len(source_files(job/"source"))} files)')


def bootstrap_checkpoint(seed):
    if seed not in SEEDS:
        raise ValueError('Invalid bootstrap seed')
    return DATA/'runs'/BOOTSTRAP/'shared'/f'seed_{seed}/bootstrap_training/best_vit_tower.pth'


def runner_args(seed, policy='entropy', *, mode='preflight'):
    if seed not in SEEDS or policy not in POLICIES or mode not in ('preflight','bootstrap','check-bootstrap','cycle'):
        raise ValueError('Invalid Task17 runner request')
    args = ['--registry', str(JOB/'registry.toml'), '--suite-id', SUITE,
            '--bootstrap-source-suite-id', BOOTSTRAP,
            '--training-initialization', 'warm_start', '--strategies', policy,
            '--seed', str(seed), '--split-seed', str(SPLIT_SEED),
            '--initial-fraction', '.10', '--minimum-initial-per-class', '5',
            '--l0-selection', 'uniform_random', '--l0-count', str(L0_SIZE),
            '--acquisition-fraction', '.10', '--max-cycles', str(CYCLES),
            '--max-epochs', '50', '--patience', '10',
            '--batch-size', '512', '--validation-batch-size', '2048',
            '--learning-rate', '.0001', '--train-workers', '0',
            '--validation-workers', '0', '--inference-workers', '0',
            '--evaluation-workers', '0', '--assert-excluded-manifest', str(EXCLUDE)]
    if mode == 'bootstrap':
        return args + ['--bootstrap-only', '--run']
    if mode in ('check-bootstrap', 'cycle'):
        args.append('--require-existing-bootstrap')
    return args


def execute_runner(args, *, job_dir=None):
    root = JOB if job_dir is None else Path(job_dir)
    env = dict(os.environ,
               PYTHONPATH=str(root/'source/src') + os.pathsep + os.environ.get('PYTHONPATH',''))
    subprocess.run([sys.executable, str(root/'source/scripts/run_astrominn_closed_loop.py'),
                    *args], env=env, check=True)


def manifest_ids(path, key):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return {row[key] for row in csv.DictReader(f)}


def split_audit():
    # Call only after preflight has materialized split manifests.
    frozen = str(JOB/'source/src')
    if frozen not in sys.path:
        sys.path.insert(0, frozen)
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    ds = DatasetRegistry.from_toml(JOB/'registry.toml').resolve_profile('astrominn_ztf').dataset
    original = PERSISTENT/'runs'/TASK4/'shared/historical_replay_split'
    current = PERSISTENT/'runs'/SUITE/'shared/historical_replay_split'
    orig_l = manifest_ids(original/'initial_labeled_manifest.csv', ds.object_id_column)
    orig_u = set(CandidatePoolManifest.from_csv(original/'candidate_pool/candidate_pool.csv').object_ids)
    new_l = manifest_ids(current/'initial_labeled_manifest.csv', ds.object_id_column)
    new_u = set(CandidatePoolManifest.from_csv(current/'candidate_pool/candidate_pool.csv').object_ids)
    if len(orig_l) != L0_SIZE or len(new_l) != L0_SIZE or len(new_u) != U0_SIZE:
        raise ValueError('Task17 L0/U0 expected sizes violated')
    if orig_l & orig_u or new_l & new_u or orig_l | orig_u != new_l | new_u:
        raise ValueError('Task17 and Task4 do not partition same historical population')
    if orig_l == new_l:
        raise ValueError('Random L0 is identical to Task4 stratified L0')
    summary = load(current/'split_summary.json')
    if (summary['design'] != 'historical_train_object_uniform_random_replay'
            or summary['initial_object_count_requested'] != L0_SIZE
            or summary['split_seed'] != SPLIT_SEED):
        raise ValueError('Task17 split provenance mismatch')
    return {'new_L0': len(new_l), 'new_U0': len(new_u),
            'L0_overlap_with_Task4': len(orig_l & new_l),
            'new_L0_class_counts': summary['initial_object_label_counts'],
            'new_L0_sample_counts': summary['initial_sample_label_counts'],
            'absent_classes': [k for k,v in summary['initial_object_label_counts'].items() if not v],
            'validation_test_unchanged': True,
            'hidden_labels_used_in_selection': False}


def preflight():
    verify()
    for seed in SEEDS:
        execute_runner(runner_args(seed, mode='preflight'))
    report = split_audit()
    dump(JOB/'preflight_audit.json', report)
    print(json.dumps(report, indent=2))
    print('Task17 preflight passed; no training, GPU inference, or oracle queries')


def filesystem_probe():
    probe = DATA/'runs'/SUITE/'shared/runtime_probes'
    probe.mkdir(parents=True, exist_ok=True)
    path = probe/f'{os.getpid()}.probe'
    data = os.urandom(16*1024*1024)
    try:
        with path.open('wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        if path.read_bytes() != data:
            raise RuntimeError('Task17 readback mismatch')
        print('Task17 /work/hdd 16MiB write/readback passed')
    finally:
        path.unlink(missing_ok=True)


def bootstrap_job(seed, job_dir):
    verify(job_dir)
    if not (JOB/'preflight_audit.json').is_file():
        raise RuntimeError('Run Task17 preflight before bootstrap')
    execute_runner(runner_args(seed, mode='bootstrap'), job_dir=job_dir)
    execute_runner(runner_args(seed, mode='check-bootstrap'), job_dir=job_dir)
    print(f'Bootstrap {seed} completed and validated')


def validate_bootstrap(seed):
    if not bootstrap_checkpoint(seed).is_file():
        raise FileNotFoundError(f'Fresh Task17 bootstrap missing for seed {seed}')
    # Stronger than checkpoint existence: frozen trainer validates its artifacts.
    execute_runner(runner_args(seed, mode='check-bootstrap'))


def pilot(seed=135):
    if seed not in SEEDS:
        raise ValueError('Unsupported pilot seed')
    verify()
    validate_bootstrap(seed)
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (
        DEFAULT_LEGACY_CODE_DIR, predict_astrominn_vit_pool)
    from al_pipeline.core.scoring.entropy import EntropyScorer
    from al_pipeline.core.scoring.random import RandomScorer
    from al_pipeline.core.selection import TopKSelector
    from al_pipeline.core.pool import Pool
    import numpy as np
    output = JOB/'pilot'/f'seed_{seed}/result.json'
    if output.is_file():
        print(json.dumps(load(output), indent=2)); return
    registry = DatasetRegistry.from_toml(JOB/'registry.toml')
    profile = registry.resolve_profile('astrominn_ztf').profile
    candidate = PERSISTENT/'runs'/SUITE/'shared/historical_replay_split/candidate_pool'
    ids = tuple(CandidatePoolManifest.from_csv(candidate/'candidate_pool.csv').object_ids)
    cp = bootstrap_checkpoint(seed)
    start = time.perf_counter()
    inference = predict_astrominn_vit_pool(
        pool_path=candidate, object_ids=ids, device='cuda', batch_size=64,
        num_workers=0, legacy_code_dir=DEFAULT_LEGACY_CODE_DIR,
        artifact_dir=Path(profile.options['checkpoint']).parent,
        vit_checkpoint=cp)
    seconds = time.perf_counter() - start
    if tuple(inference.object_ids) != ids:
        raise ValueError('Object-level inference did not match U0')
    pool = Pool.from_ids(ids)
    selected = {}
    for policy in POLICIES:
        if policy == 'entropy':
            scores = EntropyScorer().score(inference.predictions)
        else:
            scores = RandomScorer(np.random.default_rng(np.random.SeedSequence([seed, 1]))).score(len(ids))
        batch = TopKSelector().select(ids=ids, scores=scores, k=BATCH_SIZE, pool=pool)
        if len(batch) != BATCH_SIZE or len(set(batch.ids)) != BATCH_SIZE:
            raise ValueError('Pilot acquisition mismatch')
        selected[policy] = batch.ids
    dump(output, {'seed': seed, 'checkpoint_sha256': sha(cp), 'candidate_count': len(ids),
                  'acquisition_budget': BATCH_SIZE, 'inference_seconds': seconds,
                  'selected_ids': selected, 'unique': True,
                  'validation_labels_accessed': False, 'test_labels_accessed': False})
    print(json.dumps({'seed': seed, 'inference_seconds': seconds,
                      'candidate_count': len(ids), 'acquisition_budget': BATCH_SIZE,
                      'policies': list(POLICIES)}, indent=2))


def pilot_status():
    path = JOB/'pilot/seed_135/result.json'
    print(json.dumps({'found': path.is_file(), 'path': str(path)}, indent=2))


def valid_walltime(walltime):
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d', walltime) or walltime == '00:00:00':
        raise ValueError('Invalid Slurm walltime')


def submit_bootstraps(walltime, seed_filter=None):
    verify(); valid_walltime(walltime)
    if not (JOB/'preflight_audit.json').is_file():
        raise RuntimeError('Task17 preflight required before bootstrap jobs')
    if seed_filter is not None and seed_filter not in SEEDS:
        raise ValueError('Invalid seed')
    ledger = JOB/'bootstrap_jobs.json'
    record = load(ledger) if ledger.is_file() else {'jobs': []}
    known = {x['seed'] for x in record['jobs']}
    for seed in SEEDS:
        if seed_filter is not None and seed != seed_filter or seed in known:
            continue
        cmd = ['sbatch','--parsable','--hold','--time',walltime,
               '--job-name',f'T17_L0_{seed}',
               '--output',str(JOB/'logs/%x_%j.out'),
               '--error',str(JOB/'logs/%x_%j.err'),
               str(JOB/'source/scripts/research/historical_replay/task17/task17_worker.sbatch'),
               str(JOB),'bootstrap',str(seed)]
        jid = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        if not jid.isdigit():
            raise RuntimeError('Unexpected sbatch response ' + jid)
        record['jobs'].append({'seed':seed,'job_id':jid,'walltime':walltime,'released':False})
        dump(ledger,record)
        print(f'Submitted HELD bootstrap seed {seed}: {jid}')


def release_bootstraps(seed_filter=None):
    record = load(JOB/'bootstrap_jobs.json')
    if seed_filter is not None and seed_filter not in SEEDS:
        raise ValueError('Invalid seed')
    for item in record['jobs']:
        if (seed_filter is None or item['seed']==seed_filter) and not item['released']:
            subprocess.run(['scontrol','release',item['job_id']],check=True)
            item['released']=True
            dump(JOB/'bootstrap_jobs.json',record)
            print(f"Released bootstrap {item['seed']}: {item['job_id']}")


def submit_cycles(walltime, policy_filter=None, seed_filter=None):
    verify(); valid_walltime(walltime)
    if policy_filter is not None and policy_filter not in POLICIES:
        raise ValueError('Invalid policy')
    if seed_filter is not None and seed_filter not in SEEDS:
        raise ValueError('Invalid seed')
    pilot_record = JOB/'pilot/seed_135/result.json'
    if not pilot_record.is_file():
        raise RuntimeError('Task17 real-model acquisition pilot required before AL submission')
    pilot_data = load(pilot_record)
    if pilot_data.get('unique') is not True or set(pilot_data['selected_ids']) != set(POLICIES):
        raise ValueError('Invalid Task17 acquisition pilot')
    ledger = JOB/'submitted.json'
    record = load(ledger) if ledger.is_file() else {'jobs': []}
    known = {(x['policy'],x['seed']) for x in record['jobs']}
    for seed in SEEDS:
        if seed_filter is not None and seed != seed_filter:
            continue
        # Do not submit a trajectory whose new L0 checkpoint cannot be verified.
        pending = [p for p in POLICIES if (policy_filter is None or p==policy_filter)
                   and (p,seed) not in known]
        if not pending:
            continue
        validate_bootstrap(seed)
        for policy in pending:
            generated, entries = [], []
            prior = None
            try:
                for cycle in range(1,CYCLES+1):
                    cmd = ['sbatch','--parsable','--hold','--time',walltime,
                           '--job-name',f'T17_{policy}_{seed}_{cycle}',
                           '--output',str(JOB/'logs/%x_%j.out'),
                           '--error',str(JOB/'logs/%x_%j.err')]
                    if prior:
                        cmd += ['--dependency',f'afterok:{prior}']
                    cmd += [str(JOB/'source/scripts/research/historical_replay/task17/task17_worker.sbatch'),
                            str(JOB),'cycle',policy,str(seed),str(cycle)]
                    jid = subprocess.check_output(cmd,text=True).strip().split(';')[0]
                    if not jid.isdigit():
                        raise RuntimeError('Unexpected sbatch response ' + jid)
                    generated.append(jid)
                    entries.append({'policy':policy,'seed':seed,'cycle':cycle,
                                    'job_id':jid,'dependency':f'afterok:{prior}' if prior else None,
                                    'released':False,'walltime':walltime})
                    prior=jid
            except Exception:
                if generated:
                    subprocess.run(['scancel',*generated],check=False)
                raise
            record['jobs'].extend(entries)
            dump(ledger,record)
            print(f'Submitted HELD AL chain {policy}/{seed}: {len(entries)} jobs')
    print(f'Task17: {len(record["jobs"])} / {2*3*CYCLES} AL jobs in ledger; no automatic releases')


def release_cycles(policy=None, seed=None, cycle=None, all_jobs=False):
    record=load(JOB/'submitted.json')
    if all_jobs:
        if any(x is not None for x in (policy,seed,cycle)):
            raise ValueError('Cannot combine --all with policy/seed/cycle')
        targets=[x for x in reversed(record['jobs']) if not x['released']]
    else:
        if policy not in POLICIES or seed not in SEEDS or cycle not in range(1,CYCLES+1):
            raise ValueError('Specify policy, seed, cycle or --all')
        targets=[x for x in record['jobs'] if (x['policy'],x['seed'],x['cycle'])==(policy,seed,cycle)]
        if len(targets)!=1:
            raise ValueError('No matching Task17 job')
    for x in targets:
        if x['released']:
            continue
        subprocess.run(['scontrol','release',x['job_id']],check=True)
        x['released']=True
        dump(JOB/'submitted.json',record)
        print(f"Released {x['job_id']} {x['policy']} seed {x['seed']} cycle {x['cycle']}")


def cycle_job(policy,seed,cycle,job_dir):
    if policy not in POLICIES or seed not in SEEDS or cycle not in range(1,CYCLES+1):
        raise ValueError('Invalid Task17 job')
    verify(job_dir)
    # The runner validates completed L0 bootstrap before doing any AL work.
    execute_runner(runner_args(seed, policy, mode='cycle')+
                   ['--device','cuda','--cycle',str(cycle),'--run'], job_dir=job_dir)


def status():
    cases=[]
    for policy in POLICIES:
        for seed in SEEDS:
            root=PERSISTENT/'runs'/SUITE/policy/f'seed_{seed}'
            p=root/'run_state.json'
            n=int(load(p).get('last_completed_cycle',0)) if p.exists() else 0
            cases.append({'policy':policy,'seed':seed,'completed_cycles':n,
                          'bootstrap_exists':bootstrap_checkpoint(seed).is_file(),
                          'complete':n==CYCLES})
    print(json.dumps({'cases':cases,'completed_series':sum(x['complete'] for x in cases),
                      'total_series':len(cases)},indent=2))


def collect(destination):
    """Export compact acquisition/validation results, never unrevealed labels."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Task17 export already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    from zipfile import ZipFile, ZIP_DEFLATED

    paths = [
        JOB/'scientific_plan.json', JOB/'preflight_audit.json',
        JOB/'submitted.json', JOB/'bootstrap_jobs.json',
        JOB/'source_parent.json', JOB/'frozen_manifest.json',
        JOB/'pilot/seed_135/result.json',
    ]
    split = PERSISTENT/'runs'/SUITE/'shared/historical_replay_split'
    paths += [split/'split_summary.json', split/'initial_labeled_manifest.csv']
    for policy in POLICIES:
        for seed in SEEDS:
            rel = Path(SUITE)/policy/f'seed_{seed}'
            run = PERSISTENT/'runs'/rel
            art = DATA/'runs'/rel
            paths += [run/name for name in ('run_state.json', 'evaluation_history.json',
                                            'experiment.json')]
            for cycle in range(CYCLES + 1):
                c = f'cycle_{cycle:03d}'
                if cycle:
                    paths += [run/c/name for name in ('selection.json',
                                                       'revealed_labels.json',
                                                       'cycle_summary.json')]
                paths += [art/c/'evaluation'/name for name in (
                    'evaluation_summary.json', 'validation_per_class_metrics.json',
                    'validation_object_confusion.csv', 'validation_sample_confusion.csv',
                )]

    archive = {}
    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue
        if path.is_relative_to(JOB):
            name = 'control/' + str(path.relative_to(JOB))
        elif path.is_relative_to(PERSISTENT):
            name = 'persistent/' + str(path.relative_to(PERSISTENT))
        elif path.is_relative_to(DATA):
            name = 'artifacts/' + str(path.relative_to(DATA))
        else:
            raise ValueError(f'Unexpected Task17 export file: {path}')
        archive[name] = path

    with ZipFile(destination, 'x', ZIP_DEFLATED) as out:
        for name, path in sorted(archive.items()):
            out.write(path, arcname=name)
    print(json.dumps({'output': str(destination), 'files': len(archive),
                      'sha256': sha(destination)}, indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','prepare','verify','preflight','filesystem-probe',
        'submit-bootstrap','release-bootstrap','job-bootstrap','pilot','pilot-status',
        'submit-cycles','release-cycles','job-cycle','status','collect'])
    p.add_argument('positional',nargs='*')
    p.add_argument('--overlay',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--job-dir',type=Path,default=JOB)
    p.add_argument('--walltime',default='00:35:00')
    p.add_argument('--seed',type=int)
    p.add_argument('--policy',choices=POLICIES)
    p.add_argument('--cycle',type=int)
    p.add_argument('--all',action='store_true')
    p.add_argument('--destination',type=Path)
    a=p.parse_args()
    if a.command=='plan': print(json.dumps(plan(),indent=2))
    elif a.command=='prepare': prepare(a.overlay)
    elif a.command=='verify': verify(a.job_dir)
    elif a.command=='preflight': preflight()
    elif a.command=='filesystem-probe': filesystem_probe()
    elif a.command=='submit-bootstrap': submit_bootstraps(a.walltime,a.seed)
    elif a.command=='release-bootstrap': release_bootstraps(a.seed)
    elif a.command=='job-bootstrap':
        if len(a.positional)!=1: raise ValueError('job-bootstrap needs seed')
        bootstrap_job(int(a.positional[0]),a.job_dir)
    elif a.command=='pilot': pilot(135 if a.seed is None else a.seed)
    elif a.command=='pilot-status': pilot_status()
    elif a.command=='submit-cycles': submit_cycles(a.walltime,a.policy,a.seed)
    elif a.command=='release-cycles': release_cycles(a.policy,a.seed,a.cycle,a.all)
    elif a.command=='job-cycle':
        if len(a.positional)!=3: raise ValueError('job-cycle needs policy seed cycle')
        cycle_job(a.positional[0],int(a.positional[1]),int(a.positional[2]),a.job_dir)
    elif a.command=='status': status()
    elif a.command=='collect':
        if a.destination is None: raise ValueError('--destination required')
        collect(a.destination)

if __name__=='__main__': main()
