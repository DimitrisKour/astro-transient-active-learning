#!/usr/bin/env python3
"""Task10: 20 fixed-seed independent Task4 entropy seed357/cycle5 retries.

No acquisition; validation only. Explicit commands: plan, prepare, verify, status, submit, run.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

PERSISTENT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
SOURCE = DATA / 'recovery' / 'astrominn_task8b_first_rejection_recovery_v1'
SUITE = 'astrominn_task10_entropy_s357_c5_seed20_v1'
ROOT = DATA / 'recovery' / SUITE
CASE_KEY = 'entropy_s357'
SEEDS = tuple(range(2001, 2021))
SCHEMA = 'task10_entropy20_v1'
SCRIPTS = (
    'scripts/research/historical_replay/task10/task10_entropy20.py',
    'scripts/research/historical_replay/task10/task10_entropy20_report.py',
    'scripts/research/historical_replay/task10/task10_entropy20_worker.sbatch',
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def verified_evidence(task8b: Path):
    """Use the verified Task8B first-rejection manifest and Task4 provenance."""
    from al_pipeline.astronomy.analysis.seed_recovery import verify_source_case
    from al_pipeline.core.analysis.seed_recovery import RecoveryCase
    src_plan_path = task8b / 'plan.json'
    base = read(src_plan_path)
    if base.get('schema') != 'task8b_first_rejection_v1' or len(base.get('jobs', [])) != 30:
        raise ValueError('Unexpected Task8B source experiment contract')
    if sha256(Path(base['source_audit'])) != base['source_audit_sha256']:
        raise ValueError('Task8A source report has changed')
    from al_pipeline.core.analysis.seed_recovery import read_first_rejections
    original_case = [c for c in read_first_rejections(Path(base['source_audit'])) if c.policy == 'entropy' and c.seed == 357]
    if len(original_case) != 1 or original_case[0] != RecoveryCase('entropy', 357, 5, 4):
        raise ValueError('Unexpected entropy seed357 first rejection')
    evidence = verify_source_case(original_case[0], Path(base['persistent']), Path(base['data']), task8b / 'registry.toml').to_dict()
    if evidence != base['cases'][CASE_KEY]:
        raise ValueError('Task8B verified evidence no longer matches Task4 artifacts')
    if sha256(task8b / 'registry.toml') != base['reference_registry_sha256']:
        raise ValueError('Task8B registry modified')
    ss = read(task8b / 'source_snapshot.json')
    if not ss or not all((task8b / 'source' / p).is_file() and sha256(task8b / 'source' / p) == h for p, h in ss.items()):
        raise ValueError('Task8B frozen source changed')
    return base, evidence


def expected_files(repo: Path, source: Path):
    folder = source / 'source' / 'src'
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    content = {str(p.relative_to(source / 'source')): p.read_bytes() for p in folder.rglob('*.py')}
    # Reuse the validated Task8B report utility and its strict checkpoint/prediction loader.
    orig_report = source / 'source' / 'scripts/adaptive/task8b_report.py'
    if not orig_report.is_file():
        raise FileNotFoundError(orig_report)
    content['scripts/adaptive/task8b_report.py'] = orig_report.read_bytes()
    for rel in SCRIPTS:
        path = repo / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        if rel in content:
            raise ValueError('Task10 script collides with frozen training source')
        content[rel] = path.read_bytes()
    return content


def plan(*, repo: Path, source: Path, job: Path):
    base, inp = verified_evidence(source)
    files = expected_files(repo, source)
    source_digest = sha256(source / 'source_snapshot.json')
    jobs = [{'index': i, 'retry_seed': seed, 'output_dir': str(job / 'fits' / f'retry_{seed}')}
            for i, seed in enumerate(SEEDS)]
    assert len(jobs) == len(SEEDS) == 20
    return {
        'schema': SCHEMA, 'suite': SUITE, 'case_key': CASE_KEY,
        'source_task8b_plan_sha256': sha256(source / 'plan.json'),
        'source_task8b_source_snapshot_sha256': source_digest,
        'source_task8b': str(source),
        'source_task4_audit': base['source_audit'],
        'evidence': inp, 'retry_seeds': list(SEEDS), 'jobs': jobs,
        'training_contract': {
            'initialization': 'pretrained', 'warm_start': 'Task4 previous checkpoint, cycle4',
            'data': 'exact fixed Task4 entropy seed357 cumulative L5',
            'source_trainer': 'frozen Task4 training implementation from Task8B',
            'epochs_max': 50, 'patience': 10, 'class_weight_exponent': .85,
            'class_weight_source': 'training_manifest', 'new_object_weighting': 'per_sample',
            'randomness': 'distinct process and loader seeds 2001..2020',
            'selection': 'weighted validation loss-best checkpoint within each fit',
            'test_used': False, 'acquisition': False,
        },
        'source_file_hashes': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    }


def prepare(*, repo: Path, source: Path, job: Path):
    planned = plan(repo=repo, source=source, job=job)
    target = job / 'plan.json'
    if target.exists():
        verify(job, deep=True)
        if read(target) != planned:
            raise ValueError('Plan changed; refuse to alter existing frozen experiment')
        return planned
    if job.exists() and any(job.iterdir()):
        raise RuntimeError(f'Partial experiment directory; inspect before continuing: {job}')
    contents = expected_files(repo, source)
    job.mkdir(parents=True, exist_ok=True)
    for rel, body in contents.items():
        dest = job / 'source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    shutil.copy2(source / 'registry.toml', job / 'registry.toml')
    (job / 'logs').mkdir(exist_ok=True)
    write(target, planned)
    verify(job, deep=True)
    return planned


def verify(job: Path, deep=False):
    p = read(job / 'plan.json')
    if (p['schema'] != SCHEMA or p['suite'] != SUITE or p['case_key'] != CASE_KEY or
        p['retry_seeds'] != list(SEEDS) or len(p['jobs']) != 20 or
        [x['retry_seed'] for x in p['jobs']] != list(SEEDS) or
        [x['index'] for x in p['jobs']] != list(range(20))):
        raise ValueError('Invalid frozen Task10 plan')
    for rel, digest in p['source_file_hashes'].items():
        path = job / 'source' / rel
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Frozen source changed: {rel}')
    task8b = Path(p['source_task8b'])
    if sha256(job / 'registry.toml') != sha256(task8b / 'registry.toml'):
        raise ValueError('Experiment registry changed')
    if (sha256(task8b / 'plan.json') != p['source_task8b_plan_sha256'] or
        sha256(task8b / 'source_snapshot.json') != p['source_task8b_source_snapshot_sha256']):
        raise ValueError('Task8B frozen source record changed')
    if deep:
        _, evidence = verified_evidence(task8b)
        if evidence != p['evidence']:
            raise ValueError('Original cumulative data/checkpoints changed')
    return p


def run(job: Path, index: int):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Training requires an explicit Slurm GPU allocation')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    p = verify(job, deep=True)
    if not 0 <= index < 20:
        raise ValueError('Index out of range')
    j = p['jobs'][index]
    e = p['evidence']
    output = Path(j['output_dir'])
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainer, AstroMiNNViTTrainingConfig
    from al_pipeline.astronomy.astrominn.evaluation import AstroMiNNViTEvaluator
    from al_pipeline.core.training.base import TrainingRequest
    dataset_orig = DatasetRegistry.from_toml(job / 'registry.toml').resolve_profile('astrominn_ztf').dataset
    dataset = replace(dataset_orig, train_manifest=Path(e['l0_manifest']))
    policy = AstroMiNNViTTrainingConfig(**{**e['model_policy'], 'loader_seed': j['retry_seed']})
    if (policy.class_weight_source != 'training_manifest' or policy.class_weight_exponent != .85
            or policy.new_object_weighting != 'per_sample' or policy.max_epochs != 50
            or policy.patience != 10 or policy.overwrite):
        raise ValueError('Task4 training policy changed')
    request = TrainingRequest(
        train_manifest=Path(e['training_manifest']), validation_manifest=Path(e['validation_manifest']),
        output_dir=output / 'training', run_id=f'{SUITE}_seed_{j["retry_seed"]}',
        cycle=5, strategy='entropy', seed=j['retry_seed'], initial_checkpoint=Path(e['initial_checkpoint']))
    trainer = AstroMiNNViTTrainer(dataset=dataset, policy=policy, device='cuda',
                                  initialization='pretrained', artifact_dir=Path(e['model_artifact_dir']))
    done = trainer.completed_result(request)
    if done is None:
        if request.output_dir.exists() and any(request.output_dir.iterdir()):
            raise RuntimeError(f'Incomplete training directory, requires manual investigation: {request.output_dir}')
        done = trainer.train(request)
    evaluator = AstroMiNNViTEvaluator(dataset=dataset, manifests={'validation': Path(e['validation_manifest'])},
                                     device='cuda', num_workers=0, artifact_dir=Path(e['model_artifact_dir']))
    evaluation = evaluator.evaluate(checkpoint_path=done.checkpoint_path, output_dir=output / 'evaluation')
    marker = {'job': j, 'checkpoint': str(done.checkpoint_path),
              'checkpoint_sha256': sha256(done.checkpoint_path),
              'training_summary_sha256': sha256(done.summary_path),
              'evaluation_summary_sha256': sha256(evaluation.summary_path),
              'training_manifest_sha256': e['manifest_sha256'],
              'initial_checkpoint_sha256': e['initial_checkpoint_sha256'],
              'validation_only': True, 'training_elapsed_seconds': done.elapsed_seconds}
    path = output / 'complete.json'
    if path.exists():
        if read(path) != marker:
            raise ValueError(f'Completion record inconsistent for {j["retry_seed"]}')
    else:
        write(path, marker)
    print(json.dumps({'completed_seed': j['retry_seed'], 'elapsed_seconds': done.elapsed_seconds}))


def status(job: Path):
    p = verify(job)
    good, pending, invalid = [], [], []
    for j in p['jobs']:
        path = Path(j['output_dir']) / 'complete.json'
        if not path.is_file():
            pending.append(j['retry_seed']); continue
        try:
            d = read(path)
            checkpoint = Path(d['checkpoint'])
            if (d['job'] != j or not d['validation_only'] or
                    sha256(checkpoint) != d['checkpoint_sha256'] or
                    sha256(Path(j['output_dir'])/'training/training_summary.json') != d['training_summary_sha256'] or
                    sha256(Path(j['output_dir'])/'evaluation/evaluation_summary.json') != d['evaluation_summary_sha256']):
                raise ValueError('Incomplete or altered fit')
            good.append(j['retry_seed'])
        except Exception as ex:
            invalid.append({'seed': j['retry_seed'], 'error': str(ex)})
    return {'case': CASE_KEY, 'completed': len(good), 'planned': 20,
            'completed_seeds': good, 'pending_seeds': pending, 'invalid': invalid}


def submit(job: Path, walltime: str, max_concurrent: int):
    p = verify(job, deep=True)
    if not 1 <= max_concurrent <= 4:
        raise ValueError('max-concurrent must be between 1 and 4')
    if not re.fullmatch(r'\d{2}:\d{2}:\d{2}', walltime):
        raise ValueError('walltime must be HH:MM:SS')
    hh, mm, ss = [int(x) for x in walltime.split(':')]
    if hh * 3600 + mm * 60 + ss <= 0 or hh * 3600 + mm * 60 + ss > 20 * 60:
        raise ValueError('walltime must be 1 second to 20 minutes')
    for name in ('submitted.json', 'submission_pending.json'):
        if (job / name).exists():
            raise RuntimeError(f'Already submitted; refusing duplicate: {job / name}')
    state = status(job)
    if state['invalid'] or state['completed'] or len(state['pending_seeds']) != 20:
        raise RuntimeError('Refusing to submit nonempty or inconsistent experiment')
    worker = job/'source/scripts/research/historical_replay/task10/task10_entropy20_worker.sbatch'
    cmd = ['sbatch', '--parsable', f'--time={walltime}',
           f'--array=0-19%{max_concurrent}',
           f'--output={job}/logs/T10_%A_%a.out',
           f'--error={job}/logs/T10_%A_%a.err',
           str(worker), str(job)]
    # Durable marker *before* submission prevents accidental double submit after interruption.
    write(job / 'submission_pending.json', {'cmd': cmd})
    try:
        finished = subprocess.run(cmd, capture_output=True, text=True, check=True)
        jid = finished.stdout.strip().split(';')[0]
        if not jid.isdigit():
            raise RuntimeError(f'Unexpected sbatch response: {finished.stdout!r}')
        write(job / 'submitted.json', {'job_id': jid, 'array': '0-19', 'max_concurrent': max_concurrent,
                                       'command': cmd})
        (job / 'submission_pending.json').unlink()
        print(json.dumps({'submitted_array_job_id': jid, 'array': '0-19', 'max_concurrent': max_concurrent}, indent=2))
    except BaseException:
        raise RuntimeError('Submission may have occurred; inspect submission_pending.json and squeue before retrying')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'verify', 'status', 'submit', 'run'))
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--source-task8b', type=Path, default=SOURCE)
    parser.add_argument('--job-dir', type=Path, default=ROOT)
    parser.add_argument('--index', type=int)
    parser.add_argument('--walltime', default='00:20:00')
    parser.add_argument('--max-concurrent', type=int, default=2)
    a = parser.parse_args()
    if a.command == 'plan':
        print(json.dumps(plan(repo=a.repo, source=a.source_task8b, job=a.job_dir), indent=2))
    elif a.command == 'prepare':
        p = prepare(repo=a.repo, source=a.source_task8b, job=a.job_dir)
        print(json.dumps({'prepared': str(a.job_dir), 'jobs': len(p['jobs']), 'submitted': False}, indent=2))
    elif a.command == 'verify':
        verify(a.job_dir, deep=True)
        print('Task10 frozen code and exact Task8B/Task4 sources verified')
    elif a.command == 'status':
        print(json.dumps(status(a.job_dir), indent=2))
    elif a.command == 'submit':
        submit(a.job_dir, a.walltime, a.max_concurrent)
    elif a.command == 'run':
        if a.index is None:
            parser.error('--index required for run')
        run(a.job_dir, a.index)


if __name__ == '__main__':
    main()
