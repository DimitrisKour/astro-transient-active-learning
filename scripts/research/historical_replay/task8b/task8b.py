#!/usr/bin/env python3
"""Task8B first-rejection recovery. prepare/verify/plan/status do not train or submit.

To run training, submit the explicit Slurm array or invoke `run --index N` on GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from al_pipeline.core.analysis.seed_recovery import (RULE, SOURCE_SUITE, RETRY_SEEDS,
    read_first_rejections, job_grid, sha256)
from al_pipeline.astronomy.analysis.seed_recovery import verify_source_case, load_json

PERSISTENT = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
SUITE = 'astrominn_task8b_first_rejection_recovery_v1'
JOB_DIR = DATA / 'recovery' / SUITE
REFERENCE = PERSISTENT / 'closed_loop' / SOURCE_SUITE
REPORT = DATA / 'analysis' / 'task8a_retrospective_acceptance_v1' / 'first_rejections.csv'
OVERLAY_FILES = (
    'src/al_pipeline/core/analysis/__init__.py',
    'src/al_pipeline/core/analysis/seed_recovery.py',
    'src/al_pipeline/astronomy/analysis/__init__.py',
    'src/al_pipeline/astronomy/analysis/seed_recovery.py',
    'scripts/research/historical_replay/task8b/task8b.py',
    'scripts/research/historical_replay/task8b/task8b_report.py',
    'scripts/research/historical_replay/task8b/task8b_worker.sbatch',
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def verify_task4_snapshot(base: Path) -> str:
    snapshot = base / 'source_snapshot.json'
    if not snapshot.is_file():
        raise FileNotFoundError(f'Frozen Task4 source snapshot not found: {snapshot}')
    records = load_json(snapshot)
    for relpath, digest in records.items():
        path = base / 'source' / relpath
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Task4 source snapshot inconsistent: {path}')
    if not (base / 'registry.toml').is_file():
        raise FileNotFoundError(base / 'registry.toml')
    return sha256(snapshot)


def expected_content(repo: Path, frozen_task4: Path) -> dict[str, bytes]:
    """Preserve exact Task4 trainer; add only new Task8B code on top."""
    old_src = frozen_task4 / 'source' / 'src'
    if not old_src.is_dir():
        raise FileNotFoundError(old_src)
    content = {str(p.relative_to(frozen_task4 / 'source')): p.read_bytes()
               for p in old_src.rglob('*.py')}
    for rel in OVERLAY_FILES:
        p = repo / rel
        if not p.is_file():
            raise FileNotFoundError(p)
        # Never silently replace a pre-existing experimental module.
        if rel in content and content[rel] != p.read_bytes():
            raise ValueError(f'Overlay collides with existing frozen Task4 code: {rel}')
        content[rel] = p.read_bytes()
    return content


def create_plan(*, repo: Path, job_dir: Path, task4: Path, report: Path,
                persistent: Path, data: Path) -> dict:
    reference_digest = verify_task4_snapshot(task4)
    report_sha = sha256(report)
    cases = read_first_rejections(report)
    if len(cases) != 6:
        raise ValueError('Expected six independent recovery cases')
    case_inputs = {}
    for case in cases:
        evidence = verify_source_case(case, persistent, data, task4 / 'registry.toml')
        key = f'{case.policy}_s{case.seed}'
        case_inputs[key] = evidence.to_dict()
    jobs = [{**job.as_dict(), 'output_dir': str(job_dir / 'fits' / job.name)}
            for job in job_grid(cases)]
    return {'schema': 'task8b_first_rejection_v1', 'suite': SUITE,
            'reference_suite': SOURCE_SUITE, 'rule': RULE,
            'source_audit': str(report), 'source_audit_sha256': report_sha,
            'reference_snapshot': str(task4 / 'source_snapshot.json'),
            'reference_snapshot_sha256': reference_digest,
            'reference_registry_sha256': sha256(task4 / 'registry.toml'),
            'persistent': str(persistent), 'data': str(data),
            'model_training': {'checkpoint_selection': 'weighted_validation_loss_best',
                'class_weight_exponent': .85, 'new_object_weighting': 'per_sample',
                'warm_start': 'original_previous_checkpoint',
                'train_manifest': 'original_rejected_cycle_cumulative_L_t',
                'validation_only': True, 'optimizer_scheduler': 'fresh_per_fit',
                'expert_towers': 'frozen', 'no_acquisition': True},
            'retry_seeds': list(RETRY_SEEDS), 'cases': case_inputs,
            'jobs': jobs, 'training_scheduled': False}


def freeze(*, repo: Path, job_dir: Path, task4: Path, report: Path,
           persistent: Path, data: Path) -> dict:
    repo = repo.resolve()
    task4 = task4.resolve()
    if job_dir.resolve().is_relative_to((persistent / 'runs').resolve()) or \
       job_dir.resolve().is_relative_to((data / 'runs').resolve()):
        raise ValueError('Retry outputs must not be located within original runs')
    plan = create_plan(repo=repo, job_dir=job_dir, task4=task4, report=report,
                       persistent=persistent, data=data)
    content = expected_content(repo, task4)
    signatures = {name: hashlib.sha256(raw).hexdigest() for name, raw in content.items()}
    planned = job_dir / 'plan.json'
    if planned.exists():
        if load_json(planned) != plan or load_json(job_dir / 'source_snapshot.json') != signatures:
            raise ValueError('Frozen Task8B plan/source changed; refuse overwrite')
        verify(job_dir, deep=True)
        return plan
    if (job_dir / 'source').exists() or (job_dir / 'source_snapshot.json').exists():
        raise RuntimeError('Partial snapshot exists; inspect manually')
    job_dir.mkdir(parents=True, exist_ok=True)
    for name, raw in content.items():
        p = job_dir / 'source' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    (job_dir / 'logs').mkdir(exist_ok=True)
    shutil.copy2(task4 / 'registry.toml', job_dir / 'registry.toml')
    atomic_json(job_dir / 'source_snapshot.json', signatures)
    atomic_json(planned, plan)
    verify(job_dir, deep=True)
    return plan


def verify(job_dir: Path, *, deep: bool = False) -> dict:
    plan = load_json(job_dir / 'plan.json')
    if (plan['schema'] != 'task8b_first_rejection_v1' or
            plan['suite'] != SUITE or plan['rule'] != RULE or len(plan['jobs']) != 30):
        raise ValueError('Invalid Task8B frozen plan')
    if sha256(job_dir / 'registry.toml') != plan['reference_registry_sha256']:
        raise ValueError('Task8B registry mutated')
    for rel, digest in load_json(job_dir / 'source_snapshot.json').items():
        path = job_dir / 'source' / rel
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Frozen Task8B source differs: {path}')
    if sha256(Path(plan['source_audit'])) != plan['source_audit_sha256']:
        raise ValueError('Task8A source report changed since Task8B freeze')
    if deep:
        task4 = Path(plan['reference_snapshot']).parent
        if (verify_task4_snapshot(task4) != plan['reference_snapshot_sha256']
                or sha256(task4 / 'registry.toml') != plan['reference_registry_sha256']):
            raise ValueError('Frozen Task4 reference changed')
        cases = read_first_rejections(Path(plan['source_audit']))
        for case in cases:
            key = f'{case.policy}_s{case.seed}'
            expected = plan['cases'][key]
            actual = verify_source_case(case, Path(plan['persistent']), Path(plan['data']),
                                        job_dir / 'registry.toml').to_dict()
            if actual != expected:
                raise ValueError(f'Source case inputs changed: {key}')
    return plan


def train_one(job_dir: Path, index: int) -> None:
    """The ONLY command here that performs training/inference; explicit GPU invocation."""
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Task8B training requires an explicit SLURM GPU job')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; refuse CPU fallback')
    plan = verify(job_dir, deep=True)
    if not 0 <= index < len(plan['jobs']):
        raise ValueError('Invalid array task index')
    job = plan['jobs'][index]
    key = f"{job['policy']}_s{job['seed']}"
    inp = plan['cases'][key]
    output = Path(job['output_dir'])
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainingConfig, AstroMiNNViTTrainer
    from al_pipeline.astronomy.astrominn.evaluation import AstroMiNNViTEvaluator
    from al_pipeline.core.training.base import TrainingRequest
    original = DatasetRegistry.from_toml(job_dir / 'registry.toml').resolve_profile('astrominn_ztf').dataset
    dataset = replace(original, train_manifest=Path(inp['l0_manifest']))
    policy = AstroMiNNViTTrainingConfig(**{**inp['model_policy'], 'loader_seed': job['retry_seed']})
    if (policy.class_weight_exponent != .85 or policy.new_object_weighting != 'per_sample'
            or policy.class_weight_source != 'training_manifest' or policy.overwrite):
        raise ValueError('Unexpected training policy')
    request = TrainingRequest(
        train_manifest=Path(inp['training_manifest']), validation_manifest=Path(inp['validation_manifest']),
        output_dir=output / 'training', run_id=f'{SUITE}_{job["name"]}',
        cycle=job['rejected_cycle'], strategy=job['policy'], seed=job['retry_seed'],
        initial_checkpoint=Path(inp['initial_checkpoint']))
    trainer = AstroMiNNViTTrainer(dataset=dataset, policy=policy, device='cuda',
                                  initialization='pretrained', artifact_dir=Path(inp['model_artifact_dir']))
    finished = trainer.completed_result(request)
    if finished is None:
        # Partial output is never deleted or implicitly restarted: investigate timeouts.
        if request.output_dir.exists() and any(request.output_dir.iterdir()):
            raise RuntimeError(f'Partial training output exists; refusing overwrite: {request.output_dir}')
        finished = trainer.train(request)
    evaluator = AstroMiNNViTEvaluator(
        dataset=dataset, manifests={'validation': Path(inp['validation_manifest'])},
        device='cuda', num_workers=0, artifact_dir=Path(inp['model_artifact_dir']))
    result = evaluator.evaluate(checkpoint_path=finished.checkpoint_path,
                                output_dir=output / 'evaluation')
    completion = {'job': job, 'checkpoint_sha256': sha256(finished.checkpoint_path),
                  'checkpoint_path': str(finished.checkpoint_path),
                  'training_summary_sha256': sha256(finished.summary_path),
                  'validation_evaluation_summary': str(result.summary_path),
                  'validation_only': True, 'original_training_manifest_sha256': inp['manifest_sha256'],
                  'initial_checkpoint_sha256': inp['initial_checkpoint_sha256'],
                  'training_elapsed_seconds': finished.elapsed_seconds}
    marker = output / 'complete.json'
    if marker.exists() and load_json(marker) != completion:
        raise ValueError('Completion marker does not match current artifacts')
    if not marker.exists():
        atomic_json(marker, completion)
    print(json.dumps({'completed': job['name'], 'seconds': finished.elapsed_seconds}, indent=2))


def status(job_dir: Path) -> dict:
    plan = verify(job_dir)
    complete, incomplete, invalid = [], [], []
    for j in plan['jobs']:
        path = Path(j['output_dir']) / 'complete.json'
        if not path.exists():
            incomplete.append(j['name'])
        else:
            try:
                marker = load_json(path)
                if marker['job'] != j or not Path(marker['checkpoint_path']).is_file():
                    raise ValueError('Completion mismatch')
                if sha256(Path(marker['checkpoint_path'])) != marker['checkpoint_sha256']:
                    raise ValueError('Checkpoint hash mismatch')
                complete.append(j['name'])
            except Exception as exc:
                invalid.append({'job': j['name'], 'error': str(exc)})
    return {'completed': len(complete), 'pending': len(incomplete), 'invalid': invalid,
            'completed_names': complete, 'pending_names': incomplete}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan', 'prepare', 'verify', 'submit', 'run', 'status'))
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument('--job-dir', type=Path, default=JOB_DIR)
    p.add_argument('--task4', type=Path, default=REFERENCE)
    p.add_argument('--report', type=Path, default=REPORT)
    p.add_argument('--persistent', type=Path, default=PERSISTENT)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--index', type=int)
    p.add_argument('--walltime', default='00:20:00')
    p.add_argument('--max-concurrent', type=int, default=3)
    a = p.parse_args()
    if a.command == 'plan':
        plan = create_plan(repo=a.repo, job_dir=a.job_dir, task4=a.task4,
                           report=a.report, persistent=a.persistent, data=a.data)
        print(json.dumps({'case_cycles': [(c['case']['policy'], c['case']['seed'], c['case']['rejected_cycle'])
                                           for c in plan['cases'].values()],
                          'total_new_fits': len(plan['jobs']), 'job_dir': str(a.job_dir),
                          'read_only': True, 'submission': 'none'}, indent=2))
    elif a.command == 'prepare':
        plan = freeze(repo=a.repo, job_dir=a.job_dir, task4=a.task4,
                      report=a.report, persistent=a.persistent, data=a.data)
        print(json.dumps({'prepared': str(a.job_dir), 'fits': len(plan['jobs']),
                          'submitted': False}, indent=2))
    elif a.command == 'verify':
        verify(a.job_dir, deep=True)
        print('Task8B frozen source, Task4 origin, and all six cases verified')
    elif a.command == 'run':
        if a.index is None:
            p.error('--index is required for run')
        train_one(a.job_dir, a.index)
    elif a.command == 'status':
        print(json.dumps(status(a.job_dir), indent=2))
    elif a.command == 'submit':
        plan = verify(a.job_dir, deep=True)
        if not 1 <= a.max_concurrent <= 6:
            p.error('--max-concurrent must be 1–6')
        if not __import__('re').fullmatch(r'\d{2}:\d{2}:\d{2}', a.walltime):
            p.error('Invalid walltime')
        worker = a.job_dir / 'source' / 'scripts' / 'adaptive' / 'task8b_worker.sbatch'
        cmd = ['sbatch', f'--time={a.walltime}',
               f'--array=0-{len(plan["jobs"])-1}%{a.max_concurrent}',
               f'--output={a.job_dir}/logs/T8B_%A_%a.out',
               f'--error={a.job_dir}/logs/T8B_%A_%a.err',
               str(worker), str(a.job_dir)]
        print('SUBMITTING (explicit request):', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
