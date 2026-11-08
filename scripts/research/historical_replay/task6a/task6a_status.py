#!/usr/bin/env python3
"""Read-only submitted-job manifest audit for Task6A new-L0 v2.

This deliberately lives OUTSIDE task6a_l0_*.py: no frozen-science snapshot
changes are required to use it with already queued jobs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

DEFAULT_JOB_DIR = Path('/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_weight05_newl0_v1')
SEEDS = (135, 246, 357)
POLICIES = ('entropy', 'entropy_balanced_explore')


def expected_jobs() -> dict[str, tuple[int, str, int, str | None]]:
    expected = {}
    for seed in SEEDS:
        expected[f'l0:{seed}'] = (seed, 'bootstrap', 0, None)
        for policy in POLICIES:
            for cycle in range(1, 10):
                parent = f'l0:{seed}' if cycle == 1 else f'{seed}:{policy}:{cycle-1}'
                expected[f'{seed}:{policy}:{cycle}'] = (seed, policy, cycle, parent)
    assert len(expected) == 57
    return expected


def audit_manifest(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(doc, dict):
        raise ValueError('Submission manifest must be a dictionary, not a list')
    records = doc.get('jobs')
    if not isinstance(records, list):
        raise ValueError(f'{path} has no jobs array; inspect the manifest before making changes')
    expected = expected_jobs()
    if len(records) != 57:
        raise ValueError(f'Expected 57 submitted jobs; found {len(records)}')
    by_key = {}
    ids = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Every job must be a dictionary')
        key = row.get('key')
        if not isinstance(key, str) or key not in expected or key in by_key:
            raise ValueError(f'Unexpected or duplicate trajectory key {key!r}')
        seed, policy, cycle, parent = expected[key]
        if (row.get('seed'), row.get('policy'), row.get('cycle'), row.get('parent')) != (seed, policy, cycle, parent):
            raise ValueError(f'Trajectory mapping differs for {key}')
        job_id = str(row.get('job_id', ''))
        if not job_id.isdigit() or job_id in ids:
            raise ValueError(f'Invalid or duplicate SLURM job ID: {job_id!r}')
        ids.add(job_id)
        by_key[key] = row
    if set(by_key) != set(expected):
        raise ValueError('Manifest does not cover all six trajectories and three bootstrap fits')
    for key, row in by_key.items():
        parent = expected[key][3]
        expected_parent_id = None if parent is None else str(by_key[parent]['job_id'])
        recorded = row.get('dependency_job_id')
        if recorded != expected_parent_id:
            raise ValueError(f'Incorrect SLURM dependency in {key}: {recorded!r} vs {expected_parent_id!r}')
    return {
        'manifest': str(path),
        'manifest_status': doc.get('status', 'unknown'),
        'total': len(records),
        'l0': sum(row['cycle'] == 0 for row in records),
        'al_cycles': sum(row['cycle'] > 0 for row in records),
        'policy_counts': dict(sorted(Counter(row['policy'] for row in records).items())),
        'unique_slurm_job_ids': len(ids),
        'dependencies_verified': True,
        'caution': 'This checks the local manifest only; it does not confirm current SLURM execution state.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, default=DEFAULT_JOB_DIR)
    args = parser.parse_args()
    for name in ('submitted.json', 'submission_failed.json', 'submission_pending.json'):
        path = args.job_dir / name
        if path.exists():
            if name != 'submitted.json':
                raise SystemExit(f'Incomplete or failed submission marker: {path}; DO NOT resubmit blindly')
            print(json.dumps(audit_manifest(path), indent=2, sort_keys=True))
            return
    raise SystemExit(f'No submission manifest found in {args.job_dir}')


if __name__ == '__main__':
    main()
