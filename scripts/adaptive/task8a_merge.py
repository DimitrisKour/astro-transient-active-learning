#!/usr/bin/env python3
"""Combine three independent Task 8A suite audits without recomputation.

Checks all shards and refuses missing, mismatched, duplicated or incomplete
reports. No access to any training or evaluation source artifacts.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

SUITES = ('task4', 'task6b', 'task6c')
TABLES = ('adjacent_comparisons', 'per_class_changes', 'prediction_flips',
          'acquisition_batches', 'exploratory_rules', 'first_rejections',
          'trajectory_summary', 'rule_summary')
DEFAULT_OUTPUT = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task8a_retrospective_acceptance_v1')


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def atomic_text(path: Path, text: str) -> None:
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'Cannot merge empty table {path.name}')
    columns = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def merge(output: Path, *, suites: tuple[str, ...] = SUITES) -> dict:
    if tuple(suites) != SUITES:
        raise ValueError('Only the complete three-suite Task 8A report can be merged')
    summaries = []
    combined = {name: [] for name in TABLES}
    for suite in suites:
        shard = output / 'shards' / suite
        summary_path = shard / 'summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError(f'Incomplete shard {suite}: {summary_path}')
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if (summary.get('schema') != 'task8a_retrospective_acceptance_v1'
                or summary.get('suites') != [suite]
                or summary.get('test_accessed') is not False
                or summary.get('training_or_inference_performed') is not False):
            raise ValueError(f'Shard summary provenance mismatch for {suite}')
        if not isinstance(summary.get('bootstrap_repetitions'), int) or summary['bootstrap_repetitions'] < 20:
            raise ValueError(f'Invalid bootstrap repetitions for {suite}')
        if len(summary.get('completed_cycles', {})) != 6:
            raise ValueError(f'Expected six trajectories in {suite} shard')
        for name in TABLES:
            rows = read_rows(shard / f'{name}.csv')
            if any(r.get('suite_key') != suite for r in rows):
                raise ValueError(f'Mixed or missing suite provenance in {suite}/{name}.csv')
            combined[name].extend(rows)
        if len([r for r in combined['adjacent_comparisons'] if r['suite_key']==suite]) != summary['n_adjacent_comparisons']:
            raise ValueError(f'Comparison count mismatch for {suite}')
        if len([r for r in combined['prediction_flips'] if r['suite_key']==suite]) != summary['n_changed_object_records']:
            raise ValueError(f'Prediction-flip count mismatch for {suite}')
        if len([r for r in combined['first_rejections'] if r['suite_key']==suite]) != summary['n_first_rejection_cases']:
            raise ValueError(f'First-rejection count mismatch for {suite}')
        summaries.append(summary)
    if len({s['bootstrap_repetitions'] for s in summaries}) != 1:
        raise ValueError('Inconsistent bootstrap settings between shards')
    if len(combined['per_class_changes']) != 5 * len(combined['adjacent_comparisons']):
        raise ValueError('Expected five per-class rows per adjacent comparison')
    for name in TABLES:
        if not combined[name]:
            raise ValueError(f'Empty merged table: {name}')
    output.mkdir(parents=True, exist_ok=True)
    merged = {**summaries[0],
              'suites': list(suites), 'primary': 'task4',
              'completed_cycles': {k: v for s in summaries for k,v in s['completed_cycles'].items()},
              'n_adjacent_comparisons': sum(s['n_adjacent_comparisons'] for s in summaries),
              'n_changed_object_records': sum(s['n_changed_object_records'] for s in summaries),
              'n_first_rejection_cases': sum(s['n_first_rejection_cases'] for s in summaries),
              'missing_secondary': [x for s in summaries for x in s['missing_secondary']],
              'trajectory_summary': [x for s in summaries for x in s['trajectory_summary']],
              'rule_summary': [x for s in summaries for x in s['rule_summary']],
              'execution': 'three independent CPU suite shards, then read-only merge',
              'merged_shard_suites': list(suites)}
    for name, rows in combined.items():
        write_csv(output / f'{name}.csv', rows)
    atomic_text(output / 'summary.json', json.dumps(merged, indent=2, sort_keys=True) + '\n')
    atomic_text(output / 'README.md',
                '# Task 8A: merged read-only retrospective checkpoint audit\n\n'
                'Three independent suite shard reports were merged without retraining or '
                'inference. All object-level validation metrics, bootstrap uncertainty and '
                'illustrative first-rejection rules are unchanged from the single-job audit. '
                'Later saved checkpoints are not acceptance-controlled counterfactuals. '
                'Test predictions were not accessed. The validation set is reused for '
                'epoch selection, so intervals are descriptive.\n')
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = merge(args.output)
    print(json.dumps({'merged_suites': summary['suites'],
                      'comparisons': summary['n_adjacent_comparisons'],
                      'changed_object_records': summary['n_changed_object_records'],
                      'first_rejection_cases': summary['n_first_rejection_cases']}, indent=2))

if __name__ == '__main__':
    main()
