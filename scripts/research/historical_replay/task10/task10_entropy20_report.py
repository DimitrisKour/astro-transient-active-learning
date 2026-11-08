#!/usr/bin/env python3
"""Task10 validation-only readout: 20 seed outcomes and fixed incremental ensembles.

No model fitting or acquisition. No test predictions read. Reports descriptive statistics,
not independent out-of-sample validation. All seed orders/combinations are fixed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

# Task8B's validation reader is shared with this historical experiment.
# Prefer the reorganized checkout, but support Task10 frozen source copies
# that retain the original Task8B report path.
_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_SHARED_READER_DIR = _SOURCE_ROOT / "scripts/research/historical_replay/task8b"
if not (_SHARED_READER_DIR / "task8b_report.py").is_file():
    _SHARED_READER_DIR = _SOURCE_ROOT / "scripts/adaptive"
if not (_SHARED_READER_DIR / "task8b_report.py").is_file():
    raise FileNotFoundError("Task10 requires the matching Task8B report dependency")
sys.path.insert(0, str(_SHARED_READER_DIR))

from al_pipeline.core.analysis.seed_recovery import sha256, macro_tolerance_accept
from task8b_report import load_prediction, aligned, scores, compare, CLASSES
from task10_entropy20 import ROOT, SCHEMA, CASE_KEY, SEEDS, verify, read, write


def csvwrite(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'No rows for {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for item in rows for key in item))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_predictions(job: Path, j: dict, evidence: dict, incumbent: tuple):
    folder = Path(j['output_dir'])
    completion = read(folder/'complete.json')
    if completion['job'] != j or not completion['validation_only']:
        raise ValueError('Completion provenance mismatch')
    check = folder/'training'/'best_vit_tower.pth'
    if (str(check) != completion['checkpoint'] or sha256(check) != completion['checkpoint_sha256']
            or completion['training_manifest_sha256'] != evidence['manifest_sha256']
            or completion['initial_checkpoint_sha256'] != evidence['initial_checkpoint_sha256']
            or sha256(folder/'training/training_summary.json') != completion['training_summary_sha256']
            or sha256(folder/'evaluation/evaluation_summary.json') != completion['evaluation_summary_sha256']):
        raise ValueError('Corrupt saved training/evaluation provenance')
    t = read(folder/'training'/'training_summary.json')
    if (t['train_manifest_sha256'] != evidence['manifest_sha256']
            or t['validation_manifest_sha256'] != evidence['validation_sha256']
            or t['initial_checkpoint'] != evidence['initial_checkpoint']
            or int(t['seed']) != j['retry_seed']
            or t['policy']['loader_seed'] != j['retry_seed']
            or float(t['policy']['class_weight_exponent']) != .85
            or t['policy']['class_weight_source'] != 'training_manifest'
            or t['policy']['new_object_weighting'] != 'per_sample'):
        raise ValueError('Saved fit differs from Task4 independent-seed contract')
    result = load_prediction(folder/'evaluation/validation_predictions.npz', check)
    aligned(incumbent, result)
    return result


def make_report(job: Path, output: Path, *, bootstrap: int = 250):
    if bootstrap < 1:
        raise ValueError('At least one bootstrap sample required')
    plan = verify(job, deep=True)
    if plan['schema'] != SCHEMA or plan['case_key'] != CASE_KEY:
        raise ValueError('Unexpected study')
    evidence = plan['evidence']
    case = evidence['case']
    incumbent = load_prediction(Path(evidence['incumbent_evaluation']).parent/'validation_predictions.npz',
                                Path(evidence['initial_checkpoint']))
    original = load_prediction(Path(evidence['source_evaluation']).parent/'validation_predictions.npz',
                               Path(evidence['candidate_checkpoint']))
    aligned(incumbent, original)
    models = []
    measurements = []
    classes = []
    absolute = []
    baseline, pc, _ = compare(incumbent, original, case=case, label='original_rejected',
                              kind='original', ref_label='incumbent', bootstrap=bootstrap)
    measurements.append(baseline);classes.extend(pc)
    for label, pred in [('incumbent', incumbent), ('original_rejected', original)]:
        met = scores(pred[1], pred[2])
        absolute.append({'model':label, 'macro_f1':met['macro_f1'], 'balanced_accuracy':met['balanced_accuracy'],
                         'accuracy':met['accuracy'], 'nll':met['nll'], 'brier':met['brier']})
    if macro_tolerance_accept(baseline['delta_macro_f1']):
        raise ValueError('Source candidate was not rejected by the operational gate')
    for j in plan['jobs']:
        pred = read_predictions(job, j, evidence, incumbent)
        models.append((j['retry_seed'], pred))
        met = scores(pred[1], pred[2])
        absolute.append({'model':f"retry_{j['retry_seed']}", 'macro_f1':met['macro_f1'],
                         'balanced_accuracy':met['balanced_accuracy'], 'accuracy':met['accuracy'],
                         'nll':met['nll'], 'brier':met['brier']})
        for ref, baseline_name in ((incumbent, 'incumbent'), (original, 'original_rejected')):
            row, cl, _ = compare(ref, pred, case=case, label=f"retry_{j['retry_seed']}",
                                 kind='single', ref_label=baseline_name, bootstrap=bootstrap)
            measurements.append(row);classes.extend(cl)
    results = [r for r in measurements if r['kind']=='single' and r['reference_model']=='incumbent']
    if len(models) != 20 or len(results) != 20:
        raise ValueError('Incomplete 20 independent models')
    passing = [bool(r['passes_provisional_1pp']) for r in results]
    recovery = []
    for k in (1, 2, 3, 4, 5, 8, 10, 12, 16, 20):
        group = results[:k]
        recovery.append({
            'seeds_examined': k, 'seeds_passing': sum(passing[:k]),
            'any_passing': any(passing[:k]),
            'first_passing_seed': next((models[i][0] for i in range(k) if passing[i]), None),
            'best_delta_macro_f1': max(r['delta_macro_f1'] for r in group),
            'mean_delta_macro_f1': float(np.mean([r['delta_macro_f1'] for r in group])),
            'note': 'Fixed seed order; best-of-k is exploratory and upward biased on validation',
        })
    ensemble_sizes = (2, 4, 8, 12, 16, 20)
    ens_defs = []
    for k in ensemble_sizes:
        p = np.mean(np.stack([pred[2] for _, pred in models[:k]]), axis=0)
        means = [
            (f'E{k}', p, [int(seed) for seed, _ in models[:k]]),
            (f'I_E{k}', (k*p + incumbent[2])/(k+1), ['incumbent']+[int(seed) for seed, _ in models[:k]]),
        ]
        for name, probabilities, members in means:
            ens_defs.append({'ensemble': name, 'n_members': len(members),
                             'seeds_and_incumbent': json.dumps(members)})
            combined = (incumbent[0], incumbent[1], probabilities)
            met = scores(combined[1], combined[2])
            absolute.append({'model':name, 'macro_f1':met['macro_f1'],
                             'balanced_accuracy':met['balanced_accuracy'], 'accuracy':met['accuracy'],
                             'nll':met['nll'], 'brier':met['brier']})
            for reference, refname in ((incumbent, 'incumbent'), (original, 'original_rejected')):
                row, cl, _ = compare(reference, combined, case=case, label=name,
                                     kind='ensemble', ref_label=refname, bootstrap=bootstrap)
                measurements.append(row);classes.extend(cl)
    incumbent_metrics = scores(incumbent[1], incumbent[2])
    first = next((models[i][0] for i in range(20) if passing[i]), None)
    elapsed = sum(read(Path(j['output_dir'])/'complete.json')['training_elapsed_seconds'] for j in plan['jobs'])
    summary = {
        'suite': plan['suite'], 'case': case,
        'n_new_training_fits': 20, 'seeds': list(SEEDS),
        'n_passing_1pp': sum(passing), 'passing_fraction': sum(passing)/20,
        'first_passing_seed': first,
        'original_rejected_macro_f1_delta': baseline['delta_macro_f1'],
        'mean_single_delta_macro_f1': float(np.mean([r['delta_macro_f1'] for r in results])),
        'mean_single_delta_balanced_accuracy': float(np.mean([r['delta_balanced_accuracy'] for r in results])),
        'individual_macro_f1_std': float(np.std([r['macro_f1'] for r in results], ddof=1)),
        'reference_incumbent_macro_f1': incumbent_metrics['macro_f1'],
        'total_training_seconds': elapsed, 'new_training_hours': elapsed/3600,
        'ensemble_sizes': list(ensemble_sizes), 'bootstrap_draws': bootstrap,
        'validation_only': True, 'test_used': False, 'no_acquisition': True,
        'important': 'Repeated validation measurements; success rate from one fixed training setup. '
                     'Best-of-k estimates are optimistic; selection policy not deployed.',
    }
    output.mkdir(parents=True, exist_ok=True)
    csvwrite(output/'individual_and_ensemble_comparisons.csv', measurements)
    for row in classes:
        n = int(row['n_validation'])
        a = int(round(n * float(row['recall_reference'])))
        b = int(round(n * float(row['recall_model'])))
        row.update({'correct_reference': a, 'correct_model': b, 'net_correct_delta': b-a})
    csvwrite(output/'class_recall.csv', classes)
    csvwrite(output/'models_absolute_metrics.csv', absolute)
    csvwrite(output/'recovery_by_budget.csv', recovery)
    csvwrite(output/'ensemble_definitions.csv', ens_defs)
    write(output/'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-dir', type=Path, default=ROOT)
    p.add_argument('--out', type=Path)
    p.add_argument('--bootstrap', type=int, default=250)
    a = p.parse_args()
    print(json.dumps(make_report(a.job_dir, a.out or a.job_dir/'report', bootstrap=a.bootstrap), indent=2))


if __name__ == '__main__':
    main()
