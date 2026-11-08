#!/usr/bin/env python3
"""Task8B validation-only analysis. No GPU training, acquisition, or test access."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from al_pipeline.core.analysis.seed_recovery import macro_tolerance_accept, sha256

CLASSES = ('AGN', 'TDE', 'SN I', 'SN II', 'CV')


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def dump_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'No results for {path.name}')
    cols = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=cols)
        writer.writeheader(); writer.writerows(rows)


def probabilities(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def load_prediction(path: Path, expected_checkpoint: Path, *, validation_only=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evaluation = load_json(path.parent / 'evaluation_summary.json')
    if Path(evaluation['checkpoint_path']) != expected_checkpoint:
        raise ValueError(f'Evaluation checkpoint mismatch: {path}')
    if tuple(evaluation['model_class_names']) != CLASSES:
        raise ValueError('Prediction class order changed')
    if sha256(expected_checkpoint) != evaluation['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA differs from saved evaluation')
    if validation_only and not any(x['split'] == 'validation' for x in evaluation['splits']):
        raise ValueError('No validation split')
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive['object_ids'], dtype=str)
        y = np.asarray(archive['object_targets'], dtype=np.int64)
        v = np.asarray(archive['object_values'], dtype=np.float64)
    if (len(ids) != len(set(ids.tolist())) or len(ids) != len(y)
            or v.shape != (len(ids), 5) or not np.isfinite(v).all()
            or np.any((y < 0) | (y > 4))):
        raise ValueError('Invalid prediction NPZ')
    order = np.argsort(ids)
    return ids[order], y[order], probabilities(v[order])


def aligned(ref: tuple, other: tuple) -> None:
    if not np.array_equal(ref[0], other[0]) or not np.array_equal(ref[1], other[1]):
        raise ValueError('Object IDs/targets are not aligned')


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(axis=1)
    eps = 1e-300
    tp = np.asarray([np.sum((y == k) & (pred == k)) for k in range(5)], dtype=float)
    support = np.asarray([np.sum(y == k) for k in range(5)], dtype=float)
    if any(support == 0):
        raise ValueError('A validation class is missing')
    guessed = np.asarray([np.sum(pred == k) for k in range(5)], dtype=float)
    recall = tp / support
    f1 = np.divide(2 * tp, support + guessed, out=np.zeros(5), where=support+guessed > 0)
    one_hot = np.eye(5)[y]
    return {'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall.mean()),
            'accuracy': float(np.mean(pred == y)),
            'nll': float(-np.log(np.maximum(p[np.arange(len(y)), y], eps)).mean()),
            'brier': float(np.sum((p - one_hot)**2, axis=1).mean()),
            'recall': recall, 'f1': f1, 'support': support.astype(int), 'pred': pred}


def bootstrap_macro(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, seed: int, count=400) -> tuple[float,float]:
    rng = np.random.default_rng(seed)
    index = [np.flatnonzero(y == k) for k in range(5)]
    draws = []
    for _ in range(count):
        indices = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in index])
        a = scores(y[indices], p0[indices])['macro_f1']
        b = scores(y[indices], p1[indices])['macro_f1']
        draws.append(b-a)
    return tuple(float(v) for v in np.quantile(draws, [.025,.975]))


def compare(ref: tuple, alt: tuple, *, case: dict, label: str, kind: str, ref_label: str,
            bootstrap=400) -> tuple[dict, list[dict], list[dict]]:
    aligned(ref, alt)
    ids, y, p0 = ref
    p1 = alt[2]
    s0, s1 = scores(y,p0), scores(y,p1)
    delta = s1['macro_f1'] - s0['macro_f1']
    stable = int.from_bytes(hashlib.sha256((case['policy']+str(case['seed'])+label+ref_label).encode()).digest()[:4], 'big')
    lower,upper = bootstrap_macro(y,p0,p1,seed=stable,count=bootstrap)
    row = {'policy': case['policy'], 'seed': case['seed'], 'rejected_cycle': case['rejected_cycle'],
           'model': label, 'kind': kind, 'reference_model': ref_label,
           'reference_macro_f1': s0['macro_f1'], 'macro_f1': s1['macro_f1'],
           'delta_macro_f1': delta, 'delta_balanced_accuracy': s1['balanced_accuracy']-s0['balanced_accuracy'],
           'delta_accuracy': s1['accuracy']-s0['accuracy'],
           'delta_nll': s1['nll']-s0['nll'], 'delta_brier': s1['brier']-s0['brier'],
           'macro_ci025':lower,'macro_ci975':upper,
           'passes_provisional_1pp':macro_tolerance_accept(delta),
           'n_lost_correct': int(np.sum((s0['pred']==y)&(s1['pred']!=y))),
           'n_gained_correct':int(np.sum((s0['pred']!=y)&(s1['pred']==y))),
           'n_changed_predictions':int(np.sum(s0['pred']!=s1['pred']))}
    perclass = [dict(policy=case['policy'], seed=case['seed'], rejected_cycle=case['rejected_cycle'],
                     model=label, reference_model=ref_label, class_name=CLASSES[i],
                     n_validation=int(s0['support'][i]),
                     recall_reference=float(s0['recall'][i]),recall_model=float(s1['recall'][i]),
                     delta_recall=float(s1['recall'][i]-s0['recall'][i]),
                     delta_f1=float(s1['f1'][i]-s0['f1'][i])) for i in range(5)]
    changed = np.flatnonzero(s0['pred'] != s1['pred'])
    flips = [dict(policy=case['policy'],seed=case['seed'],rejected_cycle=case['rejected_cycle'],
                  model=label, reference_model=ref_label, object_id=ids[i],
                  true_class=CLASSES[y[i]], old_prediction=CLASSES[s0['pred'][i]],
                  new_prediction=CLASSES[s1['pred'][i]]) for i in changed]
    return row, perclass, flips


def make_report(job_dir: Path, out: Path, *, allow_partial: bool=False, bootstrap: int=400) -> dict:
    plan = load_json(job_dir / 'plan.json')
    if plan['schema']!='task8b_first_rejection_v1' or len(plan['jobs'])!=30:
        raise ValueError('Unexpected plan contract')
    if sha256(Path(plan['source_audit'])) != plan['source_audit_sha256']:
        raise ValueError('Task8A source report changed')
    out.mkdir(parents=True,exist_ok=True)
    rows, perclass, changed = [], [], []
    totals = []
    for key, evidence in sorted(plan['cases'].items()):
        case = evidence['case']
        # Use historical saved validation evaluations ONLY. No test files read.
        inc_eval = Path(evidence['incumbent_evaluation'])
        candidate_eval = Path(evidence['source_evaluation'])
        inc = load_prediction(inc_eval.parent/'validation_predictions.npz',Path(evidence['initial_checkpoint']))
        original = load_prediction(candidate_eval.parent/'validation_predictions.npz',Path(evidence['candidate_checkpoint']))
        aligned(inc,original)
        reference = compare(inc,original,case=case,label='original_rejected',kind='original',ref_label='incumbent',bootstrap=bootstrap)
        if macro_tolerance_accept(reference[0]['delta_macro_f1']):
            raise ValueError('Original candidate does not meet chosen rejection criterion')
        rows.append(reference[0]);perclass.extend(reference[1]);changed.extend(reference[2])
        retry_predictions = []
        missing = []
        for job in (j for j in plan['jobs'] if j['policy']==case['policy'] and j['seed']==case['seed']):
            folder = Path(job['output_dir'])
            if not (folder/'complete.json').is_file():
                missing.append(job['retry_seed']);continue
            marker = load_json(folder/'complete.json')
            if marker['job'] != job or not marker['validation_only']:
                raise ValueError(f'Retry completion identity mismatch: {folder}')
            checkpoint = folder/'training'/'best_vit_tower.pth'
            if (checkpoint != Path(marker['checkpoint_path']) or sha256(checkpoint)!=marker['checkpoint_sha256']
                    or marker['original_training_manifest_sha256']!=evidence['manifest_sha256']
                    or marker['initial_checkpoint_sha256']!=evidence['initial_checkpoint_sha256']):
                raise ValueError(f'Retry checkpoint/source ancestry mismatch: {folder}')
            training = load_json(folder/'training'/'training_summary.json')
            if (training['initial_checkpoint'] != evidence['initial_checkpoint']
                    or training['train_manifest_sha256'] != evidence['manifest_sha256']
                    or training['validation_manifest_sha256']!=evidence['validation_sha256']
                    or training['seed'] != job['retry_seed']
                    or training['policy']['loader_seed']!=job['retry_seed']):
                raise ValueError(f'Training provenance mismatch: {folder}')
            retr = load_prediction(folder/'evaluation'/'validation_predictions.npz',checkpoint)
            aligned(inc,retr)
            retry_predictions.append((job['retry_seed'],retr))
            for baseline,name in ((inc,'incumbent'),(original,'original_rejected')):
                item = compare(baseline,retr,case=case,label=f"retry_{job['retry_seed']}",
                               kind='single_retry',ref_label=name,bootstrap=bootstrap)
                rows.append(item[0]);perclass.extend(item[1]);changed.extend(item[2])
        if missing and not allow_partial:
            raise FileNotFoundError(f'Missing retries for {key}: {missing}; rerun after completion')
        # Ensemble uses all five retried models and no extra fits.
        if len(retry_predictions)==5:
            mean = np.mean([item[2] for _,item in retry_predictions],axis=0)
            ensemble=(inc[0],inc[1],mean)
            mixed=(inc[0],inc[1],(inc[2]+5*mean)/6)
            for combined,label in ((ensemble,'ensemble_retry_5'),(mixed,'ensemble_incumbent_plus_5')):
                for ref,ref_label in ((inc,'incumbent'),(original,'original_rejected')):
                    item=compare(ref,combined,case=case,label=label,kind='ensemble',ref_label=ref_label,bootstrap=bootstrap)
                    rows.append(item[0]);perclass.extend(item[1]);changed.extend(item[2])
        inc_rows=[r for r in rows if r['policy']==case['policy'] and r['seed']==case['seed']
                  and r['kind']=='single_retry' and r['reference_model']=='incumbent']
        deltas=[r['delta_macro_f1'] for r in inc_rows]
        totals.append({'policy':case['policy'],'seed':case['seed'],'cycle':case['rejected_cycle'],
                       'n_retry_models':len(retry_predictions),
                       'n_passing_provisional_rule':sum(macro_tolerance_accept(d) for d in deltas),
                       'fraction_passing':(sum(macro_tolerance_accept(d) for d in deltas)/len(deltas) if deltas else None),
                       'mean_delta_macro_f1':float(np.mean(deltas)) if deltas else None,
                       'min_delta_macro_f1':min(deltas) if deltas else None,
                       'max_delta_macro_f1':max(deltas) if deltas else None,
                       'original_candidate_delta_macro_f1':reference[0]['delta_macro_f1']})
    dump_csv(out/'model_comparisons.csv',rows)
    dump_csv(out/'per_class_comparisons.csv',perclass)
    if changed: dump_csv(out/'prediction_flips.csv',changed)
    dump_csv(out/'case_summary.csv',totals)
    result={'cases':len(totals),'retries_completed':sum(x['n_retry_models'] for x in totals),
            'retries_planned':30,
            'retries_passing_provisional_rule':sum(x['n_passing_provisional_rule'] for x in totals),
            'validation_only': True,
            'no_test_artifacts_accessed': True,
            'limitation':'Selection and early stopping reuse validation; best-of-five is optimistic. Not an unbiased generalization or full closed-loop estimate.',
            'reported_ensemble_models':['ensemble_retry_5','ensemble_incumbent_plus_5']}
    (out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir',type=Path,default=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/recovery/astrominn_task8b_first_rejection_recovery_v1'))
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=400)
    args=parser.parse_args()
    if args.bootstrap <= 0: parser.error('bootstrap must be positive')
    print(json.dumps(make_report(args.job_dir,args.output or args.job_dir/'report',allow_partial=args.allow_partial,bootstrap=args.bootstrap),indent=2))


if __name__=='__main__':main()
