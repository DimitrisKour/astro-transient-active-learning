#!/usr/bin/env python3
"""Task9A: fixed, read-only validation probability ensembles of Task8B first rejections.

CPU only. Does not retrain, evaluate test predictions, or choose validation-best members.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np

from task9_predictions import load_prediction, aligned, scores, CLASSES

B = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/recovery/astrominn_task8b_first_rejection_recovery_v1')
OUT = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task9a_fixed_ensembles_v1')
VARIANTS = ('E2', 'E4', 'I_E2', 'I_E4', 'E5', 'I_E5')
BASELINES = ('incumbent', 'original_rejected', 'first_passing_retry')
EXPECTED = (1001, 1002, 1003, 1004, 1005)


def jload(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda:f.read(1024*1024), b''): h.update(piece)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows: raise ValueError(f'Empty output: {path}')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def ece(y, p, bins=15):
    pred=p.argmax(1); confidence=p.max(1); correct=pred==y
    indices=np.minimum((confidence*bins).astype(int),bins-1)
    return float(sum(np.mean(indices==k)*abs(float(correct[indices==k].mean())-float(confidence[indices==k].mean()))
                     for k in range(bins) if np.any(indices==k)))


def model_stats(y,p):
    z=scores(y,p)
    return {k:z[k] for k in ('macro_f1','balanced_accuracy','accuracy','nll','brier')}|{'ece':ece(y,p)}


def assemble(inc, members, label):
    n = {'E2':2,'E4':4,'I_E2':2,'I_E4':4,'E5':5,'I_E5':5}[label]
    arrays = [q[2] for q in members[:n]]
    if label.startswith('I_'): arrays.insert(0,inc[2])
    return inc[0],inc[1],np.mean(arrays, axis=0)


def count_changes(y,p0,p1):
    a=p0.argmax(1)==y;b=p1.argmax(1)==y
    return int(np.sum(a&~b)), int(np.sum(~a&b)), int(np.sum(p0.argmax(1)!=p1.argmax(1)))


def paired_boot(y, p0, p1, *, seed, n=400):
    groups=[np.flatnonzero(y==i) for i in range(5)]
    if any(len(x)==0 for x in groups): raise ValueError('Missing validation class')
    rng=np.random.default_rng(seed)
    metric=('macro_f1','balanced_accuracy')
    diffs={m:[] for m in metric}
    for _ in range(n):
        idx=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        s0=scores(y[idx],p0[idx]); s1=scores(y[idx],p1[idx])
        for m in metric: diffs[m].append(s1[m]-s0[m])
    return {m:tuple(float(v) for v in np.quantile(diffs[m],[.025,.975])) for m in metric}


def verified_case(plan, key):
    inp=plan['cases'][key];case=inp['case']; ck=Path(inp['candidate_checkpoint']); ic=Path(inp['initial_checkpoint'])
    if sha(ck)!=inp['candidate_checkpoint_sha256'] or sha(ic)!=inp['initial_checkpoint_sha256']:
        raise ValueError(f'Original checkpoint integrity mismatch: {key}')
    if sha(Path(inp['training_manifest']))!=inp['manifest_sha256']:
        raise ValueError(f'Train manifest changed: {key}')
    if sha(Path(inp['validation_manifest']))!=inp['validation_sha256']:
        raise ValueError(f'Validation manifest changed: {key}')
    for source_name, digest_key in (('source_train_summary','source_training_summary_sha256'),
                                    ('source_evaluation','source_candidate_evaluation_sha256'),
                                    ('incumbent_evaluation','source_incumbent_evaluation_sha256')):
        if sha(Path(inp[source_name])) != inp[digest_key]:
            raise ValueError(f'Original Task4 provenance changed: {source_name} {key}')
    inc=load_prediction(Path(inp['incumbent_evaluation']).parent/'validation_predictions.npz', ic)
    orig=load_prediction(Path(inp['source_evaluation']).parent/'validation_predictions.npz', ck)
    aligned(inc,orig)
    rows=[x for x in plan['jobs'] if x['policy']==case['policy'] and x['seed']==case['seed']]
    if [r['retry_seed'] for r in rows]!=list(EXPECTED):raise ValueError(f'Unexpected retry seed ordering: {key}')
    retries=[]
    for r in rows:
        d=Path(r['output_dir']); marker=jload(d/'complete.json'); cp=d/'training'/'best_vit_tower.pth'
        if (marker['job'] != r or not marker['validation_only'] or
            Path(marker['checkpoint_path']) != cp or sha(cp)!=marker['checkpoint_sha256'] or
            marker['original_training_manifest_sha256']!=inp['manifest_sha256'] or
            marker['initial_checkpoint_sha256']!=inp['initial_checkpoint_sha256']):
            raise ValueError(f'Retry completion/provenance mismatch: {d}')
        train=jload(d/'training'/'training_summary.json')
        if (train['seed']!=r['retry_seed'] or train['policy']['loader_seed']!=r['retry_seed'] or
            train['train_manifest_sha256']!=inp['manifest_sha256'] or
            train['validation_manifest_sha256']!=inp['validation_sha256'] or
            train['initial_checkpoint']!=str(ic) or sha(d/'training'/'training_summary.json')!=marker['training_summary_sha256']):
            raise ValueError(f'Retry training metadata mismatch: {d}')
        q=load_prediction(d/'evaluation'/'validation_predictions.npz', cp)
        aligned(inc,q);retries.append(q)
    return inc,orig,retries,case


def report(task8b:Path=B, out:Path=OUT, *, bootstrap=400):
    plan=jload(task8b/'plan.json')
    if plan['schema']!='task8b_first_rejection_v1' or len(plan['cases'])!=6 or len(plan['jobs'])!=30:
        raise ValueError('Unexpected Task8B plan')
    if sha(Path(plan['source_audit']))!=plan['source_audit_sha256']:
        raise ValueError('Task8A audit changed')
    models=[]; per_class=[]; flips=[]; confusions=[]; definitions=[]; case_stats=[]
    for key in sorted(plan['cases']):
        inc,orig,retries,case=verified_case(plan,key)
        ids,y,pinc=inc; p0=orig[2]
        passing=[i for i,p in enumerate(retries) if scores(y,p[2])['macro_f1']-scores(y,pinc)['macro_f1'] >= -0.01-1e-12]
        refs={'incumbent':inc, 'original_rejected':orig}
        if passing:refs['first_passing_retry']=retries[passing[0]]
        case_stats.append({'case':key,'policy':case['policy'],'seed':case['seed'],'cycle':case['rejected_cycle'],
                          'first_passing_seed':EXPECTED[passing[0]] if passing else None})
        for variant in VARIANTS:
            ensemble=assemble(inc,retries,variant);p=ensemble[2]
            members=list(EXPECTED[:{'E2':2,'E4':4,'I_E2':2,'I_E4':4,'E5':5,'I_E5':5}[variant]])
            definitions.append({'case':key,'ensemble':variant,'includes_incumbent':variant.startswith('I_'),
                                'retry_seeds':','.join(map(str,members))})
            stat=model_stats(y,p)
            for ref_label,ref in refs.items():
                p_ref=ref[2];sref=model_stats(y,p_ref);lost,gained,changed=count_changes(y,p_ref,p)
                bseed=int.from_bytes(hashlib.sha256(f'{key}:{variant}:{ref_label}'.encode()).digest()[:4],'big')
                b=paired_boot(y,p_ref,p,seed=bseed,n=bootstrap)
                models.append({'case':key,'policy':case['policy'],'seed':case['seed'],
                    'rejected_cycle':case['rejected_cycle'],'ensemble':variant,'reference':ref_label,
                    'n_models':len(members)+int(variant.startswith('I_')),'passes_macro_1pp':int(stat['macro_f1']-sref['macro_f1']>=-.01-1e-12),
                    'lost_correct':lost,'gained_correct':gained,'changed_predictions':changed,
                    **{f'{m}':stat[m] for m in stat},
                    **{f'delta_{m}':stat[m]-sref[m] for m in stat},
                    **{f'{m}_ci025':b[m][0] for m in b}, **{f'{m}_ci975':b[m][1] for m in b}})
                pred_ref=p_ref.argmax(1);pred=p.argmax(1)
                for k,name in enumerate(CLASSES):
                    c=y==k;correct_before=int(np.sum(c&(pred_ref==y)));correct_after=int(np.sum(c&(pred==y)))
                    per_class.append({'case':key,'ensemble':variant,'reference':ref_label,'class':name,
                       'support':int(c.sum()),'correct_ref':correct_before,'correct_ensemble':correct_after,
                       'delta_correct':correct_after-correct_before,
                       'recall_ref':correct_before/int(c.sum()),'recall_ensemble':correct_after/int(c.sum()),
                       'delta_recall':(correct_after-correct_before)/int(c.sum()),
                       'precision_ref':correct_before/max(1,int(np.sum(pred_ref==k))),
                       'precision_ensemble':correct_after/max(1,int(np.sum(pred==k))),
                       'f1_ref':2*correct_before/(int(c.sum())+int(np.sum(pred_ref==k))),
                       'f1_ensemble':2*correct_after/(int(c.sum())+int(np.sum(pred==k))),
                       'large_recall_loss_warning':int(correct_before-correct_after>={'AGN':35,'TDE':3,'SN I':100,'SN II':30,'CV':10}[name])})
                    for guess,guess_name in enumerate(CLASSES):
                        confusions.append({'case':key,'ensemble':variant,'reference':ref_label,'true_class':name,
                            'predicted_class':guess_name,'reference_count':int(np.sum(c&(pred_ref==guess))),
                            'ensemble_count':int(np.sum(c&(pred==guess)))})
                for idx in np.flatnonzero(pred_ref!=pred):
                    flips.append({'case':key,'ensemble':variant,'reference':ref_label,'object_id':ids[idx],
                                  'true_class':CLASSES[int(y[idx])],'old_prediction':CLASSES[int(pred_ref[idx])],
                                  'new_prediction':CLASSES[int(pred[idx])]})
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'ensemble_comparisons.csv',models)
    write_csv(out/'class_recall.csv',per_class)
    if flips:
        write_csv(out/'prediction_flips.csv',flips)
    else:
        (out/'prediction_flips.csv').write_text('case,ensemble,reference,object_id,true_class,old_prediction,new_prediction\n',encoding='utf-8')
    write_csv(out/'ensemble_definitions.csv',definitions)
    write_csv(out/'case_summary.csv',case_stats)
    write_csv(out/'confusions.csv',confusions)
    result={'suite':'task9a_fixed_ensembles_v1','cases':len(case_stats), 'variants':list(VARIANTS),
            'comparisons':len(models),'new_training_fits':0,'validation_only':True,
            'input_task8b_plan_sha256':sha(task8b/'plan.json'),
            'analysis_script_sha256':sha(Path(__file__)),
            'task9_prediction_reader_sha256':sha(Path(__import__('task9_predictions').__file__)),
            'test_data_used':False,'bootstrap_draws':bootstrap,
            'interpretation':'Descriptive repeated-validation analysis, not unbiased generalization; no member tuning.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('command',choices=('verify','run'))
    a.add_argument('--task8b',type=Path,default=B)
    a.add_argument('--output',type=Path,default=OUT)
    a.add_argument('--bootstrap',type=int,default=400)
    p=a.parse_args()
    if p.command=='verify':
        plan=jload(p.task8b/'plan.json')
        print(json.dumps({'cases':[{'case':key,'retry_models_verified':len(verified_case(plan,key)[2])} for key in sorted(plan['cases'])],
              'all_30_retries_verified':True,'new_gpu_work':False},indent=2))
    else:
        if p.bootstrap < 1: a.error('bootstrap must be positive')
        print(json.dumps(report(p.task8b,p.output,bootstrap=p.bootstrap),indent=2))

if __name__=='__main__': main()
