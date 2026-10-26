#!/usr/bin/env python3
"""Task11 validation-only learning curves and classwise false-alert diagnostics.

No test data or predictions are read or included. Original Task4 is used only
as matched same-seed validation reference.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

SUITE='astrominn_task11_advanced_acquisition_v1'
TASK4='astrominn_closed_loop_warm_v1'
POLICIES=('badge','core_set','fisher_batch')
BASELINES=('random','entropy','entropy_diversity','entropy_balanced_explore')
SEEDS=(135,246,357)
PERSISTENT=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')


def read_json(path):return json.loads(Path(path).read_text())


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def validation_records(suite,policy,seed):
    base=PERSISTENT/'runs'/suite/policy/f'seed_{seed}'
    history=read_json(base/'evaluation_history.json')
    values={}
    for row in history:
        if row['split']=='validation':
            cycle=int(row['cycle'])
            if cycle in values:raise ValueError(f'Duplicate validation evaluation: {base}, {cycle}')
            values[cycle]=row
    if sorted(values)!=list(range(10)):
        raise ValueError(f'Expected cycles 0..9 for {suite}/{policy}/{seed}, got {sorted(values)}')
    return values


def report(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results=[];paired=[];classes=[];overlap=[]
    baselines={(p,s):validation_records(TASK4,p,s) for p in BASELINES for s in SEEDS}
    for policy in POLICIES:
        for seed in SEEDS:
            records=validation_records(SUITE,policy,seed)
            for cycle,record in records.items():
                metrics=record['metrics']
                row={'policy':policy,'seed':seed,'cycle':cycle,'n_labeled':record['n_labeled'],
                     'macro_f1':metrics['f1_macro'],'balanced_accuracy':metrics['balanced_accuracy'],
                     'accuracy':metrics['accuracy'],'nll':metrics['log_loss'],
                     'brier':metrics['brier'],'ece':metrics['ece'],
                     'average_precision_macro':metrics.get('average_precision_macro')}
                results.append(row)
                for reference in BASELINES:
                    ref=baselines[(reference,seed)][cycle]
                    paired.append({'policy':policy,'seed':seed,'cycle':cycle,'reference':reference,
                        'macro_f1_pp':100*(metrics['f1_macro']-ref['metrics']['f1_macro']),
                        'balanced_accuracy_pp':100*(metrics['balanced_accuracy']-ref['metrics']['balanced_accuracy']),
                        'accuracy_pp':100*(metrics['accuracy']-ref['metrics']['accuracy']),
                        'same_label_budget':record['n_labeled']==ref['n_labeled']})
                if cycle:
                    out=DATA/'runs'/SUITE/policy/f'seed_{seed}'/f'cycle_{cycle:03d}/evaluation'
                    clsfile=out/'validation_per_class_metrics.json'
                    confusion=out/'validation_object_confusion.csv'
                    if not clsfile.is_file() or not confusion.is_file():
                        raise FileNotFoundError(f'Classwise validation outputs missing: {clsfile}')
                    per_class=read_json(clsfile)['object']
                    with open(confusion,newline='') as f: rows=list(csv.reader(f))
                    heads=rows[0][1:]
                    matrix=[[int(v) for v in r[1:]] for r in rows[1:]]
                    for i,name in enumerate(heads):
                        cm=matrix[i]
                        tp=cm[i];fn=sum(cm)-tp;fp=sum(r[i] for r in matrix)-tp
                        m=per_class[name]
                        classes.append({'policy':policy,'seed':seed,'cycle':cycle,
                                        'class':name,'tp':tp,'fp':fp,'fn':fn,
                                        'support':tp+fn,'precision':m['precision'],
                                        'recall':m['recall'],'f1':m['f1']})
                    selected=PERSISTENT/'runs'/SUITE/policy/f'seed_{seed}'/f'cycle_{cycle:03d}/selection.json'
                    reference_selection=PERSISTENT/'runs'/TASK4/'entropy'/f'seed_{seed}'/f'cycle_{cycle:03d}/selection.json'
                    s=read_json(selected)['ids'];r=read_json(reference_selection)['ids']
                    if len(s)!=len(r):raise ValueError('Acquisition budget diverged from Task4 entropy')
                    overlap.append({'policy':policy,'seed':seed,'cycle':cycle,'acquisition_size':len(s),
                                    'overlap_with_entropy':len(set(s)&set(r)),
                                    'overlap_fraction':len(set(s)&set(r))/len(s)})
    if not all(row['same_label_budget'] for row in paired):raise ValueError('Matched labeling budgets diverged')
    results.sort(key=lambda r:(r['policy'],r['seed'],r['cycle']))
    write_csv(output/'trajectories.csv',results)
    write_csv(output/'paired_task4.csv',paired)
    write_csv(output/'classwise_validation.csv',classes)
    write_csv(output/'acquisition_overlap.csv',overlap)
    summary={'suite':SUITE,'complete_trajectories':9,'cycles':9,'training_fits':81,
             'validation_only':True,'test_predictions_read':False,
             'cases':{},'interpretation':'Matched descriptive validation curves, not independent test inference'}
    for p in POLICIES:
        pr=[r for r in results if r['policy']==p]
        au_f1=sum(sum(row['macro_f1'] for row in pr if row['seed']==s)/10 for s in SEEDS)/3
        au_ba=sum(sum(row['balanced_accuracy'] for row in pr if row['seed']==s)/10 for s in SEEDS)/3
        mean_overlap=sum(row['overlap_fraction'] for row in overlap if row['policy']==p)/27
        summary['cases'][p]={'mean_cycle0_to9_macro_f1':au_f1,'mean_cycle0_to9_balanced_accuracy':au_ba,
                             'mean_same_cycle_overlap_entropy':mean_overlap}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,default=PERSISTENT/'closed_loop'/SUITE/'report')
    args=a.parse_args()
    report(args.output)
