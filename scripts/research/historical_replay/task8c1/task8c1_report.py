#!/usr/bin/env python3
"""Read-only Task8C.1 retention, shadow-policy, acquisition, and compute report.

For rejected cycles candidate validation performance and deployed incumbent
performance are reported separately. No test labels or test metrics are loaded.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

DATA=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
SUITE='astrominn_task8c1_retention_macro1pp_v1'
REFERENCE='astrominn_closed_loop_warm_v1'
JOB=DATA/'closed_loop'/SUITE
CLASSES=('AGN','TDE','SN I','SN II','CV')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def create(job:Path,out:Path,data:Path=DATA)->dict:
    plan=load(job/'plan.json')
    if plan['suite']!=SUITE or len(plan['cases'])!=6:
        raise ValueError('Wrong Task8C.1 experiment plan')
    decisions,per_class,overlap,states=[],[],[],[]
    for case in plan['cases']:
        seed,policy=case['seed'],case['policy']
        root=data/'runs'/SUITE/policy/f'seed_{seed}'
        origin=Path('/projects/bcrv/dkourtesis/applecider_adaptive')/'runs'/REFERENCE/policy/f'seed_{seed}'
        statepath=root/'run_state.json'
        state=load(statepath) if statepath.exists() else {}
        completed=int(state.get('last_completed_cycle',0))
        rows=[]
        for cycle in range(1,completed+1):
            dec_path=root/f'cycle_{cycle:03d}'/'retention_decision.json'
            if not dec_path.is_file():
                raise RuntimeError('Completed cycle without an acceptance decision: '+str(dec_path))
            dec=load(dec_path)
            if (int(dec['cycle'])!=cycle or dec['policy']!=policy or
                int(dec['seed'])!=seed):
                raise ValueError('Decision trajectory metadata mismatch')
            obs=dec['comparison']
            row={'seed':seed,'policy':policy,'cycle':cycle,
                 'replayed_task4':dec['reference_reused'],
                 'accepted':dec['accepted'],
                 'incumbent_checkpoint':dec['incumbent_checkpoint'],
                 'candidate_checkpoint':dec['candidate_checkpoint'],
                 'incumbent_source_cycle':dec['incumbent_source_cycle'],
                 'acquired_objects':dec['acquired_this_cycle'],
                 'cumulative_manifest':dec['training_manifest'],
                 'candidate_minus_prior_macro_f1':obs['macro_delta'],
                 'candidate_minus_prior_balanced_accuracy':obs['balanced_accuracy_delta'],
                 'candidate_macro_f1':obs['candidate_metrics']['macro_f1'],
                 'candidate_balanced_accuracy':obs['candidate_metrics']['balanced_accuracy'],
                 'candidate_accuracy':obs['candidate_metrics']['accuracy'],
                 'prior_incumbent_macro_f1':obs['incumbent_metrics']['macro_f1'],
                 'prior_incumbent_balanced_accuracy':obs['incumbent_metrics']['balanced_accuracy'],
                 'active_incumbent_macro_f1':(obs['candidate_metrics']['macro_f1'] if dec['accepted'] else obs['incumbent_metrics']['macro_f1']),
                 'active_incumbent_balanced_accuracy':(obs['candidate_metrics']['balanced_accuracy'] if dec['accepted'] else obs['incumbent_metrics']['balanced_accuracy']),
                 'shadow_would_accept':obs['shadow_rule']['would_accept'],
                 'shadow_balanced_accuracy_violation':obs['shadow_rule']['ba_violation'],
                 'shadow_recall_violations':';'.join(obs['shadow_rule']['class_recall_violations'])}
            origin_summary=origin/f'cycle_{cycle:03d}'/'cycle_summary.json'
            if origin_summary.is_file():
                source_selected=set(load(origin_summary)['selected_ids'])
                cur_selected=set(load(root/f'cycle_{cycle:03d}'/'cycle_summary.json')['selected_ids'])
                row['acquisition_overlap_count']=len(source_selected & cur_selected)
                row['acquisition_overlap_fraction']=len(source_selected & cur_selected)/len(cur_selected)
                overlap.append({'seed':seed,'policy':policy,'cycle':cycle,
                                'selected':len(cur_selected),'overlap':len(source_selected & cur_selected),
                                'fraction':len(source_selected & cur_selected)/len(cur_selected)})
            for name,v in obs['classes'].items():
                per_class.append({'seed':seed,'policy':policy,'cycle':cycle,'class':name,
                    'accepted':dec['accepted'],'shadow_would_accept':obs['shadow_rule']['would_accept'],
                    **v})
            heavy=data/'runs'/SUITE/policy/f'seed_{seed}'/f'cycle_{cycle:03d}'/'training'/'training_summary.json'
            # Historical Task4 references intentionally have no duplicated
            # training artifacts in the 8C.1 run directory.
            if heavy.is_file():
                train=load(heavy)
                row['new_training_seconds']=train['elapsed_seconds']
                row['new_training_epochs']=train['epochs_completed']
            else:
                if not dec['reference_reused']:
                    raise FileNotFoundError('Missing new training summary: '+str(heavy))
                row['new_training_seconds']=0.0
                row['new_training_epochs']=0
            rows.append(row)
            decisions.append(row)
        states.append({'seed':seed,'policy':policy,'completed':completed,'active':state.get('active_cycle'),
                       'accepted':sum(x['accepted'] for x in rows),
                       'rejected':sum(not x['accepted'] for x in rows),
                       'new_fits':sum(not x['replayed_task4'] for x in rows),
                       'new_training_seconds':sum(x['new_training_seconds'] for x in rows),
                       'final_active_macro_f1':rows[-1]['active_incumbent_macro_f1'] if rows else None,
                       'final_active_balanced_accuracy':rows[-1]['active_incumbent_balanced_accuracy'] if rows else None})
    out.mkdir(parents=True,exist_ok=True)
    csv_write(out/'decisions.csv',decisions)
    csv_write(out/'class_recall.csv',per_class)
    csv_write(out/'acquisition_overlap.csv',overlap)
    csv_write(out/'trajectories.csv',states)
    summary={'suite':SUITE,'status':states,
             'completed_trajectories':sum(r['completed']==9 for r in states),
             'decisions':len(decisions),
             'accepted':sum(r['accepted'] for r in decisions),
             'rejected':sum(not r['accepted'] for r in decisions),
             'shadow_disagreements':sum(r['accepted']!=r['shadow_would_accept'] for r in decisions),
             'new_training_fits':sum(r['new_fits'] for r in states),
             'new_training_hours':sum(r['new_training_seconds'] for r in states)/3600,
             'test_read_for_acceptance':False,
             'limitation':'Validation reused for within-cycle epoch selection and retention; exploratory decision rule. Shadow is observational, not counterfactual.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir',type=Path,default=JOB)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    result=create(args.job_dir,args.out or args.job_dir/'report',args.data)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
