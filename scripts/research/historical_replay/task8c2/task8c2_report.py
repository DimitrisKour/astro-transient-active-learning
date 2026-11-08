#!/usr/bin/env python3
"""Validation-only Task8C.2 report. Candidate, chosen retry, active, and Task4 separate."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

DATA=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
PERSISTENT=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
SUITE='astrominn_task8c2_retry4_macro1pp_v1'
REFERENCE='astrominn_closed_loop_warm_v1'
JOB=DATA/'closed_loop'/SUITE
CLASSES=('AGN','TDE','SN I','SN II','CV')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader();writer.writerows(rows)


def report(job:Path,out:Path,data:Path=DATA,persistent:Path=PERSISTENT):
    plan=load(job/'plan.json')
    if plan['suite']!=SUITE or plan['retry_seeds']!=[1001,1002,1003,1004] or len(plan['cases'])!=6:
        raise ValueError('Wrong frozen Task8C.2 experimental plan')
    decisions, classes, attempts, overlap, summaries, paired=[],[],[],[],[],[]
    for case in plan['cases']:
        policy,seed=case['policy'],case['seed']
        run=data/'runs'/SUITE/policy/f'seed_{seed}'
        original=persistent/'runs'/REFERENCE/policy/f'seed_{seed}'
        ref_heavy=data/'runs'/REFERENCE/policy/f'seed_{seed}'
        state=load(run/'run_state.json') if (run/'run_state.json').is_file() else {}
        completed=int(state.get('last_completed_cycle',0))
        n_retry=0; new_seconds=0.0
        for cycle in range(1,completed+1):
            dpath=run/f'cycle_{cycle:03d}'/'retention_decision.json'
            if not dpath.is_file():
                raise RuntimeError('Cycle completed but retry decision not yet finalized: '+str(dpath))
            dec=load(dpath)
            if dec['schema']!='task8c2_bounded_retry_v1' or dec['cycle']!=cycle or dec['seed']!=seed or dec['policy']!=policy:
                raise ValueError('Decision is not from the requested case/cycle')
            obs=dec['comparison']; base=dec['base_comparison']
            evaldir=Path(dec['incumbent_evaluation_dir'])
            val=load(evaldir/'evaluation_summary.json')
            obj=next(s['object_metrics'] for s in val['splits'] if s['split']=='validation')
            summary=load(run/f'cycle_{cycle:03d}'/'cycle_summary.json')
            row={'policy':policy,'seed':seed,'cycle':cycle,'accepted':dec['accepted'],
                 'accepted_retry_seed':dec['accepted_retry_seed'],
                 'accepted_retry_number':dec['accepted_retry_number'],
                 'attempts_consumed':dec['attempts_consumed'],
                 'base_reference_reused':dec['reference_reused'],
                 'base_macro_delta':base['macro_delta'],
                 'base_ba_delta':base['balanced_accuracy_delta'],
                 'base_macro_f1':base['candidate_metrics']['macro_f1'],
                 'base_balanced_accuracy':base['candidate_metrics']['balanced_accuracy'],
                 'final_candidate_macro_delta':obs['macro_delta'],
                 'final_candidate_ba_delta':obs['balanced_accuracy_delta'],
                 'active_macro_f1':obs['candidate_metrics']['macro_f1'] if dec['accepted'] else obs['incumbent_metrics']['macro_f1'],
                 'active_balanced_accuracy':obs['candidate_metrics']['balanced_accuracy'] if dec['accepted'] else obs['incumbent_metrics']['balanced_accuracy'],
                 'active_accuracy':obj['accuracy'],
                 'shadow_would_accept':obs['shadow_rule']['would_accept'],
                 'shadow_ba_violation':obs['shadow_rule']['ba_violation'],
                 'shadow_class_flags':';'.join(obs['shadow_rule']['class_recall_violations']),
                 'incumbent_checkpoint':dec['incumbent_checkpoint'],
                 'base_checkpoint':dec['base_candidate_checkpoint'],
                 'selected_candidate_checkpoint':dec['candidate_checkpoint'],
                 'cumulative_manifest':dec['training_manifest'],
                 'acquired_this_cycle':dec['acquired_this_cycle']}
            for classname,value in obs['classes'].items():
                classes.append({'policy':policy,'seed':seed,'cycle':cycle,'class':classname,
                                'accepted':dec['accepted'],'retry_accepted':dec['accepted_retry_seed'] is not None,
                                **value})
            ref_summary=original/f'cycle_{cycle:03d}'/'cycle_summary.json'
            if ref_summary.exists():
                ref_ids=set(load(ref_summary)['selected_ids'])
                ids=set(summary['selected_ids'])
                intersection=len(ref_ids&ids)
                overlap.append({'policy':policy,'seed':seed,'cycle':cycle,'overlap':intersection,
                                'selected':len(ids),'overlap_fraction':intersection/len(ids)})
                row['acquisition_overlap_fraction']=intersection/len(ids)
                ref_eval=load(ref_heavy/f'cycle_{cycle:03d}'/'evaluation'/'evaluation_summary.json')
                ref_obj=next(s['object_metrics'] for s in ref_eval['splits'] if s['split']=='validation')
                for metric in ('f1_macro','balanced_accuracy','accuracy'):
                    if metric in ref_obj:
                        row['reference_'+metric]=ref_obj[metric]
                        mapped={'f1_macro':'active_macro_f1','balanced_accuracy':'active_balanced_accuracy','accuracy':'active_accuracy'}[metric]
                        row['delta_'+metric]=row[mapped]-ref_obj[metric]
                paired.append({k:v for k,v in row.items() if k.startswith(('delta_','reference_')) or k in ('policy','seed','cycle','accepted','accepted_retry_seed','active_macro_f1','active_balanced_accuracy','active_accuracy')})
            base_train=data/'runs'/SUITE/policy/f'seed_{seed}'/f'cycle_{cycle:03d}'/'training'/'training_summary.json'
            if base_train.is_file():new_seconds+=float(load(base_train)['elapsed_seconds'])
            count=0
            for number in (1,2,3,4):
                a=run/f'cycle_{cycle:03d}'/'retry_attempts'/f'retry_{number}.json'
                if not a.is_file():continue
                retry=load(a);count+=1;n_retry+=1
                new_seconds+=float(retry['elapsed_training_seconds'])
                attempts.append({'policy':policy,'seed':seed,'cycle':cycle,
                                 'retry_number':number,'retry_seed':retry['retry_seed'],
                                 'accepted':retry['comparison']['accepted'],
                                 'macro_delta':retry['comparison']['macro_delta'],
                                 'balanced_accuracy_delta':retry['comparison']['balanced_accuracy_delta'],
                                 'training_seconds':retry['elapsed_training_seconds']})
            if count!=dec['attempts_consumed']:
                raise ValueError(f'Retry count inconsistent with final decision {policy} {seed} {cycle}')
            decisions.append(row)
        local=[r for r in decisions if r['policy']==policy and r['seed']==seed]
        summaries.append({'policy':policy,'seed':seed,'completed':completed,
                          'active_cycle':state.get('active_cycle'),
                          'accepted':sum(r['accepted'] for r in local),
                          'rejected':sum(not r['accepted'] for r in local),
                          'retry_fits':n_retry,'retry_accepted':sum(r['accepted_retry_seed'] is not None for r in local),
                          'total_new_training_seconds':new_seconds,
                          'final_active_macro_f1':local[-1]['active_macro_f1'] if local else None,
                          'final_active_balanced_accuracy':local[-1]['active_balanced_accuracy'] if local else None})
    out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('decisions.csv',decisions),('class_recall.csv',classes),
                          ('retry_attempts.csv',attempts),('acquisition_overlap.csv',overlap),
                          ('trajectories.csv',summaries),('paired_task4.csv',paired)]:
        save_csv(out/filename,rows)
    final={'suite':SUITE,'status':summaries,
           'completed_trajectories':sum(x['completed']==9 for x in summaries),
           'decisions':len(decisions),'accepted':sum(x['accepted'] for x in decisions),
           'rejected':sum(not x['accepted'] for x in decisions),
           'retries_trained':len(attempts),
           'accepted_by_retry':sum(x['accepted_retry_seed'] is not None for x in decisions),
           'shadow_disagreements':sum(x['accepted']!=x['shadow_would_accept'] for x in decisions),
           'new_training_hours':sum(x['total_new_training_seconds'] for x in summaries)/3600,
           'test_used_for_selection':False,
           'limitations':'Validation reused for checkpoint selection and retention; shadow outcomes are observational, not alternative closed-loop trajectories.'}
    (out/'summary.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n')
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-dir',type=Path,default=JOB)
    p.add_argument('--out',type=Path)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--persistent',type=Path,default=PERSISTENT)
    a=p.parse_args()
    print(json.dumps(report(a.job_dir,a.out or a.job_dir/'report',a.data,a.persistent),indent=2))
if __name__=='__main__':main()
