#!/usr/bin/env python3
"""Validation-only comparison of ten-epoch continuation, SWA and Task4 incumbents."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from task9_predictions import load_prediction, aligned, CLASSES
from task9a_ensemble import sha, model_stats, count_changes, paired_boot, write_csv

ROOT=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task9b_swa_continuations_v1')


def report(root:Path,out:Path,bootstrap:int=400):
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    if plan.get('schema')!='task9b_swa_v1':raise ValueError('Unexpected SWA plan')
    comparisons=[];classes=[];flips=[];confusions=[];cases=[]
    for case in plan['cases']:
        key=case['key'];inp=case['source'];path=Path(case['output_dir'])
        marker=json.loads((path/'complete.json').read_text(encoding='utf-8'))
        if marker['case']!=case or not marker['validation_only'] or marker['test_used']:
            raise ValueError('Invalid SWA completion contract: '+key)
        results=marker['training_result']
        for label in ('final','swa'):
            ck=path/'training'/f'{label if label=="swa" else "final"}_vit_tower.pth'
            if sha(ck)!=results[f'{"swa" if label=="swa" else "final"}_sha256']:
                raise ValueError('SWA checkpoint hash mismatch')
            evaluation_label='single' if label=='final' else 'swa'
            if sha(path/evaluation_label/'evaluation_summary.json')!=marker[evaluation_label+'_evaluation_summary_sha256']:
                raise ValueError('SWA validation report mutated')
        inc=load_prediction(Path(inp['incumbent_evaluation']).parent/'validation_predictions.npz',Path(inp['initial_checkpoint']))
        rejected=load_prediction(Path(inp['source_evaluation']).parent/'validation_predictions.npz',Path(inp['candidate_checkpoint']))
        single=load_prediction(path/'single'/'validation_predictions.npz',path/'training'/'final_vit_tower.pth')
        swa=load_prediction(path/'swa'/'validation_predictions.npz',path/'training'/'swa_vit_tower.pth')
        for row in (rejected,single,swa):aligned(inc,row)
        models={'incumbent':inc,'original_rejected':rejected,'continued_single':single,'swa':swa}
        cases.append({'case':key,'policy':case['policy'],'seed':case['seed'],'cycle':case['rejected_cycle'],
                      'training_seconds_last_launch':results['training_seconds_this_launch'],
                      'swa_epochs':results['swa_members']})
        for candidate_name,baseline_name in [('continued_single','original_rejected'),('swa','original_rejected'),
                    ('continued_single','incumbent'),('swa','incumbent'),('swa','continued_single')]:
            ids,y,p0=models[baseline_name];p1=models[candidate_name][2]
            s0=model_stats(y,p0);s1=model_stats(y,p1)
            lost,gained,changed=count_changes(y,p0,p1)
            seed=int.from_bytes(__import__('hashlib').sha256(f'{key}/{candidate_name}/{baseline_name}'.encode()).digest()[:4],'big')
            b=paired_boot(y,p0,p1,seed=seed,n=bootstrap)
            comparisons.append({'case':key,'policy':case['policy'],'seed':case['seed'],'cycle':case['rejected_cycle'],
                'model':candidate_name,'reference':baseline_name,'passes_macro_1pp':int(s1['macro_f1']-s0['macro_f1']>=-.01-1e-12),
                'lost_correct':lost,'gained_correct':gained,'changed_predictions':changed,
                **{m:s1[m] for m in s1},**{f'delta_{m}':s1[m]-s0[m] for m in s0},
                **{f'{m}_ci025':b[m][0] for m in b},**{f'{m}_ci975':b[m][1] for m in b}})
            yhat0=p0.argmax(1);yhat1=p1.argmax(1)
            for k,name in enumerate(CLASSES):
                mask=y==k;support=int(mask.sum());before=int(np.sum(mask&(yhat0==y)));after=int(np.sum(mask&(yhat1==y)))
                classes.append({'case':key,'model':candidate_name,'reference':baseline_name,'class':name,'support':support,
                    'correct_ref':before,'correct_model':after,'delta_correct':after-before,
                    'recall_ref':before/support,'recall_model':after/support,'delta_recall':(after-before)/support,
                    'precision_ref':before/max(1,int(np.sum(yhat0==k))),
                    'precision_model':after/max(1,int(np.sum(yhat1==k))),
                    'f1_ref':2*before/(support+int(np.sum(yhat0==k))),
                    'f1_model':2*after/(support+int(np.sum(yhat1==k))),
                    'large_recall_loss_warning':int(before-after>={'AGN':35,'TDE':3,'SN I':100,'SN II':30,'CV':10}[name])})
                for guess,guess_name in enumerate(CLASSES):
                    confusions.append({'case':key,'model':candidate_name,'reference':baseline_name,
                        'true_class':name,'predicted_class':guess_name,
                        'reference_count':int(np.sum(mask&(yhat0==guess))),
                        'model_count':int(np.sum(mask&(yhat1==guess)))})
            for idx in np.flatnonzero(yhat0!=yhat1):
                flips.append({'case':key,'model':candidate_name,'reference':baseline_name,'object_id':ids[idx],
                              'true_class':CLASSES[int(y[idx])],'old_prediction':CLASSES[int(yhat0[idx])],
                              'new_prediction':CLASSES[int(yhat1[idx])]})
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('comparisons.csv',comparisons),('class_recall.csv',classes),('case_summary.csv',cases)]:
        write_csv(out/name,rows)
    write_csv(out/'confusions.csv',confusions)
    if flips:
        write_csv(out/'prediction_flips.csv',flips)
    else:
        (out/'prediction_flips.csv').write_text('case,model,reference,object_id,true_class,old_prediction,new_prediction\n',encoding='utf-8')
    result={'suite':'task9b_swa_continuations_v1','cases':len(cases),'comparisons':len(comparisons),
            'models':['continued_single','swa'],'checkpoint_init':'original_rejected_Task4',
            'averaging_epochs':[6,7,8,9,10],'constant_lr':2e-5,'validation_only':True,
            'test_read_for_selection':False,'bootstrap_draws':bootstrap,
            'interpretation':'SWA vs final single model after identical continuation; validation is repeatedly used in earlier checkpoint choices.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-dir',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    p.add_argument('--bootstrap',type=int,default=400)
    a=p.parse_args()
    if a.bootstrap <1:p.error('bootstrap must be positive')
    print(json.dumps(report(a.job_dir,a.output or a.job_dir/'report',a.bootstrap),indent=2))

if __name__=='__main__':main()
