#!/usr/bin/env python3
"""Read-only paired Task6C vs matching per-sample warm-start reference.

Only validation metrics determine checkpoint/weighting decisions; test not used.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from task6c_build import MODES
from task6c_prepare import PERSISTENT,DATA,SEEDS,POLICIES,bootstrap_check

METRICS=('f1_macro','balanced_accuracy','accuracy','log_loss','brier','ece')

def _rows(path:Path)->dict:
    if not path.is_file():return {}
    result={}
    for row in json.loads(path.read_text()):
        key=(int(row['cycle']),str(row['split']))
        if key in result:raise ValueError(f'Duplicate evaluation record at {path}: {key}')
        result[key]=row
    return result

def _classes(data:Path,suite:str,policy:str,seed:int,cycle:int)->dict:
    p=data/'runs'/suite/policy/f'seed_{seed}'/f'cycle_{cycle:03d}'/'evaluation'/'validation_per_class_metrics.json'
    if not p.is_file():return {}
    return json.loads(p.read_text())['object']

def export(path:Path,values:list[dict]):
    if not values:return
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for v in values for k in v))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(values)

def audit(p:Path,d:Path,mode:str,seed:int,policy:str)->int:
    cfg=MODES[mode];suite=cfg['suite']
    root=p/'runs'/suite/policy/f'seed_{seed}'
    artifact=d/'runs'/suite/policy/f'seed_{seed}'
    state=root/'run_state.json'
    if not state.is_file():return 0
    s=json.loads(state.read_text());complete=int(s['last_completed_cycle'])
    if s.get('active_cycle') is not None:print('WARNING active cycle',mode,seed,policy,s['active_cycle'])
    if complete<0 or complete>9:raise ValueError('Invalid completion counter')
    bootstrap=d/'runs'/cfg['bootstrap']/'shared'/f'seed_{seed}'/'bootstrap_training'/'best_vit_tower.pth'
    if not bootstrap.is_file():raise FileNotFoundError(bootstrap)
    used=set()
    for cycle in range(1,complete+1):
        summary=json.loads((root/f'cycle_{cycle:03d}'/'cycle_summary.json').read_text())
        expected=bootstrap if cycle==1 else artifact/f'cycle_{cycle-1:03d}'/'training'/'best_vit_tower.pth'
        trained=artifact/f'cycle_{cycle:03d}'/'training'/'best_vit_tower.pth'
        for k in ('inference_checkpoint','training_initial_checkpoint'):
            if Path(summary[k])!=expected:raise ValueError(f'Checkpoint ancestry mismatch at {mode}/{seed}/{policy}/{cycle}: {k}')
        if Path(summary['trained_checkpoint'])!=trained or not trained.is_file():
            raise ValueError(f'Missing trained checkpoint at {mode}/{seed}/{policy}/{cycle}')
        train=json.loads((trained.parent/'training_summary.json').read_text())
        tp=train['policy']
        if tp['class_weight_exponent']!=.85 or tp['new_object_weighting']!='equal_object' or tp['class_weight_source']!='training_manifest':
            raise ValueError(f'Wrong Task6C training objective at {mode}/{seed}/{policy}/{cycle}')
        if mode=='macro_f1' and (tp.get('checkpoint_selection')!='object_macro_f1_best' or train.get('checkpoint_selection')!='object_macro_f1_best' or
                                 not (trained.parent/'best_loss_vit_tower.pth').is_file()):
            raise ValueError('Incorrect macro-F1 checkpoint selection')
        ids=summary['selected_ids'];found=set(ids)
        if len(ids)!=(469 if cycle==9 else 473) or len(found)!=len(ids) or used&found:
            raise ValueError(f'Invalid acquisition batch at {mode}/{seed}/{policy}/{cycle}')
        used.update(found)
    if complete==9 and len(used)!=4253:raise ValueError('Did not acquire full U0')
    return complete

def run(mode:str,p:Path,d:Path,out:Path):
    cfg=MODES[mode]
    bootstrap_check(mode,strict=True)
    paired=[];classes=[];status=[]
    for policy in POLICIES:
        for seed in SEEDS:
            n=audit(p,d,mode,seed,policy)
            status.append({'policy':policy,'seed':seed,'completed_cycle':n})
            new=_rows(p/'runs'/cfg['suite']/policy/f'seed_{seed}'/'evaluation_history.json')
            ref=_rows(p/'runs'/cfg['reference']/policy/f'seed_{seed}'/'evaluation_history.json')
            for c in range(0,n+1):
                key=(c,'validation')
                if key not in new or key not in ref:raise ValueError(f'Missing matched validation at {policy}/{seed}/c{c}')
                a,b=new[key],ref[key]
                if a['n_labeled']!=b['n_labeled'] or a['n_acquired']!=b['n_acquired']:
                    raise ValueError('Mismatched acquisition budget')
                row={'mode':mode,'policy':policy,'seed':seed,'cycle':c,'n_labeled':a['n_labeled']}
                for m in METRICS:
                    x,y=a['metrics'][m],b['metrics'][m]
                    row['equal_'+m]=x;row['per_sample_'+m]=y;row['delta_'+m]=x-y
                paired.append(row)
                if c:
                    acls=_classes(d,cfg['suite'],policy,seed,c)
                    bcls=_classes(d,cfg['reference'],policy,seed,c)
                    for name in sorted(set(acls)&set(bcls)):
                        for m in ('f1','precision','recall'):
                            classes.append({'mode':mode,'seed':seed,'policy':policy,'cycle':c,
                                            'class':name,'metric':m,'equal':acls[name][m],
                                            'per_sample':bcls[name][m],'delta':acls[name][m]-bcls[name][m]})
    export(out/'paired_validation.csv',paired)
    export(out/'paired_per_class.csv',classes)
    export(out/'run_status.csv',status)
    full=[x for x in paired if x['cycle']>0]
    early=[x for x in full if x['cycle']<=5]
    def mean(rows,field):return sum(x[field] for x in rows)/len(rows) if rows else None
    summary={'schema':'task6c_equal_new_object_weighting_v1','selection':mode,
             'suite':cfg['suite'],'reference':cfg['reference'],'status':status,
             'paired_cycles':len(full),'mean_delta_macro_f1':mean(full,'delta_f1_macro'),
             'early_delta_macro_f1':mean(early,'delta_f1_macro'),
             'early_delta_balanced_accuracy':mean(early,'delta_balanced_accuracy'),
             'limitation':'Fixed validation used during checkpoint selection; test is report-only. Closed-loop acquisitions differ.'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',choices=list(MODES),required=True)
    p.add_argument('--persistent',type=Path,default=PERSISTENT)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    run(a.selection,a.persistent,a.data,a.out or a.persistent/'closed_loop'/MODES[a.selection]['suite']/'report')
if __name__=='__main__':main()
