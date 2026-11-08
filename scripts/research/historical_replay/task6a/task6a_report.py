#!/usr/bin/env python3
"""Read-only Task6A 0.5-versus-Task4 0.85 matched-budget comparison.

Validation drives scientific choice. Test is only descriptive and is not exported
in the primary comparison CSV. Checks checkpoint lineage, manifests and policy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SEEDS = (135,246,357)
POLICIES = ('entropy','entropy_balanced_explore')
NEW = 'astrominn_closed_loop_warm_weight05_v1'
REF = 'astrominn_closed_loop_warm_v1'
BASE = 'astrominn_historical_replay_v1'


def records(path: Path) -> dict[tuple[int,str],dict]:
    if not path.is_file():
        return {}
    obj=json.loads(path.read_text())
    result={}
    for x in obj:
        key=(int(x['cycle']),x['split'])
        if key in result:
            raise ValueError(f'Duplicate evaluations in {path} {key}')
        result[key]=x
    return result


def validate_trajectory(p: Path, d: Path, policy: str, seed: int) -> int:
    state_dir=p/'runs'/NEW/policy/f'seed_{seed}'
    data_dir=d/'runs'/NEW/policy/f'seed_{seed}'
    state_file=state_dir/'run_state.json'
    if not state_file.is_file():
        return 0
    state=json.loads(state_file.read_text())
    if state.get('active_cycle') is not None:
        print('WARNING active cycle:',policy,seed,state['active_cycle'])
    completed=int(state.get('last_completed_cycle',0))
    used=set()
    l0=d/'runs'/BASE/'shared'/f'seed_{seed}'/'bootstrap_training'/'best_vit_tower.pth'
    for cycle in range(1,completed+1):
        summary=json.loads((state_dir/f'cycle_{cycle:03d}'/'cycle_summary.json').read_text())
        previous=l0 if cycle==1 else data_dir/f'cycle_{cycle-1:03d}'/'training'/'best_vit_tower.pth'
        current=data_dir/f'cycle_{cycle:03d}'/'training'/'best_vit_tower.pth'
        for field in ('inference_checkpoint','training_initial_checkpoint'):
            if Path(summary[field])!=previous:
                raise ValueError(f'Warm-start ancestry mismatch {policy}/{seed}/c{cycle} {field}')
        if Path(summary['trained_checkpoint'])!=current:
            raise ValueError(f'Wrong checkpoint {policy}/{seed}/c{cycle}')
        if not current.is_file():
            raise FileNotFoundError(current)
        train=json.loads((data_dir/f'cycle_{cycle:03d}'/'training'/'training_summary.json').read_text())
        if train['policy']['class_weight_exponent']!=0.5 or train['policy']['new_object_weighting']!='per_sample':
            raise ValueError(f'Wrong training objective {policy}/{seed}/c{cycle}')
        if train['policy']['class_weight_source']!='training_manifest':
            raise ValueError('Wrong class weight source')
        ids=summary['selected_ids']; found=set(ids)
        if len(ids)!=(469 if cycle==9 else 473) or len(found)!=len(ids) or used.intersection(found):
            raise ValueError(f'Invalid acquisition batch {policy}/{seed}/c{cycle}')
        used.update(found)
    if completed==9 and len(used)!=4253:
        raise ValueError(f'Pool not exhausted for {policy}/{seed}: {len(used)}')
    return completed


def object_per_class(data: Path, suite: str, policy: str, seed: int, cycle: int) -> dict:
    file=data/'runs'/suite/policy/f'seed_{seed}'/f'cycle_{cycle:03d}'/'evaluation'/'validation_per_class_metrics.json'
    if not file.exists():
        return {}
    return json.loads(file.read_text())['object']


def export(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    keys=list(dict.fromkeys(k for x in rows for k in x))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def run(persistent: Path, data: Path, out: Path) -> None:
    primary=[]; per_class=[]; status=[]
    for policy in POLICIES:
        for seed in SEEDS:
            completed=validate_trajectory(persistent,data,policy,seed)
            status.append({'policy':policy,'seed':seed,'completed_cycle':completed})
            new=records(persistent/'runs'/NEW/policy/f'seed_{seed}'/'evaluation_history.json')
            old=records(persistent/'runs'/REF/policy/f'seed_{seed}'/'evaluation_history.json')
            for c in range(0,completed+1):
                key=(c,'validation')
                if key not in new or key not in old:
                    raise ValueError(f'Missing matched validation row at {policy}/{seed}/c{c}')
                a,b=new[key],old[key]
                if a['n_labeled']!=b['n_labeled'] or a['n_acquired']!=b['n_acquired']:
                    raise ValueError('Different label budget at matched cycle')
                row={'policy':policy,'seed':seed,'cycle':c,'n_labeled':a['n_labeled']}
                for metric in ('f1_macro','balanced_accuracy','accuracy','log_loss','brier','ece'):
                    x,y=a['metrics'][metric],b['metrics'][metric]
                    row['weight05_'+metric]=x;row['weight085_'+metric]=y;row['delta_'+metric]=x-y
                primary.append(row)
                if c:
                    cl05=object_per_class(data,NEW,policy,seed,c)
                    cl85=object_per_class(data,REF,policy,seed,c)
                    for cls in sorted(set(cl05)&set(cl85)):
                        for m in ('precision','recall','f1'):
                            per_class.append({'policy':policy,'seed':seed,'cycle':c,'class':cls,
                                              'metric':m,'weight05':cl05[cls][m],
                                              'weight085':cl85[cls][m],
                                              'delta':cl05[cls][m]-cl85[cls][m]})
    export(out/'paired_validation.csv',primary)
    export(out/'paired_per_class.csv',per_class)
    export(out/'run_status.csv',status)
    full=[x for x in primary if x['cycle']>0]
    early=[x for x in full if x['cycle']<=5]
    aggregate={
      'schema':'task6a_comparison_v1', 'reference':REF, 'new':NEW,
      'status':status, 'completed_validation_pairs':len(full),
      'mean_delta_macro_f1_all_available':sum(x['delta_f1_macro'] for x in full)/len(full) if full else None,
      'mean_delta_macro_f1_cycles1to5':sum(x['delta_f1_macro'] for x in early)/len(early) if early else None,
      'mean_delta_balanced_accuracy_cycles1to5':sum(x['delta_balanced_accuracy'] for x in early)/len(early) if early else None,
      'limitation':'Validation was used for within-cycle loss-best checkpoints; test not used for protocol selection',
    }
    (out/'summary.json').write_text(json.dumps(aggregate,indent=2,sort_keys=True)+'\n')
    print(json.dumps(aggregate,indent=2))
    print('Report:',out)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--persistent',type=Path,default=Path('/projects/bcrv/dkourtesis/applecider_adaptive'))
    p.add_argument('--data',type=Path,default=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive'))
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    run(a.persistent,a.data,a.out or a.persistent/'closed_loop'/NEW/'report')

if __name__=='__main__':main()
