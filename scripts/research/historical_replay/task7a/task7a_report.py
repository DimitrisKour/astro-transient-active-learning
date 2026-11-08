#!/usr/bin/env python3
"""Read-only comparison of temperature-scaled entropy vs unscaled Task4 entropy."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
SEEDS=(135,246,357)
NEW='astrominn_closed_loop_warm_temperature_entropy_v1'
REF='astrominn_closed_loop_warm_v1'
BOOT='astrominn_historical_replay_v1'

def histories(path:Path)->dict:
    if not path.is_file():return {}
    out={}
    for r in json.loads(path.read_text()):
        key=(int(r['cycle']),r['split'])
        if key in out:raise ValueError(f'Duplicate {key} in {path}')
        out[key]=r
    return out

def write(path:Path,rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def report(p:Path,d:Path,out:Path)->dict:
    results=[];per_class=[];temperature=[];selection=[];status=[]
    for seed in SEEDS:
        new=p/'runs'/NEW/'entropy'/f'seed_{seed}'
        old=p/'runs'/REF/'entropy'/f'seed_{seed}'
        heavy=d/'runs'/NEW/'entropy'/f'seed_{seed}'
        state=new/'run_state.json'
        if not state.exists():
            status.append({'seed':seed,'completed':0,'active':None});continue
        stat=json.loads(state.read_text())
        completed=int(stat.get('last_completed_cycle',0));active=stat.get('active_cycle')
        status.append({'seed':seed,'completed':completed,'active':active})
        old_hist=histories(old/'evaluation_history.json')
        new_hist=histories(new/'evaluation_history.json')
        old_l0=d/'runs'/BOOT/'shared'/f'seed_{seed}'/'bootstrap_training'/'best_vit_tower.pth'
        selected=set()
        for c in range(1,completed+1):
            source=new/f'cycle_{c:03d}'/'cycle_summary.json'
            summary=json.loads(source.read_text())
            before=old_l0 if c==1 else heavy/f'cycle_{c-1:03d}'/'training'/'best_vit_tower.pth'
            actual=heavy/f'cycle_{c:03d}'/'training'/'best_vit_tower.pth'
            if (Path(summary['inference_checkpoint'])!=before or
                Path(summary['training_initial_checkpoint'])!=before or
                Path(summary['trained_checkpoint'])!=actual):
                raise ValueError(f'Broken warm-start checkpoint lineage seed {seed} cycle {c}')
            ids=set(summary['selected_ids'])
            if len(ids)!=(469 if c==9 else 473) or selected.intersection(ids):raise ValueError('Invalid repeated acquisition')
            selected.update(ids)
            training=json.loads((heavy/f'cycle_{c:03d}'/'training'/'training_summary.json').read_text())
            policy=training['policy']
            if (policy['class_weight_exponent']!=0.85 or
                policy['class_weight_source']!='training_manifest' or
                policy['new_object_weighting']!='per_sample'):
                raise ValueError('Task7A training contract mismatch')
            audit_path=heavy/f'cycle_{c:03d}'/'temperature_calibration.json'
            audit=json.loads(audit_path.read_text())
            if audit['cycle']!=c or audit['checkpoint']!=str(before):
                raise ValueError('Invalid checkpoint temperature provenance')
            temperature.append({'seed':seed,'cycle':c,'temperature':audit['temperature'],
                'nll_original':audit['nll_original'],'nll_scaled':audit['nll_scaled']})
            before_selection=set(json.loads((old/f'cycle_{c:03d}'/'cycle_summary.json').read_text())['selected_ids'])
            selection.append({'seed':seed,'cycle':c,'overlap':len(ids & before_selection),
                              'out_of_473':473,'overlap_fraction':len(ids & before_selection)/473})
            old_rec=old_hist.get((c,'validation'));new_rec=new_hist.get((c,'validation'))
            if old_rec and new_rec:
                record={'seed':seed,'cycle':c,'n_labeled':new_rec['n_labeled']}
                for k in ('f1_macro','balanced_accuracy','accuracy','log_loss','brier','ece'):
                    if k in new_rec['metrics'] and k in old_rec['metrics']:
                        record[f'temperature_{k}']=new_rec['metrics'][k]
                        record[f'plain_{k}']=old_rec['metrics'][k]
                        record[f'delta_{k}']=new_rec['metrics'][k]-old_rec['metrics'][k]
                results.append(record)
            cur=heavy/f'cycle_{c:03d}'/'evaluation'/'validation_per_class_metrics.json'
            baseline=d/'runs'/REF/'entropy'/f'seed_{seed}'/f'cycle_{c:03d}'/'evaluation'/'validation_per_class_metrics.json'
            if cur.is_file() and baseline.is_file():
                l=json.loads(cur.read_text())['object'];r=json.loads(baseline.read_text())['object']
                for name in l:
                    for metric in ('f1','recall','precision'):
                        if metric in l[name] and metric in r[name]:
                            per_class.append({'seed':seed,'cycle':c,'class':name,'metric':metric,
                                              'temperature':l[name][metric], 'plain':r[name][metric],
                                              'delta':l[name][metric]-r[name][metric]})
    for name,rows in (('validation_paired.csv',results),('per_class_paired.csv',per_class),
                      ('temperatures.csv',temperature),('selection_overlap.csv',selection),('run_status.csv',status)):
        write(out/name,rows)
    obj={'suite':NEW,'reference':REF,'status':status,'validation_pairs':len(results),
         'calibration_records':len(temperature),'test_used_for_selection':False}
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(obj,indent=2)+'\n')
    return obj

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persistent',type=Path,default=Path('/projects/bcrv/dkourtesis/applecider_adaptive'))
    parser.add_argument('--data',type=Path,default=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive'))
    parser.add_argument('--out',type=Path)
    a=parser.parse_args()
    out=a.out or a.persistent/'closed_loop'/NEW/'report'
    print(json.dumps(report(a.persistent,a.data,out),indent=2))
if __name__=='__main__':main()
