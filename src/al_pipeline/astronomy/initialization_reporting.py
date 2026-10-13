"""Portable metadata-only reports for matched initialization trajectories."""
from __future__ import annotations
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from al_pipeline.astronomy.initialization_compare import MODES
from al_pipeline.astronomy.training_diagnostics import load_plan, sha256, write_json

METRICS = ('f1_macro','balanced_accuracy','accuracy')
ALL_MODES = ('random_restart', *MODES)

def _csv(path, rows, fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def report_comparison(plan_path: Path, output_dir: Path, *, plots: bool = True) -> dict:
    plan_path=Path(plan_path).resolve();plan=load_plan(plan_path)
    if plan.get('kind') != 'initialization_compare': raise ValueError('Wrong plan kind.')
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    records=[];incomplete=[];fit_times=[];per_class=[];digest=sha256(plan_path)
    def add(seed, policy, cycle, mode, objects, samples, training, evaluation, origin, task_id=''):
        if len(evaluation)!=2 or {s['split'] for s in evaluation}!={'validation','test'}:
            raise ValueError('Incomplete/duplicate evaluation splits.')
        for split in evaluation:
            for metric in METRICS:
                value=float(split['object_metrics'][metric])*100
                if not math.isfinite(value) or not 0<=value<=100: raise ValueError('Invalid object metric.')
                records.append({'seed':seed,'policy':policy,'cycle':cycle,'mode':mode,'split':split['split'],
                                'metric':metric,'value_percent':value,'objects':objects,'samples':samples,
                                'fraction':objects/plan['full_data_objects'],'origin':origin,'task_id':task_id,
                                'epochs_completed':training['epochs_completed'],'best_epoch':training['best_epoch'],
                                'training_seconds':training['elapsed_seconds']})
    for ref in plan['references']:
        add(ref['seed'],ref['policy'],ref['cycle'],'random_restart',ref['objects'],ref['samples'],
            ref['training'],ref['evaluation'],'reused_original')
    for task in plan['tasks']:
        # Relative paths keep reporting portable after downloading metadata only.
        directory=plan_path.parent/'tasks'/task['task_id'];p=directory/'result.json'
        if not p.exists():
            status=directory/'status.json'
            incomplete.append({'task_id':task['task_id'],'seed':task['seed'],'cycle':task['cycle'],
                               'mode':task['mode'],'status':json.loads(status.read_text()).get('status') if status.exists() else 'not_started'})
            continue
        result=json.loads(p.read_text())
        if result.get('task')!=task or result.get('plan_sha256')!=digest: raise ValueError('Result belongs to a different plan/task.')
        init=result['initialization']
        if not init['loaded_vit_checkpoint']: raise ValueError('A new fit did not load its expected checkpoint.')
        if task['initial_task']:
            parent=plan_path.parent/'tasks'/task['initial_task']/'result.json'
            if not parent.exists(): raise ValueError('Completed warm-start result lacks its predecessor.')
            pr=json.loads(parent.read_text())
            predecessor=next(t for t in plan['tasks'] if t['task_id']==task['initial_task'])
            if pr.get('task')!=predecessor or pr.get('plan_sha256')!=digest: raise ValueError('Invalid predecessor identity.')
            if init['checkpoint_sha256']!=pr['checkpoint_sha256']: raise ValueError('Warm-start checkpoint lineage mismatch.')
        elif init['checkpoint_sha256']!=task['initial_checkpoint_sha256']:
            raise ValueError('L0 checkpoint lineage mismatch.')
        fit_times.append({'task_id':task['task_id'],'seed':task['seed'],'policy':task['policy_name'],
                          'cycle':task['cycle'],'mode':task['mode'],'shared_cycle_1':task['cycle']==1,
                          'training_seconds':result['training']['elapsed_seconds'],
                          'wall_seconds_this_attempt':result['wall_seconds_this_attempt']})
        for mode in task['modes']:
            add(task['seed'],task['policy_name'],task['cycle'],mode,task['objects'],task['samples'],
                result['training'],result['evaluation'],'new_fit',task['task_id'])
            for split, classes in result.get('object_per_class',{}).items():
                for name, metrics in classes.items():
                    per_class.append({'seed':task['seed'],'policy':task['policy_name'],'cycle':task['cycle'],
                                      'mode':mode,'split':split,'class_name':name,**metrics})
    grouped=defaultdict(list);paired=defaultdict(dict)
    for r in records:
        grouped[r['policy'],r['mode'],r['cycle'],r['split'],r['metric']].append(r)
        paired[r['seed'],r['policy'],r['cycle'],r['split'],r['metric']][r['mode']]=r['value_percent']
    aggregates=[];differences=[]
    for (policy,mode,cycle,split,metric), rows in sorted(grouped.items()):
        values=[r['value_percent'] for r in rows]
        aggregates.append({'policy':policy,'mode':mode,'cycle':cycle,'split':split,'metric':metric,
                           'mean_percent':statistics.mean(values),'sd_pp':statistics.stdev(values) if len(values)>1 else None,
                           'completed_seeds':len(values),'planned_seeds':len(plan['seeds']),
                           'all_seeds_complete':len(values)==len(plan['seeds'])})
    for (seed,policy,cycle,split,metric), values in sorted(paired.items()):
        for a,b in (('restart_from_l0','random_restart'),('warm_start','random_restart'),('warm_start','restart_from_l0')):
            if a in values and b in values:
                differences.append({'seed':seed,'policy':policy,'cycle':cycle,'split':split,'metric':metric,
                                    'comparison':a+' minus '+b,'delta_pp':values[a]-values[b]})
    # Complete trajectories only for plateau/drop summaries. Do not fill missing fits.
    curves=defaultdict(dict)
    for r in records: curves[r['seed'],r['policy'],r['mode'],r['split'],r['metric']][r['cycle']]=r['value_percent']
    trajectories=[]
    for (seed,policy,mode,split,metric), values in sorted(curves.items()):
        complete=set(values)==set(range(1,plan['max_cycle']+1))
        early=statistics.mean(values[c] for c in range(1,6)) if all(c in values for c in range(1,6)) else None
        deltas=[values[c]-values[c-1] for c in range(2,plan['max_cycle']+1) if c in values and c-1 in values]
        trajectories.append({'seed':seed,'policy':policy,'mode':mode,'split':split,'metric':metric,
           'completed_cycles':len(values),'complete_trajectory':complete,'early_c1_to_c5_mean_percent':early,
           'c6_to_c7_delta_pp':values[7]-values[6] if 6 in values and 7 in values else None,
           'full_data_percent':values.get(9),'c5_minus_full_pp':values[5]-values[9] if 5 in values and 9 in values else None,
           'c6_minus_full_pp':values[6]-values[9] if 6 in values and 9 in values else None,
           'largest_downward_step_pp':max([0,*(-d for d in deltas)]) if complete else None,
           'downward_steps':sum(d<0 for d in deltas) if complete else None})
    fields=['seed','policy','cycle','mode','split','metric','value_percent','objects','samples','fraction','origin','task_id','epochs_completed','best_epoch','training_seconds']
    _csv(output_dir/'metrics.csv',records,fields)
    _csv(output_dir/'aggregate_metrics.csv',aggregates,['policy','mode','cycle','split','metric','mean_percent','sd_pp','completed_seeds','planned_seeds','all_seeds_complete'])
    _csv(output_dir/'paired_differences.csv',differences,['seed','policy','cycle','split','metric','comparison','delta_pp'])
    _csv(output_dir/'trajectory_summary.csv',trajectories,['seed','policy','mode','split','metric','completed_cycles','complete_trajectory','early_c1_to_c5_mean_percent','c6_to_c7_delta_pp','full_data_percent','c5_minus_full_pp','c6_minus_full_pp','largest_downward_step_pp','downward_steps'])
    _csv(output_dir/'new_fit_runtimes.csv',fit_times,['task_id','seed','policy','cycle','mode','shared_cycle_1','training_seconds','wall_seconds_this_attempt'])
    _csv(output_dir/'new_fit_per_class.csv',per_class,['seed','policy','cycle','mode','split','class_name','precision','recall','f1','support'])
    summary={'planned_new_fits':len(plan['tasks']),'completed_new_fits':len(fit_times),
             'existing_random_fits_reused':len(plan['references']),'incomplete':incomplete,
             'new_training_seconds_completed':sum(r['training_seconds'] for r in fit_times),
             'counts_shared_cycle_1_once':True,'interpretation':plan['interpretation'],
             'notes':['SD is across training seeds, not a confidence interval.','Partial aggregates identify completed seed counts; use paired differences for matched comparisons.',
                      'Descriptive early and endpoint comparisons do not select stopping budgets from test results.',
                      'Training times exclude some loading/evaluation overhead; no equal-total-compute claim.']}
    if plots:
        try: _plots(records,plan,output_dir)
        except ImportError as exc: summary['plot_warning']=str(exc)
    write_json(output_dir/'summary.json',summary)
    return summary


def _plots(records,plan,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':10})
    colors={'random_restart':'#222222','restart_from_l0':'#4c78a8','warm_start':'#9a6a30'}
    labels={'random_restart':'Random restart (existing)','restart_from_l0':'Restart from L0','warm_start':'Warm-start'}
    for split in ('validation','test'):
        for metric in METRICS:
            fig, axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
            for ax,policy in zip(axes.flat,plan['strategies']):
                for mode in ALL_MODES:
                    grouped=defaultdict(list)
                    for r in records:
                        if (r['split'],r['metric'],r['policy'],r['mode'])==(split,metric,policy,mode): grouped[r['cycle']].append(r['value_percent'])
                    x=sorted(grouped);y=[statistics.mean(grouped[c]) for c in x]
                    # Omit incomplete-seed means from comparison plots, avoiding mismatched groups.
                    x=[c for c in x if len(grouped[c])==len(plan['seeds'])];y=[statistics.mean(grouped[c]) for c in x]
                    if x:
                        sd=[statistics.stdev(grouped[c]) if len(grouped[c])>1 else 0 for c in x]
                        ax.plot(x,y,color=colors[mode],marker='o',ms=3,label=labels[mode])
                        ax.fill_between(x,[a-b for a,b in zip(y,sd)],[a+b for a,b in zip(y,sd)],color=colors[mode],alpha=.1)
                ax.set_title(policy.replace('_',' '));ax.spines[['top','right']].set_visible(False)
                ax.set_xticks(range(1,plan['max_cycle']+1));ax.grid(alpha=.15)
            for ax in axes[-1]:ax.set_xlabel('Acquisition cycle (saved batches)')
            for ax in axes[:,0]:ax.set_ylabel(metric.replace('_',' ')+' (%)')
            handles,legend_labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,legend_labels,loc='lower center',ncol=3,frameon=False)
            fig.suptitle(split.capitalize()+': means ± seed SD; complete seed groups only')
            fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(out/f'{split}_{metric}.pdf');fig.savefig(out/f'{split}_{metric}.png',dpi=180);plt.close(fig)
    for seed in plan['seeds']:
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
        for ax,policy in zip(axes.flat,plan['strategies']):
            for mode in ALL_MODES:
                rows=sorted([r for r in records if (r['seed'],r['policy'],r['mode'],r['split'],r['metric'])==(seed,policy,mode,'test','f1_macro')],key=lambda r:r['cycle'])
                ax.plot([r['cycle'] for r in rows],[r['value_percent'] for r in rows],color=colors[mode],marker='o',ms=3,label=labels[mode])
            ax.set_title(policy.replace('_',' '));ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
        for ax in axes[-1]:ax.set_xlabel('Acquisition cycle (saved batches)')
        for ax in axes[:,0]:ax.set_ylabel('Test macro-F1 (%)')
        handles,ls=axes.flat[0].get_legend_handles_labels();fig.legend(handles,ls,loc='lower center',ncol=3,frameon=False)
        fig.suptitle(f'Training seed {seed}: individual trajectories');fig.tight_layout(rect=(0,.07,1,.95))
        fig.savefig(out/f'test_f1_seed_{seed}.pdf');plt.close(fig)
