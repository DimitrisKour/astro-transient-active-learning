#!/usr/bin/env python3
"""Task18: AnchorAL historical replay, 3 seeds x 9 cycles, frozen Task11 parent.

Acquisition-only pilot uses the L0 checkpoint and confirmed L0 labels.
Training jobs are held and released explicitly. All scientific runs remain
separate from Tasks 4/11-17. External U0 labels never inform acquisition.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

POLICY='anchoral'
SEEDS=(135,246,357)
CYCLES=range(1,10)
SUITE='astrominn_task18_anchoral_v1'
PERSISTENT=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
JOB=PERSISTENT/'closed_loop'/SUITE
TASK11=PERSISTENT/'closed_loop/astrominn_task11_advanced_acquisition_v1'
BASE='astrominn_historical_replay_v1'
EXCLUDE=PERSISTENT/'runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and p.suffix in ('.py','.sh','.sbatch','.toml')}

def plan():
    return {'suite':SUITE,'method':'AnchorAL class-anchored cosine neighborhoods and entropy',
            'citation':'Lesci and Vlachos NAACL 2024; adaptation, not exact computational reproduction',
            'policies':[POLICY],'seeds':list(SEEDS),'cycles':9,
            'new_fits':27,'batch':473,'last_batch':469,'complete_pool':True,
            'source_parent':'manifest-verified Task11 source',
            'bootstrap':'reuse original 3 seed-specific L0 checkpoints',
            'split_seed':135,'initial_fraction':.10,'acquisition_fraction':.10,
            'training_initialization':'warm_start','max_epochs':50,'patience':10,
            'training_batch_size':512,'validation_batch_size':2048,
            'learning_rate':1e-4,'validation_only':True,
            'anchor_rule':{'anchors_per_class':5,'neighbors_per_anchor':200,
                           'max_subpool':1419,'similarity':'cosine',
                           'sampling':'seeded classwise cosine kmeans++',
                           'selection':'top entropy within similarity-ranked anchor subpool'},
            'limitations':'full U inference precedes retrieval; not an ANN acceleration benchmark'}

def prepare(overlay):
    from task18_patch import patch_source
    if (JOB/'frozen_manifest.json').exists():
        verify();print('Task18 already prepared');return
    if (JOB/'source').exists() or (JOB/'registry.toml').exists():
        raise RuntimeError('Partial Task18 preparation; inspect first')
    parent_manifest=TASK11/'frozen_manifest.json'
    if load(parent_manifest)!=hashes(TASK11/'source'):
        raise RuntimeError('Task11 frozen parent manifest mismatch')
    for seed in SEEDS:
        cp=DATA/'runs'/BASE/'shared'/f'seed_{seed}/bootstrap_training/best_vit_tower.pth'
        if not cp.is_file():raise FileNotFoundError(cp)
    if not EXCLUDE.is_file():raise FileNotFoundError(EXCLUDE)
    overlay=Path(overlay).resolve()
    JOB.mkdir(parents=True,exist_ok=True);(JOB/'logs').mkdir(exist_ok=True)
    stage=JOB/'source.tmp'
    if stage.exists():raise RuntimeError('Stale Task18 staging dir')
    try:
        shutil.copytree(TASK11/'source',stage,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for p in (stage/'scripts/adaptive').glob('task11*'):p.unlink()
        (stage/'scripts/research/historical_replay/task18').mkdir(parents=True, exist_ok=True)
        for p in (overlay/'scripts/research/historical_replay/task18').glob('task18*'):
            shutil.copy2(p,stage/'scripts/research/historical_replay/task18'/p.name)
        changed=patch_source(stage,overlay)
        for f in stage.rglob('*.py'):
            ast.parse(f.read_text(),filename=str(f))
        stage.rename(JOB/'source')
        (JOB/'registry.toml').write_text((TASK11/'registry.toml').read_text())
        dump(JOB/'scientific_plan.json',plan())
        dump(JOB/'source_parent.json',{'parent_manifest_sha256':sha(parent_manifest),
                                      'patched_files':changed})
        dump(JOB/'frozen_manifest.json',hashes(JOB/'source'))
    except BaseException:
        if stage.exists():shutil.rmtree(stage)
        raise
    verify()

def verify(job=JOB):
    job=Path(job)
    if load(job/'scientific_plan.json')!=plan():
        raise RuntimeError('Task18 scientific plan mismatch')
    if load(job/'frozen_manifest.json')!=hashes(job/'source'):
        raise RuntimeError('Task18 source modified after freeze')
    print('Task18 frozen source verified')

def args(seed):
    if seed not in SEEDS:raise ValueError('Invalid seed')
    return ['--registry',str(JOB/'registry.toml'),'--suite-id',SUITE,
            '--bootstrap-source-suite-id',BASE,'--require-existing-bootstrap',
            '--training-initialization','warm_start','--strategies',POLICY,
            '--seed',str(seed),'--split-seed','135',
            '--initial-fraction','.10','--minimum-initial-per-class','5',
            '--acquisition-fraction','.10','--max-cycles','9','--complete-pool',
            '--max-epochs','50','--patience','10','--batch-size','512',
            '--validation-batch-size','2048','--learning-rate','0.0001',
            '--train-workers','0','--validation-workers','0',
            '--inference-workers','0','--evaluation-workers','0',
            '--assert-excluded-manifest',str(EXCLUDE)]

def runner(seed,additional):
    env=dict(os.environ,PYTHONPATH=str(JOB/'source/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    subprocess.run([sys.executable,str(JOB/'source/scripts/run_astrominn_closed_loop.py'),
                    *args(seed),*additional],env=env,check=True)

def preflight():
    verify()
    for seed in SEEDS:runner(seed,[])
    print('Task18 preflight completed without GPU training')

def pilot(seed=135):
    verify()
    if seed not in SEEDS:raise ValueError('Invalid seed')
    result=JOB/'pilot'/f'seed_{seed}/result.json'
    if result.exists():print(result.read_text());return
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (
        DEFAULT_LEGACY_CODE_DIR,predict_astrominn_vit_pool)
    from al_pipeline.astronomy.analysis.task11_labeled_features import predict_confirmed_objects
    from al_pipeline.core.scoring.entropy import EntropyScorer
    from al_pipeline.core.advanced_acquisition.anchoral import AnchorALInput,AnchorALSelector
    from al_pipeline.core.pool import Pool
    import csv
    registry=DatasetRegistry.from_toml(JOB/'registry.toml')
    resolved=registry.resolve_profile('astrominn_ztf')
    ds=resolved.dataset
    root=PERSISTENT/'runs'/SUITE/'shared/historical_replay_split'
    ref=PERSISTENT/'runs/astrominn_closed_loop_warm_v1/shared/historical_replay_split'
    ids=lambda f: {r[ds.object_id_column] for r in csv.DictReader(open(f,newline=''))}
    if ids(root/'initial_labeled_manifest.csv')!=ids(ref/'initial_labeled_manifest.csv'):
        raise ValueError('AnchorAL L0 differs from reference')
    original=CandidatePoolManifest.from_csv(ref/'candidate_pool/candidate_pool.csv')
    poolpath=root/'candidate_pool'
    candidate=CandidatePoolManifest.from_csv(poolpath/'candidate_pool.csv')
    if tuple(candidate.object_ids)!=tuple(original.object_ids):
        raise ValueError('AnchorAL U0 differs from reference')
    cp=DATA/'runs'/BASE/'shared'/f'seed_{seed}/bootstrap_training/best_vit_tower.pth'
    artifact=Path(resolved.profile.options['checkpoint']).parent
    t0=time.perf_counter()
    pred=predict_astrominn_vit_pool(pool_path=poolpath,object_ids=candidate.object_ids,
                device='cuda',batch_size=64,num_workers=0,
                legacy_code_dir=DEFAULT_LEGACY_CODE_DIR,artifact_dir=artifact,vit_checkpoint=cp)
    confirmed=predict_confirmed_objects(
        manifest_path=root/'initial_labeled_manifest.csv',dataset=ds,
        checkpoint_path=cp,artifact_dir=artifact,device='cuda',batch_size=64,
        num_workers=0,legacy_code_dir=DEFAULT_LEGACY_CODE_DIR)
    if pred.embeddings is None or confirmed.embeddings is None:
        raise ValueError('Missing object-level embeddings')
    known={}
    with (root/'initial_labeled_manifest.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            lab=str(row[ds.label_column]);oid=str(row[ds.object_id_column])
            if oid in known and lab!=known[oid]:raise ValueError('Conflicting L0 labels')
            known[oid]=lab
    from al_pipeline.astronomy.astrominn.training import MODEL_CLASS_NAMES,LABEL_TO_INDEX
    canonical={label:MODEL_CLASS_NAMES[ix] for label,ix in LABEL_TO_INDEX.items()}
    inp=AnchorALInput(candidate_ids=tuple(pred.object_ids),
        candidate_embeddings=pred.embeddings,labeled_ids=tuple(confirmed.object_ids),
        labeled_embeddings=confirmed.embeddings,
        labeled_labels=tuple(canonical[known[x]] for x in confirmed.object_ids),
        class_names=tuple(MODEL_CLASS_NAMES),seed=seed,cycle=1)
    scores=EntropyScorer().score(pred.predictions)
    selection=AnchorALSelector(inp).select(ids=pred.object_ids,scores=scores,k=473,
                                          pool=Pool.from_ids(pred.object_ids))
    if len(set(selection.ids))!=473:raise RuntimeError('Pilot selection mismatch')
    reference=PERSISTENT/'runs/astrominn_closed_loop_warm_v1/entropy'/f'seed_{seed}/cycle_001/selection.json'
    overlap=len(set(selection.ids)&set(load(reference)['ids']))
    dump(result,{'seed':seed,'selected_ids':selection.ids,
        'budget':473,'unique':True,'entropy_overlap':overlap,
        'elapsed_seconds':time.perf_counter()-t0,'selection_metadata':selection.metadata,
        'checkpoint_sha256':sha(cp),'no_hidden_labels_for_selection':True})
    print(json.dumps({'seed':seed,'budget':473,'entropy_overlap':overlap,
                      'elapsed_seconds':load(result)['elapsed_seconds']},indent=2))

def submit(walltime):
    verify()
    if not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d',walltime) or walltime=='00:00:00':
        raise ValueError('Invalid walltime')
    ledger=JOB/'submitted.json'
    data=load(ledger) if ledger.exists() else {'jobs':[],'state':'held_partial'}
    existing={(x['seed'],x['cycle']):x for x in data['jobs']}
    for seed in SEEDS:
        for cycle in CYCLES:
            if (seed,cycle) in existing:continue
            previous=existing.get((seed,cycle-1))
            if cycle>1 and not previous:raise RuntimeError('Prior cycle missing from ledger')
            cmd=['sbatch','--parsable','--hold','--time',walltime,
                 '--job-name',f'T18_anchor_{seed}_{cycle}',
                 '--output',str(JOB/'logs/%x_%j.out'),
                 '--error',str(JOB/'logs/%x_%j.err')]
            if previous:cmd+=['--dependency',f'afterok:{previous["job_id"]}']
            cmd += [str(JOB/'source/scripts/research/historical_replay/task18/task18_worker.sbatch'),
                    str(JOB),'job',str(seed),str(cycle)]
            out=subprocess.check_output(cmd,text=True).strip().split(';')[0]
            if not out.isdigit():raise RuntimeError('Unexpected Slurm job ID '+out)
            item={'seed':seed,'cycle':cycle,'job_id':out,
                  'parent':previous['job_id'] if previous else None,'released':False}
            data['jobs'].append(item);existing[(seed,cycle)]=item;dump(ledger,data)
            print('HELD',out,seed,cycle)
    data['state']='held' if len(data['jobs'])==27 else 'held_partial'
    dump(ledger,data)
    print(f'Task18: {len(data["jobs"])}/27 jobs recorded HELD')

def release(all_jobs=False,seed=None,cycle=None):
    verify()
    data=load(JOB/'submitted.json')
    if all_jobs:
        if seed is not None or cycle is not None:raise ValueError('Cannot combine --all and seed/cycle')
        targets=list(reversed(data['jobs']))
    else:
        if seed not in SEEDS or cycle not in CYCLES:raise ValueError('Specify seed and cycle')
        targets=[x for x in data['jobs'] if x['seed']==seed and x['cycle']==cycle]
    for row in targets:
        if row['released']:continue
        subprocess.run(['scontrol','release',row['job_id']],check=True)
        row['released']=True
        data['state']='partially_released';dump(JOB/'submitted.json',data)
        print('Released',row['job_id'])
    if len(data['jobs'])==27 and all(x['released'] for x in data['jobs']):
        data['state']='released_all';dump(JOB/'submitted.json',data)

def status():
    rows=[]
    for seed in SEEDS:
        path=PERSISTENT/'runs'/SUITE/POLICY/f'seed_{seed}'
        state=path/'run_state.json'
        done=int(load(state).get('last_completed_cycle',0)) if state.is_file() else 0
        rows.append({'seed':seed,'completed_cycles':done,'complete':done==9})
    print(json.dumps({'trajectories':rows,
        'complete':sum(x['complete'] for x in rows),'total':3},indent=2))

def collect(destination):
    from zipfile import ZipFile,ZIP_DEFLATED
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    paths=[JOB/'scientific_plan.json',JOB/'source_parent.json',
           JOB/'frozen_manifest.json',JOB/'submitted.json',JOB/'pilot/seed_135/result.json']
    for seed in SEEDS:
        rel=Path(SUITE)/POLICY/f'seed_{seed}'
        run=PERSISTENT/'runs'/rel
        art=DATA/'runs'/rel
        paths += [run/f for f in ('run_state.json','experiment.json','evaluation_history.json')]
        for i in range(10):
            cyc=f'cycle_{i:03d}'
            if i:
                paths += [run/cyc/f for f in ('selection.json','revealed_labels.json','cycle_summary.json')]
            paths += [art/cyc/'evaluation'/f for f in (
                'evaluation_summary.json','validation_per_class_metrics.json',
                'validation_object_confusion.csv','validation_sample_confusion.csv')]
    with ZipFile(destination,'x',ZIP_DEFLATED) as z:
        for p in dict.fromkeys(paths):
            if not p.is_file():continue
            if p.is_relative_to(JOB):name='control/'+str(p.relative_to(JOB))
            elif p.is_relative_to(PERSISTENT):name='persistent/'+str(p.relative_to(PERSISTENT))
            elif p.is_relative_to(DATA):name='artifacts/'+str(p.relative_to(DATA))
            else:raise RuntimeError('Unexpected archive path')
            z.write(p,name)
    print('Collected',destination)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('plan','prepare','verify','preflight','pilot',
          'submit','release','status','collect','job'))
    p.add_argument('seed_pos',nargs='?',type=int)
    p.add_argument('cycle_pos',nargs='?',type=int)
    p.add_argument('--seed',type=int)
    p.add_argument('--cycle',type=int)
    p.add_argument('--all',action='store_true')
    p.add_argument('--walltime',default='00:20:00')
    p.add_argument('--overlay',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--job-dir',type=Path,default=JOB)
    p.add_argument('--destination',type=Path,default=Path('/u/dkourtesis/task18_anchoral_results.zip'))
    a=p.parse_args()
    if a.command=='plan':print(json.dumps(plan(),indent=2))
    elif a.command=='prepare':prepare(a.overlay)
    elif a.command=='verify':verify(a.job_dir)
    elif a.command=='preflight':preflight()
    elif a.command=='pilot':pilot(a.seed if a.seed is not None else 135)
    elif a.command=='submit':submit(a.walltime)
    elif a.command=='release':release(a.all,a.seed,a.cycle)
    elif a.command=='status':status()
    elif a.command=='collect':collect(a.destination)
    elif a.command=='job':
        verify(a.job_dir)
        if a.seed_pos not in SEEDS or a.cycle_pos not in CYCLES:raise ValueError('Invalid job case')
        runner(a.seed_pos,['--device','cuda','--cycle',str(a.cycle_pos),'--run'])

if __name__=='__main__':main()
