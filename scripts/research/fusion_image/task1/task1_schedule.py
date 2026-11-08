#!/usr/bin/env python3
"""Safely submit Task 1 as independent, afterok-chained, restart-aware SLURM jobs.

--submit-phase1 queues random and entropy ONLY. --submit-phase2 requires the
entire first phase completed (including sealed test evaluation). Partial sbatch
submission is recoverable: job IDs are appended immediately and reused.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from task1_closed_loop import (
    ROOT, REPO, SEEDS, CONFIGS, PHASE1, PHASE2, CYCLES,
    l0_dir, cycle_dir, verify_plan, _all_required_complete,
)

JOB_FIELDS=('kind','initialization','configuration','policy','seed','cycle','job_id','dependency')


def load_jobs(path:Path)->dict[tuple[str,str,str,str,int,int],dict]:
    entries={}
    if path.is_file():
        with path.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                if set(r)!=set(JOB_FIELDS) or not r['job_id'].split('_')[0].isdigit():
                    raise ValueError('Malformed Task 1 SLURM submission registry')
                key=(r['kind'],r['initialization'],r['configuration'],r['policy'],
                     int(r['seed']),int(r['cycle']))
                if key in entries:raise ValueError('Duplicate recorded submission identity')
                entries[key]=r
    return entries


def append_job(path:Path,row:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=JOB_FIELDS,delimiter='\t')
        if f.tell()==0:writer.writeheader()
        writer.writerow(row);f.flush();os.fsync(f.fileno())


def previous_completed(path:Path)->bool:
    return (path/'complete.json').is_file()


def submit(*,root:Path,mode:str,limit:int|None=None):
    plan=verify_plan(root,require_test_cache=True)
    # Fail before spending GPU allocations if the dedicated TEST cache is not
    # fully published or if its immutable arrays were corrupted. This uses
    # labels only for manifest/stage-coverage checks, not for model decisions.
    from al_pipeline.astronomy.fusion_image.test_fastcache_v1 import validate
    validate(root=Path(plan['test_cache']),verify_shards=0)
    if mode=='phase2' and not _all_required_complete(root,plan,PHASE1):
        raise RuntimeError('Cannot release advanced policies until every phase-1 trajectory completed')
    if mode not in ('l0','phase1','phase2'):raise ValueError('Unsupported phase')
    logfile=root/'slurm_jobs.tsv';old=load_jobs(logfile)
    scripts=Path(plan['repo'])/'scripts/research/fusion_image/task1/task1_cycle.sbatch'
    if not scripts.is_file():raise FileNotFoundError(scripts)
    if limit is not None and limit<1:raise ValueError('Limit must be positive')
    submitted=0

    def submit_one(*,kind:str,init:str,config:str,policy:str,seed:int,cycle:int,
                   parent_key:tuple|None,parent_path:Path|None):
        nonlocal submitted
        key=(kind,init,config,policy,seed,cycle)
        out=l0_dir(root,init,seed) if kind=='l0' else cycle_dir(root,config,policy,seed,cycle)
        if previous_completed(out):return
        if key in old:return
        if limit is not None and submitted>=limit:return
        dep=''
        if parent_path is not None and not previous_completed(parent_path):
            if parent_key not in old:
                # Earlier job may have failed scheduling (or quota hit).
                return
            dep=old[parent_key]['job_id']
        log=root/'logs';log.mkdir(parents=True,exist_ok=True)
        label=f't1_{init[0]}_{config}_{policy}_s{seed}_c{cycle}' if kind=='cycle' else f't1_l0_{init}_s{seed}'
        label=label[:90]
        command=['sbatch','--parsable','--job-name',label,
                 '--output',str(log/f'{label}_%j.out'),
                 '--error',str(log/f'{label}_%j.err'),
                 '--export',f'ALL,TASK1_ROOT={root},TASK1_KIND={kind},TASK1_INIT={init},'
                           f'TASK1_CONFIG={config},TASK1_POLICY={policy},TASK1_SEED={seed},TASK1_CYCLE={cycle}']
        if dep:command.append('--dependency=afterok:'+dep)
        command.append(str(scripts))
        try:
            proc=subprocess.run(command,check=True,text=True,capture_output=True)
        except subprocess.CalledProcessError as exc:
            print(f'[task1-schedule] sbatch refused {label}: {exc.stderr}',file=sys.stderr)
            print('Previously submitted job IDs were preserved; rerun this command after queue slots free.',file=sys.stderr)
            raise
        jobid=proc.stdout.strip().split(';')[0]
        if not jobid.isdigit():raise RuntimeError(f'Invalid sbatch response: {proc.stdout!r}')
        row={'kind':kind,'initialization':init,'configuration':config,'policy':policy,
             'seed':seed,'cycle':cycle,'job_id':jobid,'dependency':dep}
        append_job(logfile,row)
        old[key]=row;submitted+=1
        print(f'[task1-schedule] {label} job={jobid} dep={dep or "none"}',flush=True)

    if mode in ('l0','phase1'):
        for init in ('randomized','pretrained'):
            for seed in SEEDS:
                submit_one(kind='l0',init=init,config='',policy='',seed=seed,cycle=0,
                           parent_key=None,parent_path=None)

    if mode in ('phase1','phase2'):
        policies=PHASE1 if mode=='phase1' else PHASE2
        # Sequential chains per trajectory, with dependencies checked against
        # completion records (if parent done) OR submitted job IDs.
        for config in CONFIGS:
            init=CONFIGS[config]['initialization']
            for policy in policies:
                for seed in SEEDS:
                    for cycle in range(1,CYCLES+1):
                        if cycle==1:
                            parent=l0_dir(root,init,seed)
                            parent_key=('l0',init,'','',seed,0)
                        else:
                            parent=cycle_dir(root,config,policy,seed,cycle-1)
                            parent_key=('cycle',init,config,policy,seed,cycle-1)
                        submit_one(kind='cycle',init=init,config=config,policy=policy,
                                   seed=seed,cycle=cycle,parent_key=parent_key,
                                   parent_path=parent)
    print(json.dumps({'phase':mode,'new_jobs':submitted,'registry':str(logfile),
                      'registered_total':len(old),'limit':limit},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('l0','phase1','phase2','status'))
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.mode=='status':
        old=load_jobs(a.root/'slurm_jobs.tsv')
        print(json.dumps({'registered_slurm_jobs':len(old),
            'l0':sum(k[0]=='l0' for k in old),
            'phase1':sum(k[0]=='cycle' and k[3] in PHASE1 for k in old),
            'phase2':sum(k[0]=='cycle' and k[3] in PHASE2 for k in old)},indent=2))
    else:submit(root=a.root,mode=a.mode,limit=a.limit)

if __name__=='__main__':main()
