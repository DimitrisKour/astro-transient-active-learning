#!/usr/bin/env python3
"""Safe source freeze, deterministic split, and contract audit for Task6C."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from task6c_build import MODES, build_snapshot

PERSISTENT=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
DATA=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive')
SEEDS=(135,246,357)
POLICIES=('entropy','entropy_balanced_explore')
BASE='astrominn_historical_replay_v1'


def digest_bytes(value:bytes)->str: return hashlib.sha256(value).hexdigest()
def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def job_dir(mode:str)->Path:
    return PERSISTENT/'closed_loop'/MODES[mode]['suite']

def freeze(repo:Path,mode:str)->Path:
    cfg=MODES[mode];dest=job_dir(mode)
    registry=PERSISTENT/'closed_loop'/'astrominn_closed_loop_warm_v1'/'registry.toml'
    if not registry.is_file(): raise FileNotFoundError('Task4 routed registry missing: '+str(registry))
    runner,training=build_snapshot(repo,mode)
    files=sorted((repo/'src'/'al_pipeline').rglob('*.py'))
    files += sorted((repo/'scripts'/'adaptive').glob('task6c_*.py'))
    files += sorted((repo/'scripts'/'adaptive').glob('task6c_*.sh'))
    files += sorted((repo/'scripts'/'adaptive').glob('task6c_*.sbatch'))
    files += [repo/'scripts'/'adaptive'/'task6c.py']
    if not files or any(not x.is_file() for x in files): raise FileNotFoundError('Incomplete Task6C source')
    src={str(p.relative_to(repo)):p.read_bytes() for p in files}
    src['src/al_pipeline/astronomy/astrominn/training.py']=training.encode()
    src['scripts/run_task6c_closed_loop.py']=runner.encode()
    sha={name:digest_bytes(value) for name,value in sorted(src.items())}
    reg=registry.read_bytes()
    contract={
        'schema':'task6c_equal_object_v1','selection':mode,'suite_id':cfg['suite'],
        'reference_suite':cfg['reference'],'bootstrap_suite':cfg['bootstrap'],
        'checkpoint_selection':cfg['checkpoint'],'class_weight_exponent':0.85,
        'l0_object_weighting':'per_sample','cycle_object_weighting':'equal_object',
        'historical_objects_untouched':True,'split_seed':135,
        'policies':list(POLICIES),'seeds':list(SEEDS),'max_cycles':9,
        'acquisition_fraction':0.10,'warm_start':True,
        'early_stopping':'weighted_validation_loss',
        'registry_sha256':digest_bytes(reg),
    }
    dest.mkdir(parents=True,exist_ok=True);(dest/'logs').mkdir(exist_ok=True)
    snapshot=dest/'source_snapshot.json'
    if snapshot.exists():
        if json.loads(snapshot.read_text())!=sha: raise ValueError('Task6C repo changed after source freeze; use new suite ID')
        if json.loads((dest/'contract.json').read_text())!=contract: raise ValueError('Task6C frozen contract changed')
        if (dest/'registry.toml').read_bytes()!=reg: raise ValueError('Task6C frozen registry changed')
        verify(mode,dest)
        return dest
    if (dest/'source').exists() or (dest/'contract.json').exists() or (dest/'registry.toml').exists():
        raise ValueError('Partial Task6C source freeze: inspect, do not overwrite')
    (dest/'source').mkdir()
    for name,data in src.items():
        out=dest/'source'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    (dest/'registry.toml').write_bytes(reg)
    (dest/'contract.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
    snapshot.write_text(json.dumps(sha,indent=2,sort_keys=True)+'\n')
    verify(mode,dest)
    return dest

def verify(mode:str,dest:Path|None=None)->None:
    cfg=MODES[mode];dest=dest or job_dir(mode)
    record=json.loads((dest/'contract.json').read_text())
    if (record.get('selection')!=mode or record.get('suite_id')!=cfg['suite'] or
        record.get('checkpoint_selection')!=cfg['checkpoint'] or
        record.get('cycle_object_weighting')!='equal_object' or
        record.get('bootstrap_suite')!=cfg['bootstrap'] or
        record.get('class_weight_exponent')!=.85): raise ValueError('Task6C frozen scientific contract mismatch')
    manifest=json.loads((dest/'source_snapshot.json').read_text())
    for path,sha in manifest.items():
        f=dest/'source'/path
        if not f.is_file() or digest(f)!=sha: raise ValueError('Frozen source corrupted: '+str(f))
    if digest(dest/'registry.toml')!=record['registry_sha256']:
        raise ValueError('Frozen registry changed')

def normal(path:Path,ignored=frozenset()):
    with path.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if not r.fieldnames: raise ValueError('Missing CSV header: '+str(path))
        return sorted(tuple(sorted((k,v) for k,v in row.items() if k not in ignored)) for row in r)

def check_splits(mode:str)->None:
    p=PERSISTENT/'runs';base=p/BASE/'shared'/'historical_replay_split';new=p/MODES[mode]['suite']/'shared'/'historical_replay_split'
    b=json.loads((base/'split_summary.json').read_text());n=json.loads((new/'split_summary.json').read_text())
    keys=('split_seed','initial_fraction','minimum_initial_per_class','historical_objects','historical_samples',
          'initial_objects','initial_samples','candidate_objects','candidate_samples','maximum_alert',
          'historical_object_label_counts','initial_object_label_counts','candidate_object_label_counts',
          'initial_sample_label_counts','candidate_sample_label_counts','overlap_checks')
    for k in keys:
        if b.get(k)!=n.get(k):raise ValueError('Task6C split mismatch: '+k)
    for file in ('initial_labeled_manifest.csv','hidden_labels.csv'):
        if normal(base/file)!=normal(new/file):raise ValueError('Task6C split mismatch: '+file)
    pool='candidate_pool/candidate_pool.csv'
    if normal(base/pool,{'pool_id'})!=normal(new/pool,{'pool_id'}):raise ValueError('Task6C pool mismatch')
    print('Task6C split identical to Experiment B: 4253 U0 objects')

def _environment()->dict[str,str]:
    e=dict(os.environ)
    prefix=Path(e.get('CONDA_PREFIX',str(Path(sys.executable).parents[1])))
    if (prefix/'lib/libstdc++.so.6').is_file():
        e['LD_PRELOAD']=str(prefix/'lib/libstdc++.so.6')+(':'+e['LD_PRELOAD'] if e.get('LD_PRELOAD') else '')
        e['LD_LIBRARY_PATH']=str(prefix/'lib')+(':'+e['LD_LIBRARY_PATH'] if e.get('LD_LIBRARY_PATH') else '')
    e.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return e

def materialize(mode:str)->None:
    dest=job_dir(mode)
    env=_environment();env['PYTHONPATH']=str(dest/'source'/'src')
    cmd=[sys.executable,str(dest/'source/scripts/adaptive/task6c_split.py'),
         '--registry',str(dest/'registry.toml'),'--persistent',str(PERSISTENT),
         '--suite',MODES[mode]['suite']]
    subprocess.run(cmd,check=True,env=env)
    check_splits(mode)

def bootstrap_check(mode:str,strict:bool=True)->bool:
    cfg=MODES[mode]
    all_ok=True
    for seed in SEEDS:
        folder=DATA/'runs'/cfg['bootstrap']/'shared'/f'seed_{seed}'/'bootstrap_training'
        f=folder/'training_summary.json';c=folder/'best_vit_tower.pth'
        if not f.is_file() or not c.is_file():
            all_ok=False
            if strict:raise FileNotFoundError(f'Task6C {mode} reference L0 not yet available: {folder}')
            continue
        r=json.loads(f.read_text());p=r.get('policy',{})
        if (r.get('seed')!=seed or r.get('cycle')!=0 or
            r.get('run_id')!=f'{cfg["bootstrap"]}_bootstrap_seed{seed}' or
            p.get('class_weight_exponent')!=0.85 or
            p.get('class_weight_source')!='training_manifest' or
            p.get('new_object_weighting')!='per_sample' or
            r.get('initial_checkpoint') is not None or
            r.get('initialization')!='random_vit' or
            Path(r.get('checkpoint_path',''))!=c):
            raise ValueError('Incompatible L0 training provenance: '+str(f))
        if mode=='macro_f1' and (p.get('checkpoint_selection')!='object_macro_f1_best'
            or r.get('checkpoint_selection')!='object_macro_f1_best'
            or not (folder/'best_loss_vit_tower.pth').is_file()):
            raise ValueError('Task6B L0 is not macro-F1-selected: '+str(f))
    return all_ok
