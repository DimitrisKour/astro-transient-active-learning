"""Run the real submission script against fake Slurm and a tiny prepared plan."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]

def executable(path,content):path.write_text(content);path.chmod(0o755)

def setup(tmp_path):
    repo=tmp_path/'repo';scripts=repo/'scripts/adaptive';scripts.mkdir(parents=True)
    launcher=scripts/'submit_initialization_compare.sh';shutil.copyfile(ROOT/'scripts/adaptive/submit_initialization_compare.sh',launcher)
    commands=tmp_path/'bin';commands.mkdir();prepare=tmp_path/'prepare.py'
    prepare.write_text('''import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--persistent-root');p.add_argument('--comparison-id');p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--max-cycle',type=int)
a,_=p.parse_known_args();root=Path(a.persistent_root)/'initialization_compare'/a.comparison_id
root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True);path=root/'plan.json'
if not path.exists():
 tasks=[]
 for seed in a.seeds:
  for cycle in range(1,a.max_cycle+1):
   for lane in range(4 if cycle==1 else 8):
    i=len(tasks);tasks.append({'index':i,'task_id':f'fit_{i:03d}','seed':seed,'cycle':cycle})
 payload={'schema_version':1,'kind':'initialization_compare','metadata_root':str(root),'tasks':tasks,'seeds':a.seeds,'max_cycle':a.max_cycle}
 path.write_text(json.dumps(payload,sort_keys=True));(root/'plan.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
 frozen=root/'source/scripts/adaptive';frozen.mkdir(parents=True);(frozen/'initialization_cycle.sbatch').write_text('# frozen')
''')
    executable(commands/'python',f'''#!/bin/bash
if [[ "$1" == *prepare_initialization_compare.py ]]; then
 shift
 exec {sys.executable} {prepare} "$@"
fi
exec {sys.executable} "$@"
''')
    executable(commands/'sbatch',f'''#!{sys.executable}
import json,os,sys
from pathlib import Path
p=Path(os.environ['MOCK_COUNTER']);n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n))
with open(os.environ['MOCK_LOG'],'a') as out:out.write(json.dumps(['sbatch',*sys.argv[1:]])+'\\n')
if n==int(os.environ.get('MOCK_FAIL_AT','0')):sys.exit(1)
print(1000+n)
''')
    for command in ('scontrol','scancel'):
        executable(commands/command,f'''#!{sys.executable}
import json,os,sys
with open(os.environ['MOCK_LOG'],'a') as out:out.write(json.dumps(['{command}',*sys.argv[1:]])+'\\n')
if '{command}'=='scontrol' and sys.argv[-1]==os.environ.get('MOCK_RELEASE_FAIL',''):sys.exit(1)
''')
    executable(commands/'squeue','''#!/bin/bash
if [[ "$*" == *--noheader* ]]; then
 [[ -z "${MOCK_ACTIVE:-}" ]] || echo "$MOCK_ACTIVE"
fi
exit 0
''')
    env={k:v for k,v in os.environ.items() if k not in {'BASH_ENV','ENV'} and not k.startswith('BASH_FUNC_')}
    env.update(INIT_SKIP_ACTIVATE='1',INIT_ID='test_init',INIT_REGISTRY=str(tmp_path/'registry.toml'),
               INIT_PERSISTENT_ROOT=str(tmp_path/'persistent'),INIT_DATA_ROOT=str(tmp_path/'heavy'),
               PATH=f"{commands}:{os.environ['PATH']}",PYTHONPATH=f"{ROOT/'src'}:{os.environ.get('PYTHONPATH','')}",
               MOCK_COUNTER=str(tmp_path/'counter'),MOCK_LOG=str(tmp_path/'calls.jsonl'))
    return launcher,env,tmp_path/'persistent/initialization_compare/test_init'

def invoke(launcher,env,*args):return subprocess.run(['bash',str(launcher),*args],env=env,text=True,capture_output=True)
def calls(env):
    p=Path(env['MOCK_LOG']);return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

def test_27_held_arrays_short_times_seedwise_dependencies_and_reverse_release(tmp_path):
    launcher,env,meta=setup(tmp_path);result=invoke(launcher,env);assert result.returncode==0,result.stderr
    events=calls(env);sub=[r for r in events if r[0]=='sbatch'];assert len(sub)==27
    for n,row in enumerate(sub):
        cycle=n%9+1
        assert '--hold' in row and '--kill-on-invalid-dep=yes' in row
        time='00:10:00' if cycle<=2 else ('00:15:00' if cycle<=5 else '00:20:00')
        assert '--time='+time in row
        array=next(s for s in row if s.startswith('--array='));indices=array.split('=')[1].split('%')[0].split(',')
        assert len(indices)==(4 if cycle==1 else 8) and array.endswith('%1')
        if cycle==1:assert not any(s.startswith('--dependency=') for s in row)
        else:assert '--dependency=afterok:'+str(1000+n) in row
        assert str(meta/'source') in row[-1]
    assert [r[-1] for r in events if r[0]=='scontrol']==[str(i) for i in range(1027,1000,-1)]

def test_prepare_only_no_scheduler_actions(tmp_path):
    launcher,env,meta=setup(tmp_path);assert invoke(launcher,env,'--prepare-only').returncode==0
    assert calls(env)==[] and (meta/'plan.json').exists() and not (meta/'submissions.tsv').exists()

def test_duplicate_submission_refused(tmp_path):
    launcher,env,meta=setup(tmp_path);assert invoke(launcher,env).returncode==0;before=calls(env)
    result=invoke(launcher,env);assert result.returncode!=0 and 'Submission record exists' in result.stderr
    assert before==calls(env)

def test_failed_submission_cancels_only_new_jobs(tmp_path):
    launcher,env,meta=setup(tmp_path);env['MOCK_FAIL_AT']='2';result=invoke(launcher,env)
    assert result.returncode!=0
    assert [r for r in calls(env) if r[0]=='scancel']==[['scancel','1001']]
    assert not [r for r in calls(env) if r[0]=='scontrol']

def test_release_failure_cancels_all_new_jobs(tmp_path):
    launcher,env,meta=setup(tmp_path);env['MOCK_RELEASE_FAIL']='1027';assert invoke(launcher,env).returncode!=0
    assert [r for r in calls(env) if r[0]=='scancel']==[['scancel',*[str(i) for i in range(1001,1028)]]]

def test_retry_refuses_active_jobs(tmp_path):
    launcher,env,meta=setup(tmp_path);assert invoke(launcher,env).returncode==0;before=calls(env);env['MOCK_ACTIVE']='1001_0'
    result=invoke(launcher,env,'--retry-incomplete');assert result.returncode!=0 and 'active/pending' in result.stderr
    assert calls(env)==before

def test_retry_skips_completed_cycle_and_keeps_next_dependencies(tmp_path):
    launcher,env,meta=setup(tmp_path);env['INIT_SEEDS']='135';env['INIT_MAX_CYCLE']='3'
    assert invoke(launcher,env).returncode==0
    plan=json.loads((meta/'plan.json').read_text());digest=hashlib.sha256((meta/'plan.json').read_bytes()).hexdigest()
    checkpoint=tmp_path/'done.pth';checkpoint.write_bytes(b'done');h=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for t in plan['tasks'][:4]:
        directory=meta/'tasks'/t['task_id'];directory.mkdir(parents=True)
        (directory/'result.json').write_text(json.dumps({'task':t,'plan_sha256':digest,'checkpoint_path':str(checkpoint),'checkpoint_sha256':h}))
    before=len(calls(env));result=invoke(launcher,env,'--retry-incomplete');assert result.returncode==0,result.stderr
    sub=[r for r in calls(env)[before:] if r[0]=='sbatch'];assert len(sub)==2
    assert not any(s.startswith('--dependency=') for s in sub[0])
    assert '--dependency=afterok:1004' in sub[1]
    assert '--array=4,5,6,7,8,9,10,11%1' in sub[0]

def test_invalid_walltime_fails_before_preparation(tmp_path):
    launcher,env,meta=setup(tmp_path);assert invoke(launcher,env,'00:00:00').returncode!=0
    assert calls(env)==[] and not meta.exists()
