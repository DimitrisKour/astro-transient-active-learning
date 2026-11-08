"""Pure scheduler checks: no Delta access or GPU is needed."""
import csv
import importlib.util
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/research/historical_replay/task8c2/task8c2.py'
spec=importlib.util.spec_from_file_location('task8c2_script',SCRIPT)
task=importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def cases():
    return [{'policy':p,'seed':s,'key':f'{p}_s{s}','first_rejection':n}
            for (p,s),n in task.EXPECTED.items()]


def test_fixed_rejections_and_schedule():
    rows=cases()
    assert len(rows)==6
    assert sum(9-x['first_rejection'] for x in rows)==28
    plan=task.stage_plan(rows, list(range(6)))
    assert len(plan)==164
    assert len([x for x in plan if x['stage']=='base'])==28
    assert len([x for x in plan if x['stage']=='retry'])==136
    for idx in range(6):
        one=task.stage_plan(rows,[idx]);r=rows[idx]
        assert len(one)==4+5*(9-r['first_rejection'])
        assert one[0]['stage']=='retry' and one[0]['cycle']==r['first_rejection']
        assert [x['retry_number'] for x in one[:4]]==[1,2,3,4]
        assert one[-1]['cycle']==9 and one[-1]['retry_number']==4


def test_registry_storage_only_rewrite(tmp_path):
    ref=tmp_path/'ref';ref.mkdir()
    (ref/'registry.toml').write_text('[storage]\npersistent_root="/p"\ndata_root="/n"\n[datasets.one]\ndata_root="/d"\n[profiles.astro]\ndata_root="/e"\n')
    import tomllib
    result=tomllib.loads(task.routed_registry(ref,tmp_path/'hdd'))
    assert result['storage']=={'persistent_root':str(tmp_path/'hdd'),'data_root':str(tmp_path/'hdd')}
    assert result['datasets']['one']['data_root']=='/d'
    assert result['profiles']['astro']['data_root']=='/e'


def test_fail_closed_on_duplicate_first_rejection(tmp_path):
    path=tmp_path/'audit.csv'
    data=[{'suite_key':'task4','rule':'macro_tolerance_1pp','policy':r['policy'],'seed':r['seed'],
           'first_rejected_cycle':r['first_rejection'], 'last_accepted_cycle':r['first_rejection']-1,
           'fully_accepted_saved_trajectory':'false'} for r in cases()]
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    assert len(task.cases_from_audit(path))==6
    data[-1]=data[-2]
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    with pytest.raises(ValueError):task.cases_from_audit(path)


def test_walltime_validation():
    import re
    assert re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d','00:20:00')
    assert not re.fullmatch(r'\d{1,2}:[0-5]\d:[0-5]\d','20mins')


def test_slurm_dependencies_and_case_isolation(tmp_path,monkeypatch):
    """Only one case submitted, with exactly one attempt per distinct retry seed."""
    c=cases()
    job=tmp_path/'job';(job/'logs').mkdir(parents=True)
    plan={'cases':c, 'data':str(tmp_path)}
    states=[{'completed':r['first_rejection'],'decisions':r['first_rejection']-1,
             'rejected_cycles':[],'incomplete_decision_cycles':[r['first_rejection']]}
            for r in c]
    monkeypatch.setattr(task,'verify',lambda path,deep=False:plan)
    monkeypatch.setattr(task,'status',lambda path:{'cases':states})
    for index in (0,1):
        r=c[index]
        root=tmp_path/'runs'/task.SUITE/r['policy']/f"seed_{r['seed']}"/f"cycle_{r['first_rejection']:03d}"
        root.mkdir(parents=True)
        (root/'base_attempt.json').write_text('{"comparison":{"accepted":false},"reference_reused":true}')
    commands=[]; releases=[]
    def sbatch(command,text):
        commands.append(command)
        return str(20000000+len(commands))+'\n'
    def release(cmd,check):
        releases.append(cmd)
        return None
    monkeypatch.setattr(task.subprocess,'check_output',sbatch)
    monkeypatch.setattr(task.subprocess,'run',release)
    task.submit_gpu(job,'00:20:00',case_index=0)
    record=task.json_load(job/'gpu_submitted_case_0.json')
    assert len(record['jobs'])==4+5*(9-c[0]['first_rejection'])
    assert record['jobs'][0]['stage']=='retry' and record['jobs'][0]['retry_number']==1
    assert all('--hold' in cmd for cmd in commands)
    assert all(any(x.startswith('--dependency=afterok:') for x in cmd)
               for cmd in commands[1:])
    assert len(releases)==len(commands) and releases[0][2]==record['jobs'][-1]['job_id']
    with pytest.raises(RuntimeError,match='existing submission'):
        task.submit_gpu(job,'00:20:00',case_index=0)
    # Case 1 submission remains allowed even if case 0 has already progressed.
    states[0]['completed']=6
    task.submit_gpu(job,'00:20:00',case_index=1)
    assert (job/'gpu_submitted_case_1.json').exists()
