import csv
import importlib.util
from pathlib import Path
import json
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/research/historical_replay/task8c1/task8c1.py'
spec=importlib.util.spec_from_file_location('task8c1_script',SCRIPT)
task=importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def fake_audit(path):
    rows=[]
    for (policy,seed),cycle in task.EXPECTED.items():
        rows.append({'suite_key':'task4','rule':'macro_tolerance_1pp','policy':policy,
         'seed':seed,'first_rejected_cycle':cycle,'last_accepted_cycle':cycle-1,
         'fully_accepted_saved_trajectory':'false'})
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def test_cases_match_six_first_rejections(tmp_path):
    path=tmp_path/'first_rejections.csv';fake_audit(path)
    a=task.cases_from_audit(path)
    assert len(a)==6 and sum(9-r['first_rejection'] for r in a)==28
    assert a[0]['first_rejection']==4
    assert a[-1]['first_rejection']==6


def test_duplicate_rejection_refused(tmp_path):
    path=tmp_path/'first_rejections.csv';fake_audit(path)
    lines=path.read_text().splitlines();lines[-1]=lines[-2];path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='duplicate|Missing'):
        task.cases_from_audit(path)


def test_incorrect_first_rejection_fails_closed(tmp_path):
    path=tmp_path/'first_rejections.csv';fake_audit(path)
    lines=path.read_text().replace('entropy,135,4,3','entropy,135,2,1')
    path.write_text(lines)
    with pytest.raises(ValueError,match='disagrees'):
        task.cases_from_audit(path)


def test_routed_registry_only_storage_changes(tmp_path):
    reference=tmp_path/'ref';reference.mkdir()
    (reference/'registry.toml').write_text('''[storage]\npersistent_root = "/original"\ndata_root = "/nvme"\n[datasets.foo]\npath = "/raw/data"\n''')
    text=task.routed_registry(reference,tmp_path/'hdd')
    assert text.count(str(tmp_path/'hdd'))==2
    assert 'path = "/raw/data"' in text


def test_routed_registry_refuses_missing_root(tmp_path):
    (tmp_path/'registry.toml').write_text('[storage]\nother = "test"\n')
    with pytest.raises(ValueError,match='field'):
        task.routed_registry(tmp_path,tmp_path/'hdd')


def test_status_no_unapproved_training(tmp_path):
    plan={'suite':task.SUITE,'cases':[{'seed':135,'policy':'entropy','key':'entropy_s135'}]}
    job=tmp_path/'job';job.mkdir();(job/'plan.json').write_text(json.dumps(plan))
    # status must fail, rather than silently reporting a partly created experiment
    with pytest.raises(Exception):task.status(job)


def test_routed_registry_preserves_repeated_dataset_data_roots(tmp_path):
    """Regression: Task4 registry contains more than one data_root key."""
    reference=tmp_path/'reference';reference.mkdir()
    original='[storage]\npersistent_root = "/projects/bcrv/dkourtesis/applecider_adaptive"\ndata_root = "/work/hdd/bcrv/dkourtesis/applecider_adaptive"\n\n[datasets.first]\ndata_root = "/absolute/ztf-data"\ncheckpoint = "/absolute/expert.pth"\n\n[datasets.second]\ndata_root = "/absolute/rubin-data"\n\n[profiles.third]\ndata_root = "/absolute/extra-data"\n'
    (reference/'registry.toml').write_text(original,encoding='utf-8')
    routed=task.routed_registry(reference,tmp_path/'hdd')
    import tomllib
    before=tomllib.loads(original);after=tomllib.loads(routed)
    assert after['storage']['persistent_root']==str(tmp_path/'hdd')
    assert after['storage']['data_root']==str(tmp_path/'hdd')
    assert after['datasets']==before['datasets']
    assert after['profiles']==before['profiles']
    assert task.routed_registry(reference,tmp_path/'hdd')==routed
    assert (reference/'registry.toml').read_text()==original


def test_routed_registry_with_indented_storage_and_comments(tmp_path):
    reference=tmp_path/'ref';reference.mkdir()
    original='  [storage]  # storage roots\npersistent_root  = "/original" # keep\n  data_root  = "/nvme"\n  [datasets.foo]\ndata_root = "/do-not-touch"\n'
    (reference/'registry.toml').write_text(original)
    result=task.routed_registry(reference,tmp_path/'new')
    import tomllib
    parsed=tomllib.loads(result)
    assert parsed['storage']=={'persistent_root':str(tmp_path/'new'),'data_root':str(tmp_path/'new')}
    assert parsed['datasets']['foo']['data_root']=='/do-not-touch'
