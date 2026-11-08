"""Task17 random L0 provenance, patched source, and held Slurm DAG tests."""
from __future__ import annotations

import ast
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/research/historical_replay/task17'))
import task17
from task17_patch import patch_source


def test_budget_and_reference():
    assert task17.L0_SIZE == 476
    assert task17.U0_SIZE == 4253
    assert task17.BATCH_SIZE == 473
    assert task17.CYCLES == 3
    assert task17.BATCH_SIZE * task17.CYCLES == 1419
    p = task17.plan()
    assert p['new_bootstrap_fits'] == 3
    assert p['new_active_learning_fits'] == 18
    assert p['split']['class_coverage_guaranteed'] is False
    assert p['split']['mode'].startswith('uniform_random')


@pytest.mark.parametrize('mode', ['preflight','bootstrap','check-bootstrap','cycle'])
def test_runner_args_and_new_bootstrap(mode):
    args=task17.runner_args(135,'entropy',mode=mode)
    assert args[args.index('--bootstrap-source-suite-id')+1] == task17.BOOTSTRAP
    assert args[args.index('--l0-count')+1] == '476'
    assert args[args.index('--l0-selection')+1] == 'uniform_random'
    assert args[args.index('--split-seed')+1] == str(task17.SPLIT_SEED)
    assert '--complete-pool' not in args
    assert ('--bootstrap-only' in args) == (mode=='bootstrap')
    assert ('--require-existing-bootstrap' in args) == (mode in ('check-bootstrap','cycle'))
    assert ('--run' in args) == (mode=='bootstrap')


def test_runner_rejects_wrong_policy_seed_mode():
    for kw in [{'seed':999},{'policy':'badge'},{'mode':'unknown'}]:
        with pytest.raises(ValueError):
            task17.runner_args(kw.get('seed',135),kw.get('policy','entropy'),mode=kw.get('mode','preflight'))


def source_fixture(tmp_path, task11_frozen_source):
    folder=tmp_path/'source'
    shutil.copytree(task11_frozen_source,folder)
    return folder


def test_strict_patch_three_files_and_python_syntax(tmp_path, task11_frozen_source):
    root=source_fixture(tmp_path, task11_frozen_source)
    previous=task17.source_hashes(root)
    changed=patch_source(root)
    after=task17.source_hashes(root)
    assert len(changed)==3
    assert {k for k in after if after[k]!=previous[k]}==set(changed)
    for file in task17.source_files(root):
        if file.suffix=='.py':
            ast.parse(file.read_text(),filename=str(file))
    assert 'uniform_random' in (root/'src/al_pipeline/astronomy/experiments/historical_replay.py').read_text()
    assert 'max(count, 1)' in (root/'src/al_pipeline/astronomy/astrominn/training.py').read_text()
    with pytest.raises(ValueError, match='anchor'):
        patch_source(root)


def test_actual_uniform_object_selection_order_independence_and_idempotence(tmp_path, task11_frozen_source):
    root=source_fixture(tmp_path, task11_frozen_source)
    patch_source(root)
    sys.path.insert(0,str(root/'src'))
    for key in list(sys.modules):
        if key == 'al_pipeline' or key.startswith('al_pipeline.'):
            del sys.modules[key]
    from al_pipeline.astronomy.data.registry import DatasetSpec
    from al_pipeline.astronomy.experiments.historical_replay import materialize_astrominn_historical_replay_split
    train=tmp_path/'training.csv'; val=tmp_path/'validation.csv'; test=tmp_path/'test.csv'
    fields=['object_id','filepath','label','alert']
    all_rows=[]
    # Purposefully unbalanced population; two alert rows per object.
    groups=[('AGN',14),('TDE',2),('SN I',40),('SN II',10),('CV',7)]
    for label,n in groups:
        for i in range(n):
            oid=f'{label.replace(" ", "")}_{i:03d}'
            for alert in range(2):
                all_rows.append({'object_id':oid,'filepath':f'{oid}_{alert}.npz',
                                 'label':label,'alert':str(alert)})
    for path,rows in [(train,all_rows),(val,[{'object_id':'val01','filepath':'v','label':'AGN','alert':'0'}]),
                      (test,[{'object_id':'test01','filepath':'t','label':'AGN','alert':'0'}])]:
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    ds=DatasetSpec(name='fake',survey='ztf',modality='images',selection_unit='object',
                   train_manifest=train,validation_manifest=val,test_manifest=test,
                   data_root=tmp_path,object_id_column='object_id',
                   sample_file_column='filepath',label_column='label',options={'max_alert':10})
    kw=dict(dataset=ds,initial_fraction=.10,minimum_initial_per_class=5,
            split_seed=task17.SPLIT_SEED,selection_mode='uniform_random',
            initial_object_count=15)
    one=materialize_astrominn_historical_replay_split(destination=tmp_path/'split_one',pool_id='sample1',**kw)
    again=materialize_astrominn_historical_replay_split(destination=tmp_path/'split_one',pool_id='sample1',**kw)
    assert one.initial_object_ids==again.initial_object_ids
    assert len(one.initial_object_ids)==15
    assert len(one.candidate_object_ids)==73-15
    assert not set(one.initial_object_ids)&set(one.candidate_object_ids)
    assert all(len([r for r in all_rows if r['object_id']==oid])==2 for oid in one.initial_object_ids)
    summary=json.loads(one.summary_path.read_text())
    assert summary['design']=='historical_train_object_uniform_random_replay'
    assert summary['initial_object_count_requested']==15
    assert sum(summary['initial_object_label_counts'].values())==15
    # Compare directly to the prespecified uniform RNG operation; no class-wise sampling.
    choices=np.random.default_rng(np.random.SeedSequence([task17.SPLIT_SEED,17])).choice(
        np.asarray(sorted({r['object_id'] for r in all_rows}),dtype=object),size=15,replace=False)
    assert set(one.initial_object_ids)==set(choices)
    # Different random initial sets require different seed; no rejected sample balancing.
    two=materialize_astrominn_historical_replay_split(
        destination=tmp_path/'split_two',pool_id='sample2',**(kw|{'split_seed':777}))
    assert set(two.initial_object_ids)!=set(one.initial_object_ids)


def test_missing_class_weight_smoothing_no_hidden_labels(tmp_path, task11_frozen_source):
    root=source_fixture(tmp_path, task11_frozen_source);patch_source(root)
    sys.path.insert(0,str(root/'src'))
    for key in list(sys.modules):
        if key == 'al_pipeline' or key.startswith('al_pipeline.'):
            del sys.modules[key]
    from al_pipeline.astronomy.astrominn.training import _class_weights,AstroMiNNViTTrainingConfig
    import torch
    manifest=tmp_path/'l0.csv'
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['object_id','label'])
        writer.writeheader()
        writer.writerows([{'object_id':'x'+str(i),'label':'SN I'} for i in range(10)])
    weights,summary=_class_weights(policy=AstroMiNNViTTrainingConfig(class_weight_source='training_manifest'),
                                   device=torch.device('cpu'),train_manifest=manifest,label_column='label')
    assert summary['sample_counts']['SN I']==10
    assert summary['sample_counts']['TDE']==0
    assert 'TDE' in summary['zero_count_classes']
    assert torch.isfinite(weights).all()


def test_local_prepare_and_verify(tmp_path,monkeypatch, task11_frozen_source):
    parent=task11_frozen_source.parent
    for name,value in [('TASK11',parent),('JOB',tmp_path/'job'),
                       ('PERSISTENT',tmp_path/'persistent'),('EXCLUDE',tmp_path/'external.csv')]:
        monkeypatch.setattr(task17,name,value)
    (tmp_path/'external.csv').write_text('object_id\n')
    task17.prepare(ROOT)
    task17.verify()
    changed=task17.load(task17.JOB/'source_parent.json')['patched_files']
    assert len(changed)==3
    task17.prepare(ROOT)  # no rewrite on repeated preparation
    target=task17.JOB/'source/scripts/run_astrominn_closed_loop.py'
    target.write_text(target.read_text()+'\n# tampered\n')
    with pytest.raises(RuntimeError,match='mismatch'):
        task17.verify()


def test_held_bootstrap_dag(tmp_path,monkeypatch):
    monkeypatch.setattr(task17,'JOB',tmp_path)
    monkeypatch.setattr(task17,'verify',lambda *a,**kw:None)
    (tmp_path/'preflight_audit.json').write_text('{}')
    calls=[];release=[]
    monkeypatch.setattr(task17.subprocess,'check_output',lambda cmd,*,text: calls.append(cmd) or str(770000+len(calls)))
    monkeypatch.setattr(task17.subprocess,'run',lambda cmd,*,check: release.append(cmd))
    task17.submit_bootstraps('00:35:00')
    assert len(calls)==3
    assert all('--hold' in cmd and '--dependency' not in cmd for cmd in calls)
    task17.submit_bootstraps('00:35:00')
    assert len(calls)==3
    task17.release_bootstraps(seed_filter=135)
    assert len(release)==1
    ledger=task17.load(tmp_path/'bootstrap_jobs.json')
    assert sum(x['released'] for x in ledger['jobs'])==1


def test_held_cycles_and_per_seed_dependencies(tmp_path,monkeypatch):
    monkeypatch.setattr(task17,'JOB',tmp_path)
    monkeypatch.setattr(task17,'verify',lambda *a,**kw:None)
    monkeypatch.setattr(task17,'validate_bootstrap',lambda s:None)
    task17.dump(tmp_path/'pilot/seed_135/result.json',
                {'unique':True,'selected_ids':{'random':['x'],'entropy':['y']}})
    requests=[];release=[]
    monkeypatch.setattr(task17.subprocess,'check_output',lambda cmd,*,text: requests.append(cmd) or str(990000+len(requests)))
    monkeypatch.setattr(task17.subprocess,'run',lambda cmd,*,check: release.append(cmd))
    task17.submit_cycles('00:20:00')
    ledger=task17.load(tmp_path/'submitted.json')['jobs']
    assert len(ledger)==18
    assert len(requests)==18
    assert sum('--dependency' not in x for x in requests)==6
    assert sum('--dependency' in x for x in requests)==12
    assert all('--hold' in x for x in requests)
    assert not release
    task17.submit_cycles('00:20:00')
    assert len(requests)==18
    task17.release_cycles(policy='entropy',seed=135,cycle=1)
    assert len(release)==1
    task17.release_cycles(all_jobs=True)
    assert len(release)==18


def test_submit_guards(tmp_path,monkeypatch):
    monkeypatch.setattr(task17,'JOB',tmp_path)
    monkeypatch.setattr(task17,'verify',lambda *a,**kw:None)
    with pytest.raises(ValueError,match='walltime'):
        task17.submit_bootstraps('wrong')
    with pytest.raises(RuntimeError,match='preflight'):
        task17.submit_bootstraps('00:20:00')
    with pytest.raises(RuntimeError,match='pilot required'):
        task17.submit_cycles('00:20:00')
    task17.dump(tmp_path/'submitted.json', {'jobs':[]})
    with pytest.raises(ValueError):
        task17.release_cycles(policy='entropy',seed=135)


def test_cli_release_all_no_implicit_seed(monkeypatch):
    calls=[]
    monkeypatch.setattr(task17,'release_cycles',lambda *args:calls.append(args))
    monkeypatch.setattr(sys,'argv',['task17.py','release-cycles','--all'])
    task17.main()
    assert calls==[(None,None,None,True)]


def test_results_export_includes_validation_but_not_hidden_labels(tmp_path,monkeypatch):
    from zipfile import ZipFile
    p=tmp_path/'persistent'
    d=tmp_path/'data'
    job=p/'closed_loop'/task17.SUITE
    monkeypatch.setattr(task17,'PERSISTENT',p)
    monkeypatch.setattr(task17,'DATA',d)
    monkeypatch.setattr(task17,'JOB',job)
    split=p/'runs'/task17.SUITE/'shared/historical_replay_split'
    split.mkdir(parents=True)
    (split/'hidden_labels.csv').write_text('secret')
    (split/'split_summary.json').write_text('{}')
    (split/'initial_labeled_manifest.csv').write_text('object_id,label\nx,A\n')
    rel=Path(task17.SUITE)/'entropy'/'seed_135'
    ev=d/'runs'/rel/'cycle_001/evaluation'
    ev.mkdir(parents=True)
    (ev/'evaluation_summary.json').write_text('{"ok":true}')
    out=tmp_path/'report.zip'
    task17.collect(out)
    with ZipFile(out) as archive:
        names=archive.namelist()
    assert any(x.endswith('cycle_001/evaluation/evaluation_summary.json') for x in names)
    assert any(x.endswith('initial_labeled_manifest.csv') for x in names)
    assert not any('hidden_labels' in x for x in names)
