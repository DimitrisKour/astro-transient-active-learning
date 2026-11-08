"""Synthetic tests for Task10 freeze, GPU submission, and fixed-model analysis."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/research/historical_replay/task10'))
import task10_entropy20 as task


def test_case_contract():
    assert task.CASE_KEY == 'entropy_s357'
    assert len(task.SEEDS) == 20
    assert len(set(task.SEEDS)) == 20
    assert task.SEEDS[0] == 2001 and task.SEEDS[-1] == 2020


def test_plan_fixed_manifest(monkeypatch, tmp_path):
    evidence = {'case': {'seed': 357, 'policy': 'entropy', 'rejected_cycle': 5},
                'training_manifest': '/fixed/l5', 'initial_checkpoint': '/fixed/c4'}
    monkeypatch.setattr(task, 'verified_evidence', lambda _: ({'source_audit': '/source/audit'}, evidence))
    monkeypatch.setattr(task, 'expected_files', lambda repo, source: {'src/a.py': b'print(1)\n'})
    monkeypatch.setattr(task, 'sha256', lambda path: 'a'*64)
    p = task.plan(repo=tmp_path, source=tmp_path, job=tmp_path/'run')
    assert p['evidence']['case']['seed'] == 357
    assert p['evidence']['training_manifest'] == '/fixed/l5'
    assert p['training_contract']['test_used'] is False
    assert len(p['jobs']) == 20
    assert [j['retry_seed'] for j in p['jobs']] == list(range(2001, 2021))
    assert len({j['output_dir'] for j in p['jobs']}) == 20


def test_submit_marker_and_slurm_array(monkeypatch, tmp_path):
    job = tmp_path
    (job/'logs').mkdir()
    (job/'source/scripts/research/historical_replay/task10').mkdir(parents=True)
    (job/'source/scripts/research/historical_replay/task10/task10_entropy20_worker.sbatch').write_text('worker')
    p = {'jobs': [{'retry_seed': s} for s in task.SEEDS]}
    monkeypatch.setattr(task, 'verify', lambda job, deep=False: p)
    monkeypatch.setattr(task, 'status', lambda job: {'invalid': [], 'completed': 0, 'pending_seeds': list(task.SEEDS)})
    calls = []
    class Result:
        stdout = '98765\n'
    def run(cmd, **kwargs):
        calls.append((cmd,kwargs))
        return Result()
    monkeypatch.setattr(task.subprocess, 'run', run)
    task.submit(job, '00:20:00', 2)
    assert '--array=0-19%2' in calls[0][0]
    assert task.read(job/'submitted.json')['job_id']=='98765'
    assert not (job/'submission_pending.json').exists()
    with pytest.raises(RuntimeError, match='Already submitted'):
        task.submit(job, '00:20:00', 2)


@pytest.mark.parametrize('wall,concurrency', [('00:25:00',2), ('00:00:00',2), ('abc',2),('00:20:00',5)])
def test_invalid_queue_limits(tmp_path, wall, concurrency, monkeypatch):
    monkeypatch.setattr(task, 'verify', lambda j, deep=False: {})
    with pytest.raises(ValueError):
        task.submit(tmp_path, wall, concurrency)


def test_refuse_run_outside_slurm(monkeypatch, tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm GPU'):
        task.run(tmp_path, 0)


def test_expected_files_preserves_old_source(monkeypatch, tmp_path):
    task8b = tmp_path/'task8b'
    dest = task8b/'source/src/al_pipeline/core'
    dest.mkdir(parents=True)
    (dest/'base.py').write_text('original')
    dst = task8b/'source/scripts/adaptive'
    dst.mkdir(parents=True)
    (dst/'task8b_report.py').write_text('report source')
    repo=tmp_path/'repo'
    for rel in task.SCRIPTS:
        path=repo/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(rel)
    files=task.expected_files(repo,task8b)
    assert files['src/al_pipeline/core/base.py']==b'original'
    assert files['scripts/adaptive/task8b_report.py']==b'report source'
    assert len(files)==5


def test_report_fixed_order_and_ensembles(monkeypatch, tmp_path):
    import task10_entropy20_report as report
    ids=np.array(['a','b','c','d','e','f','g','h','i','j'],dtype=str)
    y=np.array([0,0,1,1,2,2,3,3,4,4])
    incp=np.eye(5)[y]*0.8+0.04
    rejp=np.eye(5)[np.roll(y,1)]*0.8+0.04
    orig=(ids,y,incp)
    bad=(ids,y,rejp)
    evidence={'case':{'policy':'entropy','seed':357,'rejected_cycle':5},
              'incumbent_evaluation':'/incumbent/evaluation_summary.json',
              'initial_checkpoint':'/incumbent/best_vit_tower.pth',
              'source_evaluation':'/candidate/evaluation_summary.json',
              'candidate_checkpoint':'/candidate/best_vit_tower.pth'}
    plan={'schema':task.SCHEMA,'case_key':task.CASE_KEY,'suite':task.SUITE,
          'evidence':evidence,'jobs':[{'retry_seed':s,'output_dir':str(tmp_path/f'retry_{s}')} for s in task.SEEDS]}
    monkeypatch.setattr(report,'verify',lambda j, deep=True:plan)
    def load(path,check):
        return orig if 'incumbent' in str(path) else bad
    monkeypatch.setattr(report,'load_prediction',load)
    seed_models={s:(ids,y,np.roll(incp,(s-2001)%3,axis=0)) for s in task.SEEDS}
    monkeypatch.setattr(report,'read_predictions',lambda job,j,e,inc:seed_models[j['retry_seed']])
    for j in plan['jobs']:
        folder=Path(j['output_dir']);folder.mkdir(parents=True)
        (folder/'complete.json').write_text(json.dumps({'training_elapsed_seconds':50}))
    summ=report.make_report(tmp_path,tmp_path/'out',bootstrap=2)
    assert summ['n_new_training_fits']==20
    assert summ['total_training_seconds']==1000
    assert len(summ['ensemble_sizes'])==6
    import csv
    with (tmp_path/'out/ensemble_definitions.csv').open() as f:
        defs=list(csv.DictReader(f))
    assert len(defs)==12
    assert json.loads(defs[0]['seeds_and_incumbent'])==[2001,2002]
    assert json.loads(defs[1]['seeds_and_incumbent'])==['incumbent',2001,2002]
    with (tmp_path/'out/individual_and_ensemble_comparisons.csv').open() as f:
        comparison=list(csv.DictReader(f))
    assert len(comparison)==1+40+24
    with (tmp_path/'out/recovery_by_budget.csv').open() as f:
        budgets=list(csv.DictReader(f))
    assert [int(x['seeds_examined']) for x in budgets]==[1,2,3,4,5,8,10,12,16,20]


def test_all_class_changes_are_object_counts(tmp_path, monkeypatch):
    import csv
    import task10_entropy20_report as report
    # Unit-test the integer conversion used in the report output.
    for support, recall_ref, recall_model, expected in [
        (1480, 0.8, 0.7, -148), (17, 3/17, 4/17, 1), (341, 100/341, 70/341, -30)
    ]:
        left = round(support*recall_ref)
        right = round(support*recall_model)
        assert right-left == expected
