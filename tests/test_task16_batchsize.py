"""Task16 frozen-source, budget and held-submission invariants."""
from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/adaptive'))
import task16
from task16_patch import patch_source


@pytest.mark.parametrize('fraction, batch, cycles, last', [
    ('05', 237, 18, 224), ('20', 946, 5, 469)
])
def test_full_pool_budget(fraction, batch, cycles, last):
    result = task16.schedule(fraction)
    assert result['batch'] == batch
    assert result['cycles'] == cycles
    assert result['last_batch'] == last
    assert batch * (cycles - 1) + last == task16.U0_OBJECTS
    assert result['final_labeled'] == task16.HISTORICAL_OBJECTS


def test_existing_reference_and_fits():
    plan = task16.plan()
    assert plan['planned_new_training_jobs'] == 138
    assert plan['planned_new_trajectories'] == 12
    assert plan['existing_reference']['batch'] == 473
    assert 473*8+469 == task16.U0_OBJECTS
    assert len(list(task16.jobs())) == 138
    assert {p for _,p,_,_ in task16.jobs()} == {'random', 'entropy'}


def test_budget_corresponds_to_frozen_runner(task11_frozen_source):
    parent = task11_frozen_source
    sys.path.insert(0, str(parent/'src'))
    from al_pipeline.core.experiments.budget import plan_cycle_budget
    for fraction in task16.SCHEDULES:
        expected = task16.schedule(fraction)
        actual = plan_cycle_budget(
            candidate_objects=4253,
            acquisition_size=expected['batch'],
            max_cycles=expected['cycles'],
            complete_pool=True)
        assert (actual.max_cycles, actual.final_cycle_size, actual.total_acquisitions) == (
            expected['cycles'], expected['last_batch'], 4253)


def test_runner_arguments_contain_full_pool_and_inherit_l0():
    for frac, policy, seed, _ in task16.jobs():
        args = task16.runner_args(frac, policy, seed)
        assert '--complete-pool' in args
        assert '--require-existing-bootstrap' in args
        assert args[args.index('--bootstrap-source-suite-id') + 1] == task16.BASE
        assert args[args.index('--max-cycles') + 1] == str(task16.schedule(frac)['cycles'])
        assert args[args.index('--acquisition-fraction') + 1] == str(task16.SCHEDULES[frac])
        assert args[args.index('--strategies') + 1] == policy
    with pytest.raises(ValueError):
        task16.runner_args('10', 'entropy', 135)
    with pytest.raises(ValueError):
        task16.runner_args('05', 'badge', 135)


def test_patch_is_narrow_and_strict(tmp_path, task11_frozen_source):
    parent = task11_frozen_source
    copied = tmp_path/'source'
    shutil.copytree(parent, copied)
    previous = task16.source_hashes(copied)
    changed = patch_source(copied)
    current = task16.source_hashes(copied)
    assert changed == ['scripts/run_astrominn_closed_loop.py']
    assert {key for key in previous if previous[key] != current[key]} == set(changed)
    for item in task16.source_files(copied):
        if item.suffix == '.py':
            ast.parse(item.read_text(), filename=str(item))
    assert 'Task16 runner permits only random and entropy' in (
        copied/'scripts/run_astrominn_closed_loop.py').read_text()
    with pytest.raises(ValueError, match='anchor'):
        patch_source(copied)


def test_preparation_and_freeze(tmp_path, monkeypatch, task11_frozen_source):
    parent = task11_frozen_source.parent
    for name, value in [('TASK11', parent), ('JOB', tmp_path/'job'),
                        ('PERSISTENT', tmp_path/'persistent'),
                        ('DATA', tmp_path/'data'),
                        ('EXCLUDE', tmp_path/'exclude.csv')]:
        monkeypatch.setattr(task16, name, value)
    task16.EXCLUDE.write_text('object_id\n')
    for seed in task16.SEEDS:
        cp = task16.DATA/'runs'/task16.BASE/'shared'/f'seed_{seed}/bootstrap_training/best_vit_tower.pth'
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(b'stub')
    task16.prepare(ROOT)
    task16.verify()
    plan = task16.load(task16.JOB/'scientific_plan.json')
    assert plan['planned_new_training_jobs'] == 138
    frozen = task16.JOB/'source/scripts/run_astrominn_closed_loop.py'
    assert 'Task16 runner permits' in frozen.read_text()
    assert task16.sha(parent/'source/scripts/run_astrominn_closed_loop.py') != task16.sha(frozen)
    task16.prepare(ROOT)  # must not overwrite immutable source on a second call
    frozen.write_text(frozen.read_text()+'\n# unexpected change\n')
    with pytest.raises(RuntimeError, match='mismatch'):
        task16.verify()


def make_stub_pilot(folder):
    pilot = folder/'pilot/seed_135/result.json'
    pilot.parent.mkdir(parents=True, exist_ok=True)
    pilot.write_text(json.dumps({
        'valid_replay': True,
        'cases': [{'fraction': f, 'policy': p} for f in task16.SCHEDULES for p in task16.POLICIES]
    }))


def test_held_submission_12_independent_chains_and_release(tmp_path, monkeypatch):
    monkeypatch.setattr(task16, 'JOB', tmp_path)
    monkeypatch.setattr(task16, 'verify', lambda *a,**kw: None)
    make_stub_pilot(tmp_path)
    submissions=[]
    actions=[]
    def fake_check(cmd, *, text):
        submissions.append(cmd)
        return str(830000 + len(submissions))
    def fake_run(cmd, *, check):
        actions.append(cmd)
    monkeypatch.setattr(task16.subprocess, 'check_output', fake_check)
    monkeypatch.setattr(task16.subprocess, 'run', fake_run)
    task16.submit('00:20:00')
    ledger = task16.load(tmp_path/'submitted.json')
    assert len(ledger['jobs']) == 138
    assert len(submissions) == 138
    assert all('--hold' in cmd for cmd in submissions)
    assert sum('--dependency' not in cmd for cmd in submissions) == 12
    assert sum('--dependency' in cmd for cmd in submissions) == 126
    assert not actions  # submit only holds
    assert all(not x['released'] for x in ledger['jobs'])
    task16.release('05','entropy',135,1)
    assert actions == [['scontrol','release',next(x['job_id'] for x in ledger['jobs']
            if (x['fraction'],x['policy'],x['seed'],x['cycle'])==('05','entropy',135,1))]]
    after = task16.load(tmp_path/'submitted.json')
    assert sum(x['released'] for x in after['jobs']) == 1
    task16.submit('00:20:00')
    assert len(submissions) == 138  # idempotent: no duplicate Slurm submissions


def test_invalid_walltime_prevents_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(task16, 'JOB', tmp_path)
    monkeypatch.setattr(task16, 'verify', lambda *a,**kw: None)
    with pytest.raises(ValueError, match='walltime'):
        task16.submit('nope')
    with pytest.raises(ValueError, match='walltime'):
        task16.submit('00:00:00')


def test_release_requires_full_case_or_all(tmp_path, monkeypatch):
    monkeypatch.setattr(task16, 'JOB', tmp_path)
    task16.dump(tmp_path/'submitted.json', {'state':'held','jobs':[]})
    with pytest.raises(ValueError, match='Specify'):
        task16.release(fraction='05', policy='entropy', seed=135)
    with pytest.raises(ValueError, match='cannot'):
        task16.release(fraction='05',all_jobs=True)


def test_pilot_guard_before_any_submit(tmp_path, monkeypatch):
    monkeypatch.setattr(task16, 'JOB', tmp_path)
    monkeypatch.setattr(task16, 'verify', lambda *a,**kw: None)
    with pytest.raises(RuntimeError, match='pilot required'):
        task16.submit('00:20:00')


def test_staged_20pct_submission_then_independent_05pct(tmp_path,monkeypatch):
    monkeypatch.setattr(task16, 'JOB', tmp_path)
    monkeypatch.setattr(task16, 'verify', lambda *a,**k: None)
    make_stub_pilot(tmp_path)
    requests=[]
    monkeypatch.setattr(task16.subprocess,'check_output',lambda cmd,*,text: requests.append(cmd) or str(900000+len(requests)))
    task16.submit('00:20:00',fraction_filter='20')
    ledger = task16.load(tmp_path/'submitted.json')
    assert len(ledger['jobs']) == 30 and ledger['state'] == 'held_partial'
    assert all(j['fraction'] == '20' for j in ledger['jobs'])
    task16.submit('00:35:00',fraction_filter='05',policy_filter='entropy',seed_filter=135)
    ledger = task16.load(tmp_path/'submitted.json')
    assert len(ledger['jobs']) == 48
    assert sum(j['dependency'] is None for j in ledger['jobs']) == 7
    assert ledger['state'] == 'held_partial'
    assert len(requests) == 48
    task16.submit('00:20:00',fraction_filter='20')
    assert len(requests) == 48
