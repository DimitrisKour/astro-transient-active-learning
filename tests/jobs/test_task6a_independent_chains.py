"""Scheduler-only Task6A fix: per-trajectory chains, no global barriers."""
from __future__ import annotations
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'research' / 'historical_replay' / 'task6a'
sys.path.insert(0, str(SCRIPTS))
import task6a_submit_chains as submission


def test_six_independent_chain_indices(monkeypatch, tmp_path):
    monkeypatch.setattr(submission, '_status', lambda _job, policy, seed: 0)
    plans = submission._plans(tmp_path, 9)
    assert len(plans) == 54
    assert set((p['seed'], p['policy']) for p in plans) == {
        (seed, policy) for seed in (135,246,357) for policy in ('entropy','entropy_balanced_explore')}
    for i, (seed, policy) in enumerate(
            (seed, policy) for seed in (135,246,357) for policy in ('entropy','entropy_balanced_explore')):
        chain = [p for p in plans if p['seed']==seed and p['policy']==policy]
        assert len(chain)==9
        assert [p['cycle'] for p in chain]==list(range(1,10))
        assert set(p['index'] for p in chain)=={i}


def test_probe_reuses_cycle_one(monkeypatch, tmp_path):
    (tmp_path/'probe_submission.json').write_text('{}')
    monkeypatch.setattr(submission, '_status', lambda _job, policy, seed:
                        1 if seed==135 and policy=='entropy' else 0)
    plans = submission._plans(tmp_path, 9)
    assert len(plans) == 53
    assert not any(p['seed']==135 and p['policy']=='entropy' and p['cycle']==1 for p in plans)


def test_only_same_chain_dependency(tmp_path):
    source=tmp_path/'source'
    p=dict(seed=357,policy='entropy',index=4,cycle=6)
    cmd=submission.make_command(source=source,job_dir=tmp_path,plan=p,
                                walltime='00:20:00',predecessor='123456')
    assert '--array=4' in cmd
    assert '--dependency=afterok:123456' in cmd
    assert not any('--array=0-5' in s for s in cmd)
    assert '--hold' in cmd
    assert cmd[-1]=='6'
    first=submission.make_command(source=source,job_dir=tmp_path,plan={**p,'cycle':1},
                                  walltime='00:20:00',predecessor=None)
    assert not any(a.startswith('--dependency=') for a in first)


def test_submission_mutual_exclusion(monkeypatch, tmp_path):
    # Most validations can be mocked, but submitted markers always block.
    monkeypatch.setattr(submission, 'JOB_DIR', tmp_path)
    monkeypatch.setattr(submission, 'verify', lambda _: None)
    monkeypatch.setattr(submission, 'check_splits', lambda *a: None)
    monkeypatch.setattr(submission, '_audit_state', lambda _: None)
    monkeypatch.setattr(submission, '_status', lambda *_: 0)
    (tmp_path/'submission.json').write_text('{}')
    import pytest
    with pytest.raises(RuntimeError, match='Existing submission'):
        submission.validate(tmp_path,9,'00:20:00')
