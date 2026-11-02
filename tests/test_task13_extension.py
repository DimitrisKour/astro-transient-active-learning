"""CPU-only Task13 extension regression checks."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/adaptive'))
import task13_extension as ext


def test_plan_and_budget_args(monkeypatch):
    assert list(ext.CYCLES) == [5, 6, 7, 8, 9]
    assert len(list(ext.cases())) == 30
    class Frozen:
        @staticmethod
        def runner_args(seed, policy, job_dir):
            return ['--max-cycles', '4', '--seed', str(seed), '--strategies', policy,
                    '--training-initialization', 'warm_start']
    monkeypatch.setattr(ext, 'frozen_task13', lambda: Frozen())
    first = ext.runner_args('tailor_diversity', 135, 5)
    later = ext.runner_args('uniform_mixture', 357, 9)
    assert '--extend-budget' in first and '--extend-budget' not in later
    for case in (first, later):
        assert case[case.index('--max-cycles')+1] == '9'
        assert '--complete-pool' in case
        assert case[case.index('--training-initialization')+1] == 'warm_start'


def test_held_chains_and_idempotent_release(tmp_path, monkeypatch):
    monkeypatch.setattr(ext, 'verify', lambda *_: None)
    submitted, released = [], []
    def fake_check(args, text):
        submitted.append(args)
        return str(710000 + len(submitted))
    def fake_run(args, check):
        released.append(args)
    monkeypatch.setattr(ext.subprocess, 'check_output', fake_check)
    monkeypatch.setattr(ext.subprocess, 'run', fake_run)
    ext.submit(tmp_path)
    ledger = json.loads((tmp_path/'submitted.json').read_text())
    assert ledger['state']=='held' and len(ledger['jobs'])==30
    assert all('--hold' in args for args in submitted)
    for policy in ext.POLICIES:
        for seed in ext.SEEDS:
            chain=[j for j in ledger['jobs'] if j['policy']==policy and j['seed']==seed]
            assert [j['cycle'] for j in chain]==list(ext.CYCLES)
            assert chain[0]['parent'] is None
            for a,b in zip(chain, chain[1:]):
                assert b['parent']==a['job_id']
    ext.release(tmp_path, policy='tailor_diversity', seed=135, cycle=5)
    ext.release(tmp_path, all_jobs=True)
    assert len(released)==30
    assert json.loads((tmp_path/'submitted.json').read_text())['state']=='released_all'
    ext.release(tmp_path, all_jobs=True)
    assert len(released)==30
