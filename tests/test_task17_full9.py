"""CPU-only checks for Task17 nine-cycle held controller."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/adaptive'))
import task17_full9 as full


def test_schedule_and_extension_args(monkeypatch):
    assert len(list(full.cases()))==54
    assert list(full.CYCLES)==list(range(1,10))
    class Parent:
        @staticmethod
        def runner_args(seed,policy,mode):
            assert mode=='cycle'
            return ['--max-cycles','3','--require-existing-bootstrap',
                    '--seed',str(seed),'--strategies',policy]
    monkeypatch.setattr(full,'parent',lambda:Parent())
    a=full.runner_args('entropy',135,4)
    b=full.runner_args('random',357,9)
    assert '--extend-budget' in a and '--extend-budget' not in b
    for args in (a,b):
        assert args[args.index('--max-cycles')+1]=='9'
        assert '--complete-pool' in args
        assert '--require-existing-bootstrap' in args


def test_54_held_jobs_and_dependencies(tmp_path, monkeypatch):
    import subprocess
    c=tmp_path/'control'
    (c/'source').mkdir(parents=True)
    monkeypatch.setattr(full,'verify',lambda *args:None)
    monkeypatch.setattr(full,'T17',tmp_path/'parent')
    submissions=[]
    def fake(cmd,*,text):
        submissions.append(cmd)
        return str(810000+len(submissions))
    monkeypatch.setattr(subprocess,'check_output',fake)
    full.submit(c,'00:20:00')
    jobs=json.loads((c/'submitted.json').read_text())['jobs']
    assert len(jobs)==54 and len(submissions)==54
    assert all('--hold' in cmd for cmd in submissions)
    for policy in full.POLICIES:
        for seed in full.SEEDS:
            chain=[x for x in jobs if x['policy']==policy and x['seed']==seed]
            assert [x['cycle'] for x in chain]==list(range(1,10))
            assert chain[0]['parent'] is None
            assert all(curr['parent']==prev['job_id'] for prev,curr in zip(chain,chain[1:]))
    full.submit(c,'00:20:00')
    assert len(submissions)==54


def test_release_requires_pilot_and_bootstraps(tmp_path,monkeypatch):
    c=tmp_path/'control';c.mkdir()
    full.write(c/'submitted.json',{'state':'held','jobs':[
        {'policy':'entropy','seed':135,'cycle':1,'job_id':'123','parent':None,'released':False}
    ]})
    monkeypatch.setattr(full,'verify',lambda *args:None)
    import pytest
    with pytest.raises(ValueError,match='--all'):
        full.release(c)
    with pytest.raises(RuntimeError,match='incomplete'):
        full.release(c,all_jobs=True)
