import importlib.util
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'scripts/adaptive'
sys.path.insert(0,str(SCRIPT))
import task11,task11_patch


def test_static_plan_is_nine_independent_trajectories():
    plan=task11.scientific_plan()
    assert plan['fits']==81
    assert plan['policies']==['badge','core_set','fisher_batch']
    cases=list(task11.jobs())
    assert len(cases)==81
    assert len(set(cases))==81
    assert len({(p,s) for p,s,_ in cases})==9
    assert all([c for p,s,c in cases if (p,s)==(policy,seed)]==list(range(1,10))
               for policy in task11.POLICIES for seed in task11.SEEDS)


def test_task4_scientific_options_are_unchanged():
    args=task11.task4_args(135,'badge')
    assert args[args.index('--training-initialization')+1]=='warm_start'
    assert args[args.index('--bootstrap-source-suite-id')+1]==task11.BASE
    assert args[args.index('--max-epochs')+1]=='50'
    assert args[args.index('--patience')+1]=='10'
    assert args[args.index('--split-seed')+1]=='135'
    assert '--complete-pool' in args
    assert '--run' not in args
    assert '--cycle' not in args


def test_invalid_method_rejected():
    with pytest.raises(ValueError):task11.task4_args(135,'fisher')
    with pytest.raises(ValueError):task11.task4_args(999,'badge')


def test_patch_is_strict():
    assert task11_patch.replace_once('x only x','only','once','test')=='x once x'
    with pytest.raises(ValueError):task11_patch.replace_once('a a','a','z','ambiguous')
    with pytest.raises(ValueError):task11_patch.replace_once('no','missing','z','missing')


def test_submit_requires_pilot_evidence(monkeypatch,tmp_path):
    monkeypatch.setattr(task11,'JOB',tmp_path)
    monkeypatch.setattr(task11,'verify',lambda *args,**kw:None)
    with pytest.raises(ValueError,match='pilot'):
        task11.submit('00:20:00')
