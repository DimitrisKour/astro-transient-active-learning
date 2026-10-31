"""Core DIRECT acquisition and deployment invariants; no GPU required."""
from __future__ import annotations
import ast
import importlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
LOCAL_SOURCE=ROOT/'src'
sys.path.insert(0,str(LOCAL_SOURCE))
sys.path.insert(0,str(ROOT/'scripts/adaptive'))
from al_pipeline.core.advanced_acquisition.direct import (DirectInput,DirectSelector,_losses,
                    _optimal_threshold,_shrink_interval,_allocation,_margins)


def fixture(seed=1,n=95):
    rng=np.random.default_rng(seed)
    c=5
    p=rng.dirichlet(np.ones(c),size=n)
    anchors=rng.dirichlet(np.ones(c),size=25)
    labels=tuple(('AGN','TDE','SN I','SN II','CV')[i%5] for i in range(25))
    return DirectInput(candidate_ids=tuple(f'U_{i}' for i in range(n)),probabilities=p,
        confirmed_ids=tuple(f'L_{i}' for i in range(25)),confirmed_probabilities=anchors,
        confirmed_labels=labels,class_names=('AGN','TDE','SN I','SN II','CV'))


def run(seed=61,n=95,budget=47,noise=False):
    data=fixture(n=n)
    truth=dict(zip(data.candidate_ids,[data.class_names[i%5] for i in range(n)]))
    seen=[];callbacks=[]
    def oracle(ids,*,phase,class_name,subround):
        assert not set(ids)&set(seen)
        assert all(i in truth for i in ids)
        assert all(len(q)<=5 for q in callbacks if q['phase']=='vreduce')
        seen.extend(ids)
        callbacks.append({'ids':tuple(ids),'phase':phase,'class':class_name,'subround':subround})
        return [truth[i] for i in ids]
    ans=DirectSelector(data,parallel_batch=5,seed=seed).select(budget=budget,query=oracle)
    return ans,seen,callbacks

@pytest.mark.parametrize('budget',[1,2,4,5,6,7,13,47,85,95])
def test_exact_budget_phase_order_uniqueness(budget):
    sel,seen,queries=run(budget=budget)
    assert len(sel)==budget and len(set(sel.ids))==budget
    assert set(sel.ids)==set(seen)
    assert sum(q['phase']=='vreduce' for q in queries)>0 or budget==1
    stage=[q['phase'] for q in queries]
    assert stage==sorted(stage,key=lambda x: {'vreduce':0,'boundary':1}[x])
    assert sel.metadata['phase1_budget']==budget//2
    assert sel.metadata['phase2_budget']==budget-budget//2
    assert len(sel.metadata['steps'])==budget


def test_deterministic_and_seed_changes():
    a,_,_=run(seed=4)
    b,_,_=run(seed=4)
    c,_,_=run(seed=5)
    assert a.ids==b.ids and a.metadata==b.metadata
    assert a.ids!=c.ids


def test_empirical_threshold_loss_orientation():
    # First 3 REST, last 3 positive -> cut at j=3 yields zero loss.
    obs={i:('AGN' if i>=3 else 'SN I') for i in range(6)}
    loss=_losses(obs,'AGN',6)
    assert np.argmin(loss)==3 and loss[3]==0
    assert _optimal_threshold(obs,'AGN',6)==3
    lo,hi=_shrink_interval(obs,'AGN',6,0,6,2)
    assert lo<=3<=hi and hi-lo==2


def test_margins_include_target_and_deterministic_allocation():
    p=np.array([[.8,.1,.1],[.2,.6,.2]])
    np.testing.assert_allclose(_margins(p),[[0,-.7,-.7],[-.4,0,-.4]])
    assert _allocation(17,5,[4,3,2,1,0]).tolist()==[3,3,3,4,4]


def test_invalid_input_no_candidate_labels_or_postselection_leak():
    d=fixture()
    assert not hasattr(d,'candidate_labels')
    with pytest.raises(ValueError,match='disjoint'):
        DirectInput(d.candidate_ids,d.probabilities,('U_0',)+d.confirmed_ids[1:],
                    d.confirmed_probabilities,d.confirmed_labels,d.class_names)
    def bad(ids,**kw):return ['UNKNOWN']*len(ids)
    with pytest.raises(ValueError,match='Oracle'):
        DirectSelector(d).select(budget=5,query=bad)


def test_patch_verified_parent(tmp_path, task11_frozen_source):
    from task15_patch import patch_source
    parent=task11_frozen_source
    source=tmp_path/'source'
    shutil.copytree(parent,source)
    patch_source(source,ROOT)
    for p in source.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    text=(source/'src/al_pipeline/astronomy/experiments/simulated.py').read_text()
    assert 'direct_oracle_journal.json' in text
    assert '_task15_select_or_resume' in text
    assert 'Task15 runner permits' in (source/'scripts/run_astrominn_closed_loop.py').read_text()


def test_submit_has_independent_chains_and_held_jobs(tmp_path,monkeypatch):
    import task15
    monkeypatch.setattr(task15,'JOB',tmp_path)
    monkeypatch.setattr(task15,'verify',lambda *a,**k:None)
    f=tmp_path/'pilot/direct/seed_135/result.json';f.parent.mkdir(parents=True);f.write_text(json.dumps({'selected_unique':True}))
    calls=[]
    def fake(args,*,text):
        calls.append(args)
        return str(500000+len(calls))
    monkeypatch.setattr(task15.subprocess,'check_output',fake)
    task15.submit('00:20:00')
    ledger=task15.load(tmp_path/'submitted.json')
    assert len(calls)==27 and all('--hold' in x for x in calls)
    assert len([x for x in calls if '--dependency' not in x])==3
    assert len([x for x in calls if '--dependency' in x])==24
    assert len(ledger['jobs'])==27 and all(not x['released'] for x in ledger['jobs'])


def test_invalid_walltime_refuses_submission(tmp_path,monkeypatch):
    import task15
    monkeypatch.setattr(task15,'JOB',tmp_path)
    monkeypatch.setattr(task15,'verify',lambda *a,**k:None)
    with pytest.raises(ValueError,match='walltime'):task15.submit('banana')


def test_integrated_interactive_query_journal_resume(tmp_path, task11_frozen_source):
    """Run actual patched Task15 source in an isolated Python process.

    The standard checkout does not contain Task15's frozen simulator patch;
    loading it directly produced a meaningless skip. Re-create the patch and
    execute against that code, keeping main pytest imports isolated.
    """
    import os
    import subprocess
    import textwrap
    from task15_patch import patch_source

    source=tmp_path/'source'
    shutil.copytree(task11_frozen_source,source)
    patch_source(source,ROOT)
    script=textwrap.dedent(r'''
        import json
        import types
        from pathlib import Path
        import numpy as np
        from al_pipeline.core.types import ClassificationPredictions
        from al_pipeline.astronomy.experiments.simulated import SimulatedActiveLearningExperiment
        rng=np.random.default_rng(3)
        names=('AGN','TDE','SN I','SN II','CV')
        ids=tuple(f'U_{i}' for i in range(95))
        refids=tuple(f'L_{i}' for i in range(25))
        p=rng.dirichlet(np.ones(5),size=95)
        lp=rng.dirichlet(np.ones(5),size=25)
        ref_labels=tuple(names[i%5] for i in range(25))
        fake=types.SimpleNamespace()
        fake.config=types.SimpleNamespace(seed=135)
        fake.class_names=names
        fake.hidden_labels={oid:names[i%5] for i,oid in enumerate(ids)}
        fake._task11_confirmed_inference=lambda cycle:types.SimpleNamespace(
            object_ids=refids,
            predictions=ClassificationPredictions(lp,'probabilities'))
        fake._task15_confirmed_labels=lambda cycle:dict(zip(refids,ref_labels,strict=True))
        inf=types.SimpleNamespace(object_ids=ids,
            predictions=ClassificationPredictions(p,'probabilities'))
        select=SimulatedActiveLearningExperiment._task15_select_or_resume
        path=Path('selection.json')
        first=select(fake,inference=inf,budget=47,cycle=1,path=path)
        journal=json.loads(Path('direct_oracle_journal.json').read_text())
        assert len(journal)==first.metadata['oracle_query_count']
        assert {oid for row in journal for oid in row['ids']}==set(first.ids)
        assert all(row['phase']=='vreduce' for row in journal[:sum(j['phase']=='vreduce' for j in journal)])
        assert len(first)==47
        path.unlink()  # crash after all queries, before saved selection
        fake.hidden_labels={}  # replay cannot request any new labels
        second=select(fake,inference=inf,budget=47,cycle=1,path=path)
        assert second.ids==first.ids and second.metadata==first.metadata
        assert json.loads(Path('direct_oracle_journal.json').read_text())==journal
        third=select(fake,inference=inf,budget=47,cycle=1,path=path)
        assert third.ids==first.ids
    ''')
    env={**os.environ, 'PYTHONPATH':str(source/'src')}
    result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,
                          env=env,capture_output=True,text=True)
    assert result.returncode==0, result.stdout+'\n'+result.stderr
