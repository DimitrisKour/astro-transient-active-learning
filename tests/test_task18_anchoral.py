"""Task18 AnchorAL deterministic acquisition and Slurm safety checks."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/research/historical_replay/task18'))
from al_pipeline.core.advanced_acquisition.anchoral import AnchorALInput,AnchorALSelector
from al_pipeline.core.pool import Pool
import task18


def toy():
    rng=np.random.default_rng(22)
    u=rng.normal(size=(30,5))
    l=rng.normal(size=(9,5))
    return AnchorALInput(
        candidate_ids=tuple(f'u{i}' for i in range(len(u))),
        candidate_embeddings=u,
        labeled_ids=tuple(f'l{i}' for i in range(len(l))),
        labeled_embeddings=l,
        labeled_labels=('AGN',)*4+('CV',)*3+('SN I',)*2,
        class_names=('AGN','CV','SN I','SN II','TDE'),
        seed=135,cycle=1,anchors_per_class=3,neighbors_per_anchor=2,max_subpool=12)


def test_deterministic_selected_subset_and_metadata():
    d=toy();scores=np.linspace(0,1,len(d.candidate_ids))
    a=AnchorALSelector(d).select(ids=d.candidate_ids,scores=scores,k=8,
                                 pool=Pool.from_ids(d.candidate_ids))
    b=AnchorALSelector(d).select(ids=d.candidate_ids,scores=scores,k=8,
                                 pool=Pool.from_ids(d.candidate_ids))
    assert a.ids==b.ids
    assert len(set(a.ids))==8
    assert a.metadata['labeled_counts_by_class']['TDE']==0
    assert a.metadata['no_candidate_labels_used'] is True
    assert a.metadata['subpool_size']>=8


def test_eligibility_and_pure_scores():
    d=toy()
    eligible=Pool.from_ids(d.candidate_ids[5:])
    scores=np.arange(30,dtype=float)
    out=AnchorALSelector(d).select(ids=d.candidate_ids,scores=scores,k=7,pool=eligible)
    assert len(out.ids)==7
    assert set(out.ids).issubset(d.candidate_ids[5:])
    assert np.allclose(out.scores,scores[out.indices])


def test_rejects_bad_ids_and_zero_embeddings():
    d=toy()
    scores=np.ones(30)
    with pytest.raises(ValueError):
        AnchorALSelector(d).select(ids=d.candidate_ids[::-1],scores=scores,k=3,
                                   pool=Pool.from_ids(d.candidate_ids))
    broken=np.zeros_like(d.candidate_embeddings)
    with pytest.raises(ValueError):
        AnchorALInput(candidate_ids=d.candidate_ids,candidate_embeddings=broken,
            labeled_ids=d.labeled_ids,labeled_embeddings=d.labeled_embeddings,
            labeled_labels=d.labeled_labels,class_names=d.class_names,seed=1,cycle=1)


def test_full_pool_schedule_and_cli():
    assert task18.SEEDS==(135,246,357)
    assert len(task18.CYCLES)==9
    assert task18.plan()['new_fits']==27
    assert task18.plan()['last_batch']==469
    assert '--complete-pool' in task18.args(135)
    assert task18.args(135)[task18.args(135).index('--strategies')+1]=='anchoral'


def test_held_dependency_chains(tmp_path,monkeypatch):
    monkeypatch.setattr(task18,'JOB',tmp_path)
    monkeypatch.setattr(task18,'verify',lambda *a,**kw:None)
    requested=[]
    monkeypatch.setattr(task18.subprocess,'check_output',
       lambda cmd,*,text: requested.append(cmd) or str(820000+len(requested)))
    task18.submit('00:20:00')
    jobs=task18.load(tmp_path/'submitted.json')['jobs']
    assert len(jobs)==27
    assert all('--hold' in c for c in requested)
    for seed in task18.SEEDS:
        chain=[j for j in jobs if j['seed']==seed]
        assert [j['cycle'] for j in chain]==list(range(1,10))
        assert chain[0]['parent'] is None
        assert all(b['parent']==a['job_id'] for a,b in zip(chain,chain[1:]))
    task18.submit('00:20:00')
    assert len(requested)==27


def test_release_all_no_seed_default(monkeypatch):
    seen=[]
    monkeypatch.setattr(task18,'release',lambda *args:seen.append(args))
    monkeypatch.setattr(sys,'argv',['task18.py','release','--all'])
    task18.main()
    assert seen==[(True,None,None)]
