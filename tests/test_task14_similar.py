"""CPU-only Task14 SIMILAR mathematical and safety tests."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
# Use the checked-out package; integration tests create patched frozen copies.
sys.path.insert(0,str(ROOT/'scripts/adaptive'))
from al_pipeline.core.advanced_acquisition.similar import SimilarInput, SimilarScorer, SimilarSelector
from al_pipeline.core.pool import Pool


def make_input(policy,seed=7,n=38):
    rng=np.random.default_rng(seed)
    p=rng.dirichlet([1,1,1,1,1],size=n)
    h=rng.normal(size=(n,17))
    lp=rng.dirichlet([1,1,1,1,1],size=11)
    lh=rng.normal(size=(11,17))
    return SimilarInput(candidate_ids=tuple(f'U_{i:03d}' for i in range(n)),
         candidate_probabilities=p,candidate_embeddings=h,policy=policy,
         reference_ids=() if policy=='similar_fl' else tuple(f'L_{i:03d}' for i in range(11)),
         reference_probabilities=None if policy=='similar_fl' else lp,
         reference_embeddings=None if policy=='similar_fl' else lh,
         reference_labels=('SN I','SN II','AGN','CV','TDE','SN I','AGN','CV','TDE','SN I','SN II')
                 if policy=='similar_flqmi_rare' else (),
         rare_classes=('CV','TDE') if policy=='similar_flqmi_rare' else ())


def brute_greedy(inp,k,eligible=None):
    matrix,base=inp.build()
    covered=np.zeros(len(matrix),dtype=np.float32)
    relevance=matrix.max(axis=0) if inp.policy=='similar_flqmi_rare' else np.zeros(matrix.shape[1])
    chosen=[];gain=[]
    elig=set(range(len(inp.candidate_ids))) if eligible is None else set(eligible)
    for _ in range(k):
        best=None
        for idx in sorted(elig-set(chosen)):
            effective=np.maximum(covered,base) if base is not None else covered
            marginal=(np.maximum(matrix[:,idx]-effective,0).sum(dtype=np.float64)
                      +float(relevance[idx]))
            if best is None or marginal>best[0]:best=(marginal,idx)
        value,row=best
        chosen.append(row);gain.append(value)
        covered=np.maximum(covered,matrix[:,row])
    return chosen,np.array(gain)


@pytest.mark.parametrize('policy',['similar_fl','similar_flcg','similar_flqmi_rare'])
def test_matches_full_greedy(policy):
    data=make_input(policy)
    scores=SimilarScorer().score(data)
    outcome=SimilarSelector(data).select(ids=data.candidate_ids,scores=scores,k=18,
             pool=Pool.from_ids(data.candidate_ids))
    naive,values=brute_greedy(data,18)
    assert outcome.indices.tolist()==naive
    np.testing.assert_allclose(outcome.scores,values,rtol=1e-5,atol=1e-5)
    assert len(set(outcome.ids))==18


@pytest.mark.parametrize('policy',['similar_fl','similar_flcg','similar_flqmi_rare'])
def test_deterministic_and_only_unlabeled(policy):
    data=make_input(policy)
    ids=data.candidate_ids
    pool=Pool.from_ids(ids,labeled_ids=ids[:5])
    scores=SimilarScorer().score(data)
    a=SimilarSelector(data).select(ids=ids,scores=scores,k=10,pool=pool)
    b=SimilarSelector(data).select(ids=ids,scores=scores,k=10,pool=pool)
    assert a.ids==b.ids
    assert not set(a.ids)&set(ids[:5])
    assert all(x in set(ids[5:]) for x in a.ids)


def test_similarity_nonnegative_bounded_and_separate_sets():
    data=make_input('similar_flcg')
    matrix,base=data.build()
    assert matrix.shape==(38,38)
    assert np.all(matrix>=0) and np.max(matrix)<=1.0000001
    assert base.shape==(38,) and np.all(base>=0)
    # No candidate truth is an input to any SIMILAR selector.
    assert not hasattr(data,'candidate_labels')


def test_rare_query_set_only_confirmed_labels():
    data=make_input('similar_flqmi_rare')
    matrix,_=data.build()
    assert matrix.shape[0]==4 # two CV and two TDE in confirmed labels
    assert matrix.shape[1]==len(data.candidate_ids)


def test_unknown_or_overlap_rejected():
    d=make_input('similar_flcg')
    d.reference_ids=(d.candidate_ids[0],)+d.reference_ids[1:]
    with pytest.raises(ValueError,match='intersect'):
        d.__post_init__()
    d=make_input('similar_flqmi_rare')
    d.rare_classes=('unknown',)
    with pytest.raises(ValueError,match='empty'):
        d.__post_init__()


def test_invalid_scoring_and_ids_rejected():
    d=make_input('similar_fl')
    scores=SimilarScorer().score(d)
    with pytest.raises(ValueError,match='own initial'):
        SimilarSelector(d).select(ids=d.candidate_ids,scores=scores+1,k=3,pool=Pool.from_ids(d.candidate_ids))
    with pytest.raises(ValueError,match='alignment'):
        SimilarSelector(d).select(ids=d.candidate_ids[::-1],scores=scores,k=3,pool=Pool.from_ids(d.candidate_ids))


def test_strict_patch_against_uploaded_task11_frozen(tmp_path, task11_frozen_source):
    import ast,shutil
    from task14_patch import patch_source
    archive=task11_frozen_source
    copied=tmp_path/'source'
    shutil.copytree(archive,copied)
    patch_source(copied,ROOT)
    for f in copied.rglob('*.py'):ast.parse(f.read_text(),filename=str(f))
    cfg=(copied/'src/al_pipeline/core/experiments/config.py').read_text()
    assert 'similar_flqmi_rare' in cfg
    runner=(copied/'scripts/run_astrominn_closed_loop.py').read_text()
    assert 'Task14 runner' in runner


def test_submission_held_independent_chains_and_explicit_release(tmp_path, monkeypatch):
    import json
    import task14
    monkeypatch.setattr(task14,'JOB',tmp_path)
    monkeypatch.setattr(task14,'verify',lambda *a,**kw:None)
    for policy in task14.POLICIES:
        f=tmp_path/'pilot'/policy/'seed_135/result.json'
        f.parent.mkdir(parents=True,exist_ok=True)
        f.write_text(json.dumps({'selected_unique':True}))
    commands=[];released=[]
    def fake_check_output(args,*,text):
        commands.append(tuple(args))
        return str(600000+len(commands))
    def fake_run(args,*,check):
        released.append(tuple(args))
        class Result:returncode=0
        return Result()
    monkeypatch.setattr(task14.subprocess,'check_output',fake_check_output)
    monkeypatch.setattr(task14.subprocess,'run',fake_run)
    task14.submit('00:20:00')
    state=task14.load(tmp_path/'submitted.json')
    assert state['state']=='held' and len(state['jobs'])==81
    assert not released
    assert all('--hold' in args for args in commands)
    for policy in task14.POLICIES:
        for seed in task14.SEEDS:
            chain=[j for j in state['jobs'] if (j['policy'],j['seed'])==(policy,seed)]
            assert len(chain)==9 and chain[0]['parent'] is None
            for earlier,later in zip(chain,chain[1:]):
                assert later['parent']==earlier['job_id']
    task14.release('similar_fl',135,1)
    assert released==[('scontrol','release','600001')]
    state=task14.load(tmp_path/'submitted.json')
    assert sum(j['released'] for j in state['jobs'])==1


def test_rare_variant_is_fl2mi_not_fl1mi():
    """Independent set-value check for the documented FL2MI objective."""
    data = make_input('similar_flqmi_rare', seed=21, n=20)
    q_by_u, _ = data.build()
    result = SimilarSelector(data).select(
        ids=data.candidate_ids, scores=SimilarScorer().score(data), k=8,
        pool=Pool.from_ids(data.candidate_ids))
    chosen = result.indices.tolist()
    assert len(chosen) == 8
    # Compute total objective directly from the mathematical definition.
    total = float(q_by_u[:, chosen].max(axis=1).sum(dtype=np.float64)
                  + q_by_u[:, chosen].max(axis=0).sum(dtype=np.float64))
    assert np.isclose(result.scores.sum(), total, rtol=1e-6, atol=1e-6)


def test_task14_release_all_cli_has_no_implicit_seed(monkeypatch):
    """Regression: --all must not conflict with the parser's default --seed."""
    import task14
    observed = []
    monkeypatch.setattr(task14, 'release', lambda *args: observed.append(args))
    monkeypatch.setattr(sys, 'argv', ['task14.py', 'release', '--all'])
    task14.main()
    assert observed == [(None, None, None, True)]


def test_task14_preflight_and_pilot_retain_seed_135_defaults(monkeypatch):
    """Changing the release CLI default must not alter scientific pilot seeds."""
    import task14
    calls = []
    monkeypatch.setattr(task14, 'preflight', lambda seed: calls.append(('preflight', seed)))
    monkeypatch.setattr(task14, 'pilot', lambda policy, seed: calls.append(('pilot', policy, seed)))
    monkeypatch.setattr(sys, 'argv', ['task14.py', 'preflight'])
    task14.main()
    monkeypatch.setattr(sys, 'argv', ['task14.py', 'pilot', 'similar_fl'])
    task14.main()
    monkeypatch.setattr(sys, 'argv', ['task14.py', 'pilot', 'similar_fl', '--seed', '246'])
    task14.main()
    assert calls == [('preflight', 135), ('pilot', 'similar_fl', 135), ('pilot', 'similar_fl', 246)]
