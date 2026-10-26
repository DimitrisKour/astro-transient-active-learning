import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[2]
P=SOURCE/'src/al_pipeline/core/advanced_acquisition/fisher_batch.py'
spec=importlib.util.spec_from_file_location('fisher_batch_test',P)
mod=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)


def data(seed=1,n=27,a=12,d=8,c=5):
    r=np.random.default_rng(seed)
    return mod.FisherBatchInput(tuple(str(i) for i in range(n)),
       r.dirichlet(np.ones(c),size=n),r.normal(size=(n,d)),
       r.dirichlet(np.ones(c),size=a),r.normal(size=(a,d)),rank=12,ridge=1.0)


def test_determinism_and_unique_batch():
    a=data();b=data()
    x,dx=a.engine.select(17)
    y,dy=b.engine.select(17)
    assert np.array_equal(x,y)
    assert np.array_equal(dx,dy)
    assert len(set(x))==17
    assert np.isfinite(dx).all() and (dx>=0).all()


def test_greedy_changes_candidate_value():
    a=data(n=15)
    initial=a.engine.initial_gains()
    chosen,gain=a.engine.select(2)
    assert np.isclose(gain[0],initial.max())
    assert gain[1] <= initial[chosen[1]] + 1e-10


def test_invalid_probabilities_rejected():
    a=data()
    probs=a.probabilities.copy();probs[0]=0
    with pytest.raises(ValueError):
        mod.FisherBatchInput(a.candidate_ids,probs,a.embeddings,a.labeled_probabilities,a.labeled_embeddings)


def test_zero_and_full_budget():
    a=data(n=11)
    assert a.engine.select(0)[0].size==0
    assert len(set(a.engine.select(11)[0]))==11
    with pytest.raises(ValueError):a.engine.select(12)


def test_labeled_fisher_changes_acquisition():
    a=data(n=25,a=8)
    b=mod.FisherBatchInput(a.candidate_ids,a.probabilities,a.embeddings,
       a.labeled_probabilities,a.labeled_embeddings*4,rank=a.rank,ridge=a.ridge)
    assert not np.allclose(a.engine.initial_gains(),b.engine.initial_gains())


def test_no_hidden_candidate_labels_in_input():
    assert 'hidden_labels' not in mod.FisherBatchInput.__dataclass_fields__
    assert 'true_labels' not in mod.FisherBatchInput.__dataclass_fields__


def test_rank_and_provenance_fixed():
    a=data()
    assert a.projection_seed==271828
    with pytest.raises(ValueError):
        mod.FisherBatchInput(a.candidate_ids,a.probabilities,a.embeddings,
           a.labeled_probabilities,a.labeled_embeddings,rank=1)
