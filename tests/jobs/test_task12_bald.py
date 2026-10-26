"""Pure numerical and PyTorch tests for coherent MC dropout BALD."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'src/al_pipeline/core/scoring/bald.py'
spec=importlib.util.spec_from_file_location('task12_bald',p)
bald=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=bald
spec.loader.exec_module(bald)


def test_bald_identical_models_zero_information():
    scores=bald.BALDScorer().score(bald.BALDInput(np.tile([[[.7,.2,.1]]],(12,20,1))))
    np.testing.assert_allclose(scores,0,atol=1e-12)


def test_bald_disagreement_positive_and_bounded():
    p=np.array([[[1.,0.],[0.,1.]]])
    scores=bald.BALDScorer().score(bald.BALDInput(p))
    np.testing.assert_allclose(scores,[np.log(2)],atol=1e-12)


def test_bald_predictive_entropy_minus_conditional_entropy():
    rng=np.random.default_rng(123)
    data=rng.dirichlet(np.ones(5),size=(17,20))
    s=bald.BALDScorer().score(bald.BALDInput(data))
    expected=bald.entropy(data.mean(axis=1))-bald.entropy(data).mean(axis=1)
    np.testing.assert_allclose(s,expected,atol=1e-12)
    assert np.all(s>=0)


def test_bald_probability_validation():
    with pytest.raises(ValueError):bald.BALDInput(np.ones((3,20,5)))
    with pytest.raises(ValueError):bald.BALDInput(np.ones((3,1,5))/5)
    with pytest.raises(ValueError):bald.BALDInput(np.full((3,20,5),np.nan))


def dropout_module():
    source=(ROOT/'src/al_pipeline/astronomy/analysis/task12_mc_dropout.py').read_text()
    # Avoid loading actual AstroMiNN dependencies: execute pure mask manager definitions.
    prefix=source.split('def latest_pool_paths(')[0]
    # Strip AstroMiNN imports so this uses only torch/numpy.
    prefix=prefix.replace('from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest','')
    prefix=prefix.replace('from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (\n    AstroMiNNViTPoolDataset, load_astrominn_vit_model,\n    DEFAULT_LEGACY_CODE_DIR,\n)','')
    prefix=prefix.replace('from al_pipeline.core.scoring.bald import BALDInput, BALDScorer','')
    namespace={}
    exec(prefix,namespace)
    return namespace


def test_shared_dropout_mask_per_sample_and_bn_eval():
    ns=dropout_module()
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit_tower=nn.Sequential(nn.BatchNorm1d(32),nn.Dropout(p=.4))
        def forward(self,x):return self.vit_tower(x)
    m=M().eval()
    x=torch.ones(4,32)
    with ns['fixed_vit_dropout_sample'](m,135):
        first=m(x)
        second=m(x)
        assert not m.vit_tower[0].training
        torch.testing.assert_close(first,second)
        torch.testing.assert_close(first[0],first[1])
    with ns['fixed_vit_dropout_sample'](m,246):
        third=m(x)
        assert not torch.equal(first,third)
    assert not m.vit_tower[1].training
    # Original dropout forward is restored after the draw.
    assert m.vit_tower[1].forward.__func__ is nn.Dropout.forward


def test_fails_closed_without_active_dropout():
    ns=dropout_module()
    class M(nn.Module):
        def __init__(self):
            super().__init__();self.vit_tower=nn.Sequential(nn.Dropout(p=0.))
    with pytest.raises(RuntimeError,match='No active ViT'):
        ns['active_vit_dropout'](M())
