"""Synthetic, GPU-free invariants for pre-Task-0 recipe planning/training.

Run on Delta with the full al-pipeline checkout to exercise imports.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3]/'src/al_pipeline/astronomy/fusion_image/recipe_training.py'
# In installed checkout this file is at tests/astronomy/fusion_image/, i.e.
# repository root is parents[3].


def extract_definitions():
    tree=ast.parse(SOURCE.read_text())
    names={'Recipe','lr_multiplier','augment_images','weight_vector','ParameterEMA'}
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    for node in nodes:
        node.decorator_list=[d for d in node.decorator_list] # preserve dataclass
    code=compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec')
    ns={'dataclass':__import__('dataclasses').dataclass,'Literal':__import__('typing').Literal,
        'math':math, 'torch':torch, 'copy':__import__('copy'), 'Counter':__import__('collections').Counter,
        'np':np,'CLASS_NAMES':('AGN','TDE','SN I','SN II','CV'),
        'label_index':lambda x: {'AGN':0,'TDE':1,'SN I':2,'SN II':3,'CV':4}[x],
        'object_labels':lambda rows,id_col,label_col:{r[id_col]: {'AGN':0,'TDE':1,'SN I':2,'SN II':3,'CV':4}[r[label_col]] for r in rows}}
    exec(code,ns)
    return ns


def test_schedule_exact_endpoints_and_monotonic_cosine():
    m=extract_definitions()
    Recipe=m['Recipe']; factor=m['lr_multiplier']
    c=Recipe(name='constant',max_epochs=80)
    assert all(factor(c,e)==1 for e in range(1,81))
    s=Recipe(name='warmup',lr_schedule='warmup_cosine')
    assert factor(s,1)==pytest.approx(0.1)
    assert factor(s,2)==pytest.approx(0.55)
    assert factor(s,3)==pytest.approx(1.0)
    assert factor(s,80)==pytest.approx(0.05)
    assert all(factor(s,e)>=factor(s,e+1) for e in range(3,80))


def test_transforms_preserve_channel_alignment_shape_and_pixel_multisets():
    m=extract_definitions()
    transform=m['augment_images']
    torch.manual_seed(77)
    base=torch.arange(63*63,dtype=torch.float32).reshape(1,1,63,63)
    image=torch.cat([base,base+10000,base+20000],dim=1).repeat(96,1,1,1)
    out=transform(image)
    assert out.shape==image.shape
    assert torch.equal(out[:,1]-out[:,0],torch.full_like(out[:,0],10000))
    assert torch.equal(out[:,2]-out[:,0],torch.full_like(out[:,0],20000))
    ordered=image[0,0].flatten().sort().values
    for i in range(len(out)):
        assert torch.equal(out[i,0].flatten().sort().values,ordered)
    assert not torch.equal(out,image)


def test_recipe_rejects_invalid_schedule():
    Recipe=extract_definitions()['Recipe']
    with pytest.raises(ValueError): Recipe(name='invalid',lr_schedule='warmup_cosine',warmup_epochs=80)
    with pytest.raises(ValueError): Recipe(name='invalid',ema_decay=1)


def test_full_history_gate_and_cases_static():
    script=Path(__file__).resolve().parents[3]/'scripts/adaptive/fusion_pre_task0.py'
    s=script.read_text()
    assert "identity['training_objects'] != 4729" in s
    assert "finalized.get('source') != SOURCE_LABEL" in s
    assert "'cosine_aug_ema'" in s and "'cosine_aug_effnum'" in s
    assert "DEFAULT_CASES = ('constant','cosine','cosine_aug')" in s
    assert "test_access':False" in s


def test_effective_number_uses_unique_objects_not_alert_rows():
    ns=extract_definitions()
    Recipe=ns['Recipe']
    rows=[{'obj_id':'a','target':'AGN'}]*10 + [
        {'obj_id':str(i),'target':c} for i,c in enumerate(('AGN','TDE','SN I','SN II','CV'), start=1)
    ]
    w=ns['weight_vector'](rows, recipe=Recipe(name='eff',class_weighting='effective_number'),
        id_col='obj_id',label_col='target')
    # Two AGN objects versus one in the other classes; ten repeated alert rows
    # for object a must not artificially change effective-number counts.
    beta=0.999
    assert w[0]/w[1]==pytest.approx((1-beta)/(1-beta**2),rel=1e-6)
    assert w.mean()==pytest.approx(1.0)


def test_ema_parameter_average_and_buffer_sync():
    EMA=extract_definitions()['ParameterEMA']
    model=torch.nn.BatchNorm1d(3)
    ema=EMA(model,decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.)
        model.running_mean.fill_(7.)
    ema.update(model)
    assert torch.allclose(ema.model.weight, torch.full_like(model.weight,2.))
    assert torch.allclose(ema.model.running_mean, torch.full_like(model.running_mean,7.))
    assert not ema.model.weight.requires_grad
