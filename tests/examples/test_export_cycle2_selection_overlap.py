"""Minimal offline checks for the read-only archived overlap exporter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/examples/export_cycle2_selection_overlap.py'
spec = importlib.util.spec_from_file_location('cycle2_overlap_export', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def selection(tmp_path, suite, policy, seed, cycle, ids):
    dest = tmp_path/'runs'/suite/policy/f'seed_{seed}'/f'cycle_{cycle:03d}'/'selection.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'ids':ids}),encoding='utf-8')


def test_pairwise_matrix_distinguishes_batch_and_cumulative(tmp_path):
    policies = [('a','s','a'),('b','s','b')]
    for seed in (135,246,357):
        selection(tmp_path,'s','a',seed,1,['a','b'],)
        selection(tmp_path,'s','a',seed,2,['c','d'],)
        selection(tmp_path,'s','b',seed,1,['a','c'],)
        selection(tmp_path,'s','b',seed,2,['d','e'],)
    rows, means, meta = module.collect(tmp_path,policies=policies,seeds=(135,246,357),batch_size=2)
    assert len(rows)==9
    assert len(means)==3
    cross = next(item for item in means if (item['policy_a'],item['policy_b'])==('a','b'))
    assert cross['cycle2_mean']== pytest.approx(1/3)
    assert cross['cumulative_mean']== pytest.approx(3/5)
    assert cross['n_seeds']==3
    assert meta['objects_per_cumulative_selection']==4
    assert all(item['cycle2_mean']==1 and item['cumulative_mean']==1
               for item in means if item['policy_a']==item['policy_b'])


def test_disjoint_cycles_required(tmp_path):
    selection(tmp_path,'s','a',135,1,['a','b'])
    selection(tmp_path,'s','a',135,2,['b','c'])
    with pytest.raises(ValueError,match='repeated selection'):
        module.collect(tmp_path,policies=[('a','s','a')],seeds=(135,),batch_size=2)


def test_duplicates_forbidden(tmp_path):
    selection(tmp_path,'s','a',135,1,['a','a'])
    with pytest.raises(ValueError,match='Duplicate acquisition'):
        module.read_selected(tmp_path,'s','a',135,1,batch_size=2)


def test_jaccard_known_value():
    assert module.jaccard({'a','b','c'},{'b','c','d'}) == (2,4,0.5)