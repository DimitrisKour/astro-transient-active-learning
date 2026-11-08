"""CPU-only regressions for controlled BTSbot L0 matrix (no actual NPZ required)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from al_pipeline.astronomy.fusion_image.backends import weighted_loss_from_logits


def matrix_module():
    import sys
    path = Path(__file__).resolve().parents[3] / 'scripts' / 'archive' / 'fusion_image' / 'early_l0' / 'fusion_l0_matrix.py'
    spec = importlib.util.spec_from_file_location('fusion_l0_matrix_tested', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_one_pass_loss_matches_legacy_double_pass():
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((13, 5)).astype(np.float32)
    ys = rng.integers(0, 5, size=13).tolist()
    weights = torch.tensor([1., 2., 0.4, 1.5, 3.], device='cpu')
    ref = sum(float(torch.nn.functional.cross_entropy(
        torch.tensor(logits[i:i+4]),torch.tensor(ys[i:i+4]),
        weight=weights,reduction='none').mean()) for i in range(0,13,4))/4
    actual = weighted_loss_from_logits(logits,ys,weights,4)
    assert actual == pytest.approx(ref,abs=1e-8)


def test_matrix_exactly_three_seeds_by_two_inits():
    m = matrix_module()
    assert m.RUNS == (('pretrained',135),('pretrained',246),('pretrained',357),
                      ('randomized',135),('randomized',246),('randomized',357))
    assert len(set(m.RUNS)) == 6


def test_report_refuses_missing_fits(tmp_path,monkeypatch):
    m=matrix_module()
    monkeypatch.setattr(m,'verify',lambda _: {'runs':[]})
    with pytest.raises(RuntimeError,match='incomplete'):
        m.report(SimpleNamespace(run_root=tmp_path))
    data=json.loads((tmp_path/'matrix_report.json').read_text())
    assert data['completed_runs']==0


def test_source_list_includes_optimized_validation_and_runner():
    m=matrix_module()
    assert 'src/al_pipeline/astronomy/fusion_image/backends.py' in m.SOURCE_FILES
    assert 'scripts/archive/fusion_image/early_l0/fusion_l0_matrix.py' in m.SOURCE_FILES


def test_prepare_rejects_entire_training_split_as_l0(tmp_path,monkeypatch):
    """No accidental 'L0 = all historical training rows' comparisons."""
    import csv
    import hashlib
    m=matrix_module()
    train=tmp_path/'train.csv'
    valid=tmp_path/'val.csv'
    test=tmp_path/'test.csv'
    pilot_tr=tmp_path/'pilot_train.csv'
    pilot_va=tmp_path/'pilot_val.csv'
    def write(path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['obj_id','alert','target'])
            w.writeheader();w.writerows(rows)
    rows=[{'obj_id':f'o{i}','alert':0,'target':m.CLASS_NAMES[i%5]} for i in range(150)]
    vals=[{'obj_id':f'v{i}','alert':0,'target':m.CLASS_NAMES[i%5]} for i in range(25)]
    tests=[{'obj_id':f't{i}','alert':0,'target':m.CLASS_NAMES[i%5]} for i in range(25)]
    for path, contents in ((train,rows),(valid,vals),(test,tests),
                           (pilot_tr,rows[:5]),(pilot_va,vals[:5])):
        write(path,contents)
    reg=tmp_path/'registry.toml';reg.write_text('test = true')
    idx=tmp_path/'built_all.csv';idx.write_text('object_id,filepath\n')
    hf=tmp_path/'hf';hf.mkdir();(hf/'pytorch_model.bin').write_bytes(b'test weights')
    upstream=tmp_path/'upstream';upstream.mkdir()
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps({'new_model_repository_sha':'fake_sha',
         'new_index_sha256':m.sha256(idx),'new_index':str(idx.resolve()),
         'full_historical_npz_coverage':True,
         'all_historical_labels_verified':True,
         'stage_probe_passed':True,
         'all_historical_stages_verified':True,
         'historical_splits':{s:{'sample_rows':len(rr),'manifest_sha256':m.sha256(p)}
                               for s,p,rr in [('train',train,rows),('validation',valid,vals),('test',test,tests)]},
         'stage_checks':{s:{'checked_rows':len(rr),'compatible_rows':len(rr),'failed_rows':0}
                         for s,rr in [('train',rows),('validation',vals),('test',tests)]}}))
    ds=SimpleNamespace(train_manifest=train,validation_manifest=valid,test_manifest=test,
                       object_id_column='obj_id',label_column='target')
    monkeypatch.setattr(m,'historical_splits',lambda _:ds)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'fake_sha\n')
    args=SimpleNamespace(repo=Path(__file__).resolve().parents[3],
        run_root=tmp_path/'output',registry=reg,audit=audit,new_index=idx,
        hf_snapshot=hf,hf_weights_sha256=m.sha256(hf/'pytorch_model.bin'),
        upstream_repo=upstream,l0=train,pilot_train=pilot_tr,pilot_validation=pilot_va,
        epochs=30,patience=6,batch_size=64,lr=1e-4,class_weight_exponent=.85)
    with pytest.raises(ValueError,match='Unexpected historical L0 object count'):
        m.prepare(args)
    assert not (args.run_root/'plan.json').exists()
