"""CPU-only TEST cache scientific invariants; synthetic examples, no real labels."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from al_pipeline.astronomy.fusion_image import test_fastcache_v1 as mod
from al_pipeline.astronomy.fusion_image.contracts import sha256,write_json_atomic


def manifest(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['obj_id','alert','target'])
        w.writeheader();w.writerows(rows)


def test_stage_deduplication_and_no_label_use():
    rows=[{'obj_id':'a','alert':'1','target':'BAD LABEL'},
          {'obj_id':'a','alert':'1','target':'ALSO BAD'},
          {'obj_id':'a','alert':'2','target':'BAD'},
          {'obj_id':'b','alert':'1','target':'BAD'}]
    stages,n=mod._stages(rows,'obj_id')
    assert n==3 and stages=={'a':[1,2],'b':[1]}


def test_full_cpu_cache_consolidation_and_resident_fetch(tmp_path,monkeypatch):
    shards=tmp_path/'shards';final=tmp_path/'complete'
    data=[{'obj_id':'a','alert':'1','target':'AGN'},
          {'obj_id':'a','alert':'1','target':'AGN'},
          {'obj_id':'a','alert':'2','target':'AGN'},
          {'obj_id':'b','alert':'1','target':'CV'},
          {'obj_id':'c','alert':'3','target':'TDE'}]
    test_csv=tmp_path/'test.csv';manifest(test_csv,data)
    idx=tmp_path/'npz_index.csv'
    with idx.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['object_id','filepath'])
        for x in 'abc':w.writerow([x,f'{x}.npz'])
    signature={'image_encoder_type':'btsbot_convnext','test_preprocessing':'synthetic'}
    checks={'test_manifest':str(test_csv.resolve()),'test_manifest_sha256':sha256(test_csv),
            'test_rows':5,'test_objects':3,'test_unique_stages':4,
            'train_manifest_sha256':'trainhash','validation_manifest_sha256':'valhash',
            'test_id_column':'obj_id','test_label_column':'target',
            'new_index':str(idx.resolve()),'new_index_sha256':sha256(idx),
            'source_plan_sha256':'sourcehash','task0_plan_sha256':'task0hash',
            'train_cache_manifest_sha256':'cachehash',
            'preprocessing_identity':signature}
    def fake_plan_inputs(**kwargs):
        return checks,{'a':[1,2],'b':[1],'c':[3]},signature
    monkeypatch.setattr(mod,'plan_inputs',fake_plan_inputs)
    monkeypatch.setattr(mod,'preprocessing_identity',lambda cfg:cfg)
    def sample(obj,alert):
        # Distinct, known, sample-specific values for exact numerical comparison.
        number=ord(obj)-ord('a')+10*alert
        return np.full(55,number,dtype=np.float32),np.full((3,63,63),number,dtype=np.float32)
    def fake_stages(path,alerts,cfg):
        return {alert:sample(Path(path).stem,alert) for alert in alerts}
    monkeypatch.setattr(mod,'read_object_stages',fake_stages)
    import al_pipeline.astronomy.fusion_image.data as data_module
    monkeypatch.setattr(data_module,'image_metadata_from_npz',
                        lambda path,alert,runtime_config:(*sample(Path(path).stem,alert),{}))
    assert mod.prepare(root=shards,task0=tmp_path,source_plan=tmp_path,
                       training_cache=tmp_path,n_shards=2)['unique_stages']==4
    for sid in range(2):mod.build_shard(shards,sid,signature,verify_samples=2)
    metadata=mod.consolidate(root=shards,target=final)
    assert metadata['test_rows']==5 and metadata['test_unique_stages']==4
    assert metadata['labels_saved_in_cache'] is False
    assert mod.validate(root=final,verify_shards=2)['validated']
    assert not any('target' in f.name or 'label' in f.name for f in final.iterdir())
    ds=mod.TestFastDataset(rows=data,root=final,id_col='obj_id',label_col='target')
    info=mod.attach_resident(ds,root=final)
    assert info['array_read_only'] and info['strategy']=='single_host_resident_buffered_load_v1'
    assert not isinstance(ds.images,np.memmap)
    assert len(ds)==5
    assert ds[0]['target_index']==ds[1]['target_index']
    assert torch.equal(ds[0]['image'],ds[1]['image'])
    assert float(ds[2]['image'].mean()) == 20.0
    assert ds[4]['obj_id']=='c' and ds[4]['alert']==3
    original=ds[0]['image'].clone()
    ds[0]['image'][:]=42
    assert torch.equal(ds[0]['image'],original), 'Cache must not allow in-place caller mutation'
    # Report/timing files must never be misinterpreted as test metrics.
    assert not (final/'evaluation_summary.json').exists()
    assert not (final/'test_per_class_metrics.json').exists()
    # Modified same-byte-size index is detected by hash.
    with (final/'stage_index.csv').open('a') as f:f.write('test,FORGED,999,999\n')
    with pytest.raises(ValueError,match='modified'):
        mod.validate(root=final,verify_shards=0)


def test_test_cache_is_explicitly_test_only(tmp_path):
    p=tmp_path/'fast_cache_manifest.json'
    write_json_atomic(p,{'format_version':1,'source':mod.SOURCE,
                         'labels_saved_in_cache':True})
    with pytest.raises(ValueError,match='label-free'):
        mod.validate(root=tmp_path,verify_shards=0)


def test_no_test_metrics_or_training_code_dependency():
    from pathlib import Path
    code=Path('src/al_pipeline/astronomy/fusion_image/test_fastcache_v1.py').read_text()
    assert 'metrics_for_loader' not in code
    assert 'training_summary.json' not in code
    assert 'best_vit_tower.pth' not in code
    inference=Path('scripts/adaptive/task0_vit_test_compare.py').read_text()
    assert 'TestFastDataset' in inference
    assert 'attach_resident' in inference
    assert 'NewImageEvaluator(' not in inference


def test_task0_model_frozen_runners_untouched():
    # Existing Task 0 script files are intentionally *absent* from this patch.
    # They have their own frozen SHA-256 hashes in Task 0 plan.json.
    import zipfile
    # This test checks the source entrypoints shipped on Delta still exist.
    assert Path('scripts/adaptive/task0_full_history.py').is_file()
    assert Path('src/al_pipeline/astronomy/fusion_image/recipe_training_resident.py').is_file()
