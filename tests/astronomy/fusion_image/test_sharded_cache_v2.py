"""Synthetic end-to-end tests of resumable, label-free, preprocessing-identical L0 cache."""
from __future__ import annotations
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from al_pipeline.astronomy.fusion_image.cache_v2 import (
    prepare, build_shard, finalize, ShardedHistoricalNPZDataset,
)
from al_pipeline.astronomy.fusion_image.contracts import load_index, load_rows
from al_pipeline.astronomy.fusion_image.data import image_metadata_from_npz


def csv_write(path, rows, columns):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader();w.writerows(rows)


@pytest.fixture
def source(tmp_path,monkeypatch):
    pkgroot=tmp_path/'fusion_model_basics'
    pkgroot.mkdir()
    for p in ('__init__.py','dataset.py','preprocessing.py'):
        (pkgroot/p).write_text('upstream synthetic file')
    pkg=types.ModuleType('fusion_model_basics')
    pkg.__file__=str(pkgroot/'__init__.py')
    ds=types.ModuleType('fusion_model_basics.dataset')
    pre=types.ModuleType('fusion_model_basics.preprocessing')
    class FakeDataset:
        def _build_metadata_tensor(self,meta,cols,jd,fid,prov):
            return torch.as_tensor(
                np.stack([meta[:,0]+i for i in range(55)],axis=1).astype(np.float32))
    ds.AlertStreamFusionDataset=FakeDataset
    ds._as_str_list=lambda arr:[str(x) for x in arr]
    pre._prepare_astrominn_image=lambda im,cfg,augment=False:torch.from_numpy(
        np.asarray(im,dtype=np.float32).copy()*2)
    for name,m in (('fusion_model_basics',pkg),('fusion_model_basics.dataset',ds),
                   ('fusion_model_basics.preprocessing',pre)):
        monkeypatch.setitem(sys.modules,name,m)
    rows={'train':[],'validation':[]}
    index_rows=[]
    for obj,split,alert_ids in [('A','train',[0,2]),('B','train',[0,1]),
                                ('C','validation',[0,3])]:
        file=tmp_path/f'{obj}.npz'
        images=np.full((2,3,63,63), ord(obj), dtype=np.float32)
        images[1] += 1
        np.savez_compressed(file,images=images,meta_data=np.asarray([[1.],[4.]],dtype=np.float32),
          meta_columns=np.asarray(['magpsf']),jd=np.asarray([1.,2.]),fid=np.asarray([1,2]),
          provenance=np.asarray([{'alert_idx':i} for i in alert_ids],dtype=object))
        rows[split].extend({'obj_id':obj,'alert':str(k),'target':'AGN'} for k in alert_ids)
        index_rows.append({'object_id':obj,'filepath':str(file)})
    rows['train'].append(dict(rows['train'][0]))
    manifests={}
    for split in rows:
        path=tmp_path/f'{split}.csv'
        csv_write(path,rows[split],['obj_id','alert','target'])
        manifests[split]=path
    index=tmp_path/'index.csv'
    csv_write(index,index_rows,['object_id','filepath'])
    cfg={'image_encoder_type':'btsbot_convnext', 'image_norm':'none',
         'cutout_size':63, 'patch_size':[49,7]}
    return manifests,index,cfg,tmp_path


def build(source):
    manifests,index,cfg,root=source
    dest=root/'cache2'
    p=prepare(cache_root=dest,manifests=manifests,index_csv=index,runtime_config=cfg,
       id_col='obj_id',label_col='target',n_shards=2,upstream_sha='abc')
    return p,dest


def test_sharded_resume_exactness_and_multiplicities(source):
    manifests,index,cfg,root=source
    plan,dest=build(source)
    assert plan['objects']==3
    assert plan['unique_alert_stages']==6
    assert plan['manifest_rows']['train']==5  # duplicate row must survive
    res=build_shard(root=dest,shard_id=0,runtime_config=cfg,verify_n=2,progress_every=1)
    assert res['stages']>=2
    res2=build_shard(root=dest,shard_id=0,runtime_config=cfg)
    assert res2==res
    with pytest.raises(FileNotFoundError):
        finalize(dest)
    build_shard(root=dest,shard_id=1,runtime_config=cfg,progress_every=1)
    result=finalize(dest)
    assert result['unique_stages']==6 and result['unique_objects']==3
    assert result['labels_saved_in_cache'] is False
    for split in ('train','validation'):
        rows=load_rows(manifests[split])
        for preload in (True,False):
            ds=ShardedHistoricalNPZDataset(rows=rows,cache_root=dest,
              runtime_config=cfg,object_column='obj_id',label_column='target',preload=preload)
            assert len(ds)==len(rows)
            original=load_index(index)
            for i, r in enumerate(rows):
                md,im,_=image_metadata_from_npz(original[r['obj_id']],int(r['alert']),runtime_config=cfg)
                assert torch.equal(md,ds[i]['metadata'])
                assert torch.equal(im,ds[i]['image'])
                assert ds[i]['target_index']==0
    assert (dest/'shard_000'/'done.json').is_file()
    assert (dest/'shard_001'/'done.json').is_file()
    assert prepare(cache_root=dest,manifests=manifests,index_csv=index,runtime_config=cfg,
       id_col='obj_id',label_col='target',n_shards=2,upstream_sha='abc')==plan
    assert finalize(dest)==result


def test_cache_fails_closed_on_changed_preprocess_or_unseen_alert(source):
    manifests,index,cfg,root=source
    _,dest=build(source)
    build_shard(root=dest,shard_id=0,runtime_config=cfg)
    build_shard(root=dest,shard_id=1,runtime_config=cfg)
    finalize(dest)
    rows=load_rows(manifests['train'])
    with pytest.raises(ValueError,match='preprocessing mismatch'):
        ShardedHistoricalNPZDataset(rows=rows,cache_root=dest,runtime_config={**cfg,'image_norm':'L2'},
                                    object_column='obj_id',label_column='target')
    rows[0]['alert']='777'
    with pytest.raises(ValueError,match='Stage not in'):
        ShardedHistoricalNPZDataset(rows=rows,cache_root=dest,runtime_config=cfg,
                                    object_column='obj_id',label_column='target')


def test_source_manifest_hash_change_rejected(source):
    manifests, index, cfg, root = source
    _, dest = build(source)
    # Never reuse a cache plan after the underlying input split changes.
    with manifests['train'].open('a') as stream:
        stream.write('A,0,AGN\n')
    with pytest.raises(ValueError, match='Changed historical train manifest'):
        build_shard(root=dest, shard_id=0, runtime_config=cfg)


def test_missing_manifest_never_published(source):
    manifests, index, cfg, root = source
    _, dest = build(source)
    build_shard(root=dest, shard_id=0, runtime_config=cfg)
    assert not (dest / 'cache_manifest.json').exists()
