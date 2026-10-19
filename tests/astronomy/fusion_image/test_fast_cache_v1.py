"""Small sharded source -> consolidated cache invariants, no GPU or upstream data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from al_pipeline.astronomy.fusion_image import fast_cache
from al_pipeline.astronomy.fusion_image.contracts import sha256


def _source(tmp_path):
    root = tmp_path / 'original'
    root.mkdir()
    plan = {'n_shards':2,'manifest_sha256': {'train':'a','validation':'b'}}
    (root/'plan.json').write_text(json.dumps(plan))
    final = {
        'source':fast_cache.SOURCE, 'cache_version':2,'labels_saved_in_cache':False,
        'plan_sha256':sha256(root/'plan.json'),'manifest_sha256':plan['manifest_sha256'],
        'manifest_rows': {'train':3,'validation':2},
        'unique_objects':4,'unique_stages':5,
        'preprocessing_identity':{'image_encoder_type':'btsbot_convnext'},
        'shards':{'0':{},'1':{}}
    }
    (root/'cache_manifest.json').write_text(json.dumps(final))
    for j, stages in enumerate([
        [('train','A',0),('validation','C',0),('train','B',0)],
        [('train','A',1),('validation','D',0)]
    ]):
        shard = root / f'shard_{j:03d}'
        shard.mkdir()
        n = len(stages)
        im = np.lib.format.open_memmap(shard/'images.npy',mode='w+',dtype='float32',shape=(n,3,63,63))
        md = np.lib.format.open_memmap(shard/'metadata.npy',mode='w+',dtype='float32',shape=(n,55))
        for k in range(n):
            im[k] = j*10 + k
            md[k] = 0.1+j*10+k
        im.flush();md.flush();del im,md
        with (shard/'stage_index.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(('split','object_id','alert','position'))
            for k,(split,oid,alert) in enumerate(stages):w.writerow((split,oid,alert,k))
        (shard/'done.json').write_text(json.dumps({
            'plan_sha256':final['plan_sha256'],'stages':n,
            'index_sha256':sha256(shard/'stage_index.csv')}))
    return root, final, plan


def test_consolidated_full_cache_exact_rows_and_label_separation(tmp_path, monkeypatch):
    source,final,plan = _source(tmp_path)
    monkeypatch.setattr(fast_cache, 'read_identity', lambda src:(final,plan))
    target = tmp_path/'nvme'
    created = fast_cache.build(source=source,target=target,batch_rows=2)
    assert created['unique_stages']==5
    assert created['labels_saved_in_cache'] is False
    assert created['manifest_rows']=={'train':3,'validation':2}
    assert fast_cache.validate(source=source,target=target,samples_per_shard=9)['matched_original_shard_samples']==5
    assert not (target/'labels.npy').exists()
    rows=[{'obj_id':'A','alert':'1','target':'TDE'},
          {'obj_id':'A','alert':'0','target':'AGN'},
          {'obj_id':'A','alert':'1','target':'TDE'},
          {'obj_id':'B','alert':'0','target':'SN I'}]
    ds=fast_cache.FastHistoricalDataset(rows=rows,cache_root=target,split='train',
         object_column='obj_id',label_column='target')
    assert len(ds)==4
    assert [ds[i]['target_index'] for i in range(4)]==[1,0,1,2]
    assert ds[0]['image'][0,0,0].item()==pytest.approx(10)
    assert ds[1]['image'][0,0,0].item()==pytest.approx(0)
    assert ds[0]['image'].equal(ds[2]['image'])
    no_labels=fast_cache.FastHistoricalDataset(rows=[{'obj_id':'A','alert':'1'}],cache_root=target,
             split='train',object_column='obj_id',label_column=None)
    assert 'target_index' not in no_labels[0]
    with pytest.raises(KeyError):
        fast_cache.FastHistoricalDataset(rows=[{'obj_id':'C','alert':'0'}],cache_root=target,
             split='train',object_column='obj_id',label_column=None)
    with pytest.raises(FileExistsError):
        fast_cache.build(source=source,target=target)
