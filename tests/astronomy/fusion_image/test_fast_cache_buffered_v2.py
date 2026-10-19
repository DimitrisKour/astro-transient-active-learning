"""Regression tests for buffered, non-mmap cache consolidation."""
from __future__ import annotations
import csv
import json
import numpy as np
import pytest

from al_pipeline.astronomy.fusion_image import fast_cache
from al_pipeline.astronomy.fusion_image.contracts import sha256


def create_source(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    plan = {'n_shards': 2, 'manifest_sha256': {'train': 'sha-t', 'validation': 'sha-v'}}
    (root/'plan.json').write_text(json.dumps(plan))
    final = {'source': fast_cache.SOURCE, 'cache_version': 2, 'labels_saved_in_cache': False,
             'plan_sha256': sha256(root/'plan.json'), 'manifest_sha256': plan['manifest_sha256'],
             'manifest_rows': {'train': 3, 'validation': 2}, 'unique_objects': 4,
             'unique_stages': 5, 'preprocessing_identity': {'image_encoder_type': 'btsbot_convnext'},
             'shards': {'0': {}, '1': {}}}
    (root/'cache_manifest.json').write_text(json.dumps(final))
    for j, records in enumerate([
        [('train','A',0),('validation','C',0),('train','B',0)],
        [('train','A',1),('validation','D',0)]
    ]):
        sr = root / f'shard_{j:03d}'
        sr.mkdir()
        images = np.zeros((len(records),3,63,63),dtype=np.float32)
        md = np.zeros((len(records),55),dtype=np.float32)
        for k in range(len(records)):
            images[k] = j*5+k
            md[k] = j*5+k+0.5
        np.save(sr/'images.npy', images)
        np.save(sr/'metadata.npy', md)
        with (sr/'stage_index.csv').open('w', newline='') as f:
            w=csv.writer(f); w.writerow(['split','object_id','alert','position'])
            for k, rec in enumerate(records): w.writerow((*rec,k))
        (sr/'done.json').write_text(json.dumps({'plan_sha256': final['plan_sha256'],
            'stages': len(records), 'index_sha256': sha256(sr/'stage_index.csv')}))
    return root, final, plan


def test_no_memory_mapped_writes_or_reads_during_build(tmp_path, monkeypatch):
    src, final, plan = create_source(tmp_path)
    monkeypatch.setattr(fast_cache,'read_identity', lambda path:(final,plan))
    def deny_memmap(*args, **kwargs):
        raise AssertionError('build must not use mmap')
    monkeypatch.setattr(np.lib.format,'open_memmap',deny_memmap)
    target=tmp_path/'published'
    manifest=fast_cache.build(source=src,target=target,batch_rows=1)
    assert manifest['writer']=='buffered_npy_sequential_v2'
    assert manifest['labels_saved_in_cache'] is False
    assert manifest['manifest_rows']=={'train':3,'validation':2}
    im=np.load(target/'images.npy',allow_pickle=False)
    md=np.load(target/'metadata.npy',allow_pickle=False)
    assert im.shape==(5,3,63,63)
    assert md.shape==(5,55)
    assert [float(im[i,0,0,0]) for i in range(5)] == [0,1,2,5,6]
    assert [float(md[i,0]) for i in range(5)] == [.5,1.5,2.5,5.5,6.5]
    assert fast_cache.validate(source=src,target=target,samples_per_shard=9)['verified']
    assert not list(tmp_path.glob('.published.building_*'))


def test_invalid_source_does_not_publish(tmp_path, monkeypatch):
    src, final, plan = create_source(tmp_path)
    monkeypatch.setattr(fast_cache,'read_identity', lambda path:(final,plan))
    # Stage CSV is invalid even though its SHA is updated; must refuse publication.
    f=src/'shard_001'/'stage_index.csv'
    rows=f.read_text().replace('D,0,1','D,0,0')
    f.write_text(rows)
    done=src/'shard_001'/'done.json'
    info=json.loads(done.read_text()); info['index_sha256']=sha256(f)
    done.write_text(json.dumps(info))
    dest=tmp_path/'published'
    with pytest.raises(RuntimeError,match='Non-bijective shard positions'):
        fast_cache.build(source=src,target=dest,batch_rows=1)
    assert not dest.exists()
    assert len(list(tmp_path.glob('.published.building_*')))==1
