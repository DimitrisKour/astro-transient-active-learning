"""Synthetic, isolated full-historical caching: no SLURM, real data or GPU."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from al_pipeline.astronomy.fusion_image.cache_v2 import ShardedHistoricalNPZDataset
from al_pipeline.astronomy.fusion_image.contracts import load_rows, sha256


def module():
    script = Path(__file__).resolve().parents[3] / 'scripts/research/fusion_image/cache/fusion_full_historical_cache_v1.py'
    spec = importlib.util.spec_from_file_location('fusion_full_historical_cache_tested', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_rows(path, columns, rows):
    with path.open('w', newline='') as stream:
        out = csv.DictWriter(stream, columns)
        out.writeheader()
        out.writerows(rows)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    pkgroot = tmp_path/'fusion_model_basics'
    pkgroot.mkdir()
    for name in ('__init__.py', 'dataset.py', 'preprocessing.py'):
        (pkgroot/name).write_text('synthetic source')
    pkg = types.ModuleType('fusion_model_basics')
    pkg.__file__ = str(pkgroot/'__init__.py')
    dataset = types.ModuleType('fusion_model_basics.dataset')
    preprocess = types.ModuleType('fusion_model_basics.preprocessing')

    class FakeDataset:
        def _build_metadata_tensor(self, meta, cols, jd, fid, prov):
            return torch.from_numpy(np.stack([meta[:, 0]+i for i in range(55)], axis=1).astype('float32'))
    dataset.AlertStreamFusionDataset = FakeDataset
    dataset._as_str_list = lambda a: [str(v) for v in a]
    preprocess._prepare_astrominn_image = lambda a, cfg, augment=False: torch.from_numpy(np.asarray(a).astype('float32').copy()*2)
    for name, item in (('fusion_model_basics', pkg),
                       ('fusion_model_basics.dataset', dataset),
                       ('fusion_model_basics.preprocessing', preprocess)):
        monkeypatch.setitem(sys.modules, name, item)

    new_index = tmp_path/'index.csv'
    train = tmp_path/'original_full_train.csv'
    val = tmp_path/'fixed_validation.csv'
    l0 = tmp_path/'original_small_al_l0.csv'
    reg = tmp_path/'registry.toml'
    reg.write_text('synthetic = true')
    rows_train, rows_val, rows_index = [], [], []
    labels = ['AGN', 'CV', 'SN I', 'SN II', 'TDE']
    for i in range(7):
        name = f'o{i}'
        path = tmp_path/f'{name}.npz'
        images = np.ones((2, 3, 63, 63), dtype=np.float32) * (i+1)
        np.savez_compressed(path, images=images,
             meta_data=np.asarray([[i+1], [i+2]], dtype='float32'),
             meta_columns=np.asarray(['magpsf']), jd=np.asarray([1., 2.]),
             fid=np.asarray([1, 2]),
             provenance=np.asarray([{'alert_idx':0},{'alert_idx':1}], dtype=object))
        rows_index.append({'object_id': name, 'filepath':str(path)})
        source = rows_train if i < 5 else rows_val
        for stage in (0, 1):
            source.append({'obj_id':name, 'alert':str(stage), 'target':labels[i % 5]})
    # Preserve duplicated sample rows, rather than deduplicating the source manifest.
    rows_train.append(dict(rows_train[0]))
    l0_rows = rows_train[:3]
    for path, rows in ((train, rows_train), (val, rows_val), (l0, l0_rows)):
        write_rows(path, ['obj_id', 'alert', 'target'], rows)
    write_rows(new_index, ['object_id', 'filepath'], rows_index)
    audit = tmp_path/'audit.json'
    audit.write_text(json.dumps({
        'full_historical_npz_coverage':True,
        'all_historical_labels_verified':True,
        'all_historical_stages_verified':True,
        'stage_probe_passed':True,
        'new_index_sha256':sha256(new_index),
        'historical_splits':{s:{'sample_rows':len(rows)} for s,rows in [('train',rows_train),('validation',rows_val)]},
        'stage_checks':{s:{'checked_rows':len(rows),'compatible_rows':len(rows),'failed_rows':0}
            for s,rows in [('train',rows_train),('validation',rows_val)]},
    }))
    prior = {
        'upstream_repo':str(pkgroot),'model_sha':'gitsha','audit':str(audit),
        'new_index':str(new_index),'registry':str(reg),'l0_manifest':str(l0),
        'historical_train_objects':5,'l0_rows':len(l0_rows),
        'hf_snapshot':str(tmp_path/'hf'),
        'file_hashes':{name:sha256(path) for name,path in (
           ('audit',audit),('new_index',new_index),('registry',reg),
           ('l0_manifest',l0),('train_manifest',train),('validation_manifest',val))},
    }
    plan = tmp_path/'source_plan.json'
    plan.write_text(json.dumps(prior))
    mod = module()
    monkeypatch.setattr(mod.subprocess, 'check_output', lambda *a,**kw:'gitsha\n')
    ds = SimpleNamespace(train_manifest=train, validation_manifest=val,
                         object_id_column='obj_id', label_column='target')
    monkeypatch.setattr(mod.DatasetRegistry, 'from_toml', lambda *a,**kw:
                        SimpleNamespace(resolve_profile=lambda x:SimpleNamespace(dataset=ds)))
    runtime = {'image_encoder_type':'btsbot_convnext','cutout_size':63,
               'image_norm':'none','patch_size':[49,7]}
    monkeypatch.setattr(mod, '_runtime', lambda x:runtime)
    return mod, plan, tmp_path/'cache', runtime, train, l0, audit, rows_train


def test_full_dataset_not_small_l0_and_label_free_runtime(fixture):
    mod, plan, root, runtime, train, l0, audit, rows = fixture
    info = mod.action('prepare', source_plan=plan, cache_root=root, n_shards=2)
    assert info['training_objects'] == 5
    assert info['training_rows'] == len(rows) == 11
    assert info['previous_small_l0_rows'] == 3
    assert info['gpu_jobs_submitted'] == 0
    assert json.loads((root/'plan.json').read_text())['manifests']['train'] == str(train)
    assert mod.action('prepare', source_plan=plan, cache_root=root, n_shards=2) == info
    for i in range(2):
        mod.action('build-shard', source_plan=plan, cache_root=root, n_shards=2, shard_id=i)
    published = mod.action('finalize', source_plan=plan, cache_root=root, n_shards=2)
    assert published['source'] == 'historical_full_train_plus_fixed_validation'
    assert published['labels_saved_in_cache'] is False
    assert published['unique_stages'] == 14
    assert mod.action('status', source_plan=plan, cache_root=root, n_shards=2)['finalized']
    # Source rows remain complete and ordered, including multiplicity.
    all_ds = ShardedHistoricalNPZDataset(rows=load_rows(train),cache_root=root,
              runtime_config=runtime,object_column='obj_id',label_column='target',preload=True)
    assert len(all_ds) == 11
    assert [all_ds[i]['obj_id'] for i in range(11)] == [r['obj_id'] for r in rows]
    assert all_ds[0]['target_index'] == all_ds[-1]['target_index']
    # The same full cache can serve a small initial AL subset or an unlabeled
    # inference pool with label_column=None, without loading hidden labels.
    small_ds = ShardedHistoricalNPZDataset(rows=load_rows(l0),cache_root=root,
              runtime_config=runtime,object_column='obj_id',label_column=None)
    assert len(small_ds) == 3 and 'target_index' not in small_ds[0]


def test_changed_full_source_manifest_fails_before_shards(fixture):
    mod, plan, root, runtime, train, l0, audit, rows = fixture
    train.write_text(train.read_text()+'o0,0,AGN\n')
    with pytest.raises(ValueError, match='FULL train manifest differs'):
        mod.action('prepare', source_plan=plan, cache_root=root, n_shards=2)
    assert not root.exists()


def test_missing_full_stage_audit_fails_closed(fixture):
    mod, plan, root, runtime, train, l0, audit, rows = fixture
    d = json.loads(audit.read_text())
    d['all_historical_stages_verified'] = False
    audit.write_text(json.dumps(d))
    src=json.loads(plan.read_text())
    src['file_hashes']['audit']=sha256(audit)
    plan.write_text(json.dumps(src))
    with pytest.raises(ValueError, match='all_historical_stages_verified'):
        mod.action('prepare', source_plan=plan, cache_root=root, n_shards=2)


def test_no_overwrite_when_full_cache_plan_differs(fixture):
    mod, plan, root, runtime, train, l0, audit, rows = fixture
    mod.action('prepare', source_plan=plan, cache_root=root, n_shards=2)
    with pytest.raises((FileExistsError, ValueError), match='Existing cache plan differs|identity'):
        mod.action('prepare', source_plan=plan, cache_root=root, n_shards=3)
