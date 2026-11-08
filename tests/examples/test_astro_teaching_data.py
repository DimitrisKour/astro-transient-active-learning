"""Portable, CPU-only tests for astronomical teaching exports."""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from al_pipeline.astronomy.teaching_data import (
    CLASS_NAMES, load_teaching_data, make_synthetic_demo, validate_teaching_data,
)
from importlib.util import module_from_spec, spec_from_file_location

ROOT = Path(__file__).resolve().parents[2]

def _load_script(name):
    spec = spec_from_file_location(name, ROOT / 'scripts' / 'examples' / (name + '.py'))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

export = _load_script('export_astro_school_features').export
export_reports = _load_script('export_astro_school_replay').export_reports


def test_synthetic_demo_is_deterministic_and_clearly_marked():
    a, b = make_synthetic_demo(), make_synthetic_demo()
    np.testing.assert_array_equal(a['features'], b['features'])
    assert 'SYNTHETIC' in str(a['provenance'])
    assert len(a['preview_indices']) >= len(CLASS_NAMES)
    assert all(np.count_nonzero((a['splits'] == split)) >= 100
               for split in ('train', 'validation', 'test'))


def test_loader_refuses_pickle_and_bad_cross_split_ids(tmp_path):
    a = make_synthetic_demo()
    path = tmp_path/'demo.npz'
    np.savez_compressed(path, **a)
    b = load_teaching_data(path)
    assert b['features'].shape == a['features'].shape
    bad = dict(a)
    bad['object_ids'] = bad['object_ids'].copy()
    bad['object_ids'][1] = bad['object_ids'][0]
    with pytest.raises(ValueError, match='unique'):
        validate_teaching_data(bad)
    bad = dict(a)
    bad['features'] = bad['features'].copy()
    bad['features'][0,0] = np.inf
    with pytest.raises(ValueError, match='Non-finite'):
        validate_teaching_data(bad)


def test_small_real_export_works_without_pytorch(tmp_path):
    # Artificial source files exercise the same manifest/sample schema; not
    # assertions about the statistics of a real astronomical dataset.
    dir = tmp_path/'samples'
    dir.mkdir()
    manifests = {}
    names = ('AGN', 'TDE', 'SN Ia', 'SN IIn', 'Cataclysmic')
    rng = np.random.default_rng(99)
    for split in ('train', 'validation', 'test'):
        path = tmp_path/f'{split}.csv'
        manifests[split] = path
        with path.open('w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=('object_id','label','alert','filepath'))
            writer.writeheader()
            for class_idx, label in enumerate(names):
                for index in range(5):
                    oid = f'{split}-{class_idx}-{index}'
                    sample_path = dir/f'{oid}.npy'
                    raw = {'obj_id':oid, 'alerte':2,
                           'metadata': rng.normal(size=34).astype(np.float32),
                           'image': rng.normal(size=(3,63,63)).astype(np.float32)}
                    np.save(sample_path, raw, allow_pickle=True)
                    writer.writerow({'object_id':oid,'label':label,'alert':2,
                                     'filepath':sample_path.name})
    args = SimpleNamespace(train_manifest=manifests['train'],
                           validation_manifest=manifests['validation'],
                           test_manifest=manifests['test'],
                           train_objects=25, validation_objects=25, test_objects=25,
                           object_id_column='object_id',file_column='filepath',
                           label_column='label',data_root=dir,seed=135,
                           output=tmp_path/'real.npz',overwrite=False)
    export(args)
    exported = load_teaching_data(args.output)
    assert exported['features'].shape[1] == 3*8*8 + 34
    assert sum(exported['splits'] == 'train') == 25
    assert exported['preview_images'].shape[1:] == (3,32,32)
    assert 'not neural-network embeddings' in str(exported['provenance'])
    assert args.output.with_suffix('.provenance.json').exists()


def test_replay_export_filters_validation_without_inventing_cycles(tmp_path):
    src = tmp_path/'report'
    src.mkdir()
    with (src/'evaluation_records.csv').open('w',newline='') as fp:
        fields = ['suite','initialization','strategy','seed','cycle','split','metric',
                  'value','n_labeled','n_acquired_total']
        writer = csv.DictWriter(fp,fieldnames=fields);writer.writeheader()
        for split, metric in [('validation','f1_macro'),('test','f1_macro'),
                              ('validation','balanced_accuracy'),('validation','accuracy')]:
            writer.writerow(dict(suite='B',initialization='warm_start',strategy='entropy',
                                 seed=135,cycle=0,split=split,metric=metric,value=.62,
                                 n_labeled=476,n_acquired_total=0))
    output = tmp_path/'out'
    export_reports(src,output)
    with (output/'astro_historical_evaluation.csv').open() as fp:
        records=list(csv.DictReader(fp))
    assert len(records)==2
    assert set(r['metric'] for r in records)=={'f1_macro','balanced_accuracy'}
    assert all(r['split']=='validation' for r in records)