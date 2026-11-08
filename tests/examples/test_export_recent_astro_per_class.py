"""Small fixture checks for the archived per-class teaching data exporter."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parent / 'export_recent_astro_per_class.py'
spec = importlib.util.spec_from_file_location('astro_school_per_class_export', SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def fixture(tmp_path):
    root = tmp_path / 'artifact'
    val = tmp_path / 'validation.csv'
    with val.open('w', newline='') as f:
        w = csv.DictWriter(f,fieldnames=['suite','strategy','seed','cycle','split','metric','value','n_acquired_total'])
        w.writeheader()
        for cycle in range(10):
            w.writerow(dict(suite='s',strategy='entropy',seed=135,cycle=cycle,split='validation',metric='f1_macro',value=.5,n_acquired_total=50*cycle))
            folder=root/'runs'/'s'/'entropy'/'seed_135'/f'cycle_{cycle:03d}'/'evaluation'
            folder.mkdir(parents=True)
            folder.joinpath('validation_per_class_metrics.json').write_text(json.dumps({
                'object':{name:{'precision':.5,'recall':.5,'f1':.5,'support':10} for name in export.CLASSES},
                'sample':{}
            }))
    return val,root


def test_export_matches_macro_and_support(tmp_path):
    val,root=fixture(tmp_path)
    records,meta=export.collect(validation_csv=val,artifact_root=root)
    assert len(records)==50
    assert meta['macro_f1_checked']==10
    assert meta['runs']==1
    assert records[0]['class_name']=='AGN'
    assert set(row['class_name'] for row in records)==set(export.CLASSES)


def test_export_rejects_mismatched_metrics(tmp_path):
    val,root=fixture(tmp_path)
    path=root/'runs'/'s'/'entropy'/'seed_135'/'cycle_000'/'evaluation'/'validation_per_class_metrics.json'
    obj=json.loads(path.read_text())
    obj['object']['TDE']['f1']=.1
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='does not reproduce'):
        export.collect(validation_csv=val,artifact_root=root)


def test_export_rejects_missing_classes(tmp_path):
    val,root=fixture(tmp_path)
    path=root/'runs'/'s'/'entropy'/'seed_135'/'cycle_000'/'evaluation'/'validation_per_class_metrics.json'
    obj=json.loads(path.read_text())
    del obj['object']['CV']
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='Unexpected class'):
        export.collect(validation_csv=val,artifact_root=root)


def test_export_rejects_incomplete_trajectory(tmp_path):
    val,root=fixture(tmp_path)
    with val.open('r', newline='') as f:
        rows=list(csv.DictReader(f))
    with val.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys());w.writeheader();w.writerows(rows[:-1])
    with pytest.raises(ValueError, match='Expected cycles'):
        export.collect(validation_csv=val,artifact_root=root)