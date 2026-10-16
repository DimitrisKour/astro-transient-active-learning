"""Regression checks for the once-per-object full historical stage audit."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from al_pipeline.astronomy.fusion_image import data

RUNTIME = {'image_encoder_type': 'patch', 'patch_size': [49, 7],
           'cutout_size': 63, 'image_norm': 'median'}


@pytest.fixture
def fake_upstream(monkeypatch):
    package = types.ModuleType('fusion_model_basics')
    package.__path__ = []
    ds = types.ModuleType('fusion_model_basics.dataset')
    prep = types.ModuleType('fusion_model_basics.preprocessing')

    class DummyDataset:
        @staticmethod
        def _build_metadata_tensor(_self, meta, cols, jd, fid, provenance):
            assert len(cols) == 1
            out = torch.zeros((len(meta), 55), dtype=torch.float32)
            out[:, 0] = torch.tensor(meta[:, 0])
            return out

    ds.AlertStreamFusionDataset = DummyDataset
    ds._as_str_list = lambda a: [str(x) for x in a.tolist()]
    prep._prepare_astrominn_image = lambda image, cfg, augment=False: (
        torch.from_numpy(np.array(image[:, 7:56, 7:56], copy=True))
    )
    for name, mod in {
        'fusion_model_basics': package,
        'fusion_model_basics.dataset': ds,
        'fusion_model_basics.preprocessing': prep,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def fixture_npz(tmp_path, provenance=(2, 5, 7), *, nan_meta=False):
    n = len(provenance)
    images = np.ones((n, 3, 63, 63), dtype=np.float32)
    meta = np.arange(n, dtype=np.float32).reshape(n, 1)
    if nan_meta:
        meta[-1, 0] = float('nan')
    p = tmp_path / 'object.npz'
    np.savez_compressed(
        p, images=images, meta_data=meta, meta_columns=np.array(['magpsf']),
        jd=np.arange(n, dtype=float), fid=np.ones(n, dtype=int),
        provenance=np.array([{'alert_idx': i} for i in provenance], dtype=object),
    )
    return p


def test_single_read_for_many_alerts_and_original_parity(tmp_path, monkeypatch, fake_upstream):
    p = fixture_npz(tmp_path)
    original_load = data.np.load
    load_count = {'n': 0}

    def counted_load(*args, **kwargs):
        load_count['n'] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(data.np, 'load', counted_load)
    result = data.audit_npz_alert_stages(p, [2, 5, 7], runtime_config=RUNTIME)
    assert result == {2: None, 5: None, 7: None}
    assert load_count['n'] == 1

    # Baseline behavior: one read per historical alert, same success criterion.
    for alert in (2, 5, 7):
        meta, img, prov = data.image_metadata_from_npz(p, alert, runtime_config=RUNTIME)
        assert meta.shape == (55,)
        assert img.shape == (3, 49, 49)
        assert prov['alert'] == alert
    assert load_count['n'] == 4


def test_missing_and_duplicate_provenance_alerts(tmp_path, fake_upstream):
    p = fixture_npz(tmp_path, (2, 2, 7))
    result = data.audit_npz_alert_stages(p, [2, 3, 7], runtime_config=RUNTIME)
    assert 'matches 2 new stages' in result[2]
    assert 'matches 0 new stages' in result[3]
    assert result[7] is None


def test_errors_isolated_to_bad_stage(tmp_path, fake_upstream):
    p = fixture_npz(tmp_path, (2, 5), nan_meta=True)
    result = data.audit_npz_alert_stages(p, [2, 5], runtime_config=RUNTIME)
    assert result[2] is None
    assert 'Nonfinite metadata' in result[5]


def test_object_file_failure_reports_each_requested_alert(tmp_path, fake_upstream):
    missing = tmp_path / 'missing.npz'
    result = data.audit_npz_alert_stages(missing, [2, 5], runtime_config=RUNTIME)
    assert set(result) == {2, 5}
    assert all(isinstance(error, str) and error for error in result.values())


def test_full_cli_groups_rows_by_object_and_writes_strict_report(
    tmp_path, monkeypatch, fake_upstream
):
    import csv
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    # Source is tested directly, but full al-pipeline may not be installed
    # in our isolated local overlay test environment.
    fake_registry = types.ModuleType('al_pipeline.astronomy.data.registry')
    fake_registry.DatasetRegistry = object
    monkeypatch.setitem(sys.modules, 'al_pipeline.astronomy.data.registry', fake_registry)

    script_path = Path(__file__).resolve().parents[3] / 'scripts/adaptive/audit_fusion_image.py'
    spec = importlib.util.spec_from_file_location('optimized_stage_audit_cli', script_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    def rows_csv(path, rows):
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    p_a = fixture_npz(tmp_path, (0, 1, 2))
    p_b = tmp_path / 'b.npz'
    p_c = tmp_path / 'c.npz'
    import shutil
    shutil.copyfile(p_a, p_b)
    shutil.copyfile(p_a, p_c)
    index_csv = tmp_path / 'built_all.csv'
    rows_csv(index_csv, [
        {'object_id':'a','filepath':str(p_a),'label_str':'AGN'},
        {'object_id':'b','filepath':str(p_b),'label_str':'CV'},
        {'object_id':'c','filepath':str(p_c),'label_str':'TDE'},
    ])
    train = tmp_path / 'train.csv'
    val = tmp_path / 'val.csv'
    test = tmp_path / 'test.csv'
    rows_csv(train, [
        {'obj_id':'a','target':'AGN','alert':'0'},
        {'obj_id':'a','target':'AGN','alert':'1'},
        {'obj_id':'a','target':'AGN','alert':'2'},
    ])
    rows_csv(val, [{'obj_id':'b','target':'CV','alert':'0'}])
    rows_csv(test, [{'obj_id':'c','target':'TDE','alert':'1'}])
    monkeypatch.setattr(audit, 'historical_dataset', lambda *_: SimpleNamespace(
        train_manifest=train, validation_manifest=val, test_manifest=test,
        object_id_column='obj_id', label_column='target',
    ))
    count = {'n':0}
    original_load = data.np.load

    def counted_load(*args, **kwargs):
        count['n'] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(data.np, 'load', counted_load)
    output = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        str(script_path), '--registry', str(tmp_path / 'not_used.toml'),
        '--new-index', str(index_csv), '--out-dir', str(output),
        '--model-sha', 'test-sha', '--check-stages', '-1', '--strict-stages'
    ])
    audit.main()
    summary = __import__('json').loads((output/'audit.json').read_text())
    assert summary['all_historical_stages_verified'] is True
    assert summary['stage_probe_passed'] is True
    assert summary['stage_scan']['npz_files_opened'] == 3
    assert summary['stage_scan']['distinct_alert_stages_requested'] == 5
    assert sum(v['checked_rows'] for v in summary['stage_checks'].values()) == 5
    assert count['n'] == 3
