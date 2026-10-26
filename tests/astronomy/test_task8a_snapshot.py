"""Synthetic AstroMiNN saved-artifact provenance/metadata tests, no GPU."""
from pathlib import Path
import json

import numpy as np
import pytest

from al_pipeline.astronomy.analysis.checkpoint_audit import (
    CLASS_NAMES, SUITES, acquisition_diagnostics, ensure_alignment, read_snapshot,
)


def create_snapshot(persistent, data, *, suite='task4', policy='entropy', seed=135,
                    cycle=0, move_one=False):
    suite_id = SUITES[suite]
    ev = data / 'runs' / suite_id / policy / f'seed_{seed}' / f'cycle_{cycle:03d}' / 'evaluation'
    ev.mkdir(parents=True, exist_ok=True)
    checkpoint = data / 'weights' / f'{suite}_{policy}_{seed}_{cycle}.pth'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'fake-but-existent-synthetic-checkpoint')
    y = np.repeat(np.arange(5), 3)
    logits = np.full((len(y), 5), -4.)
    logits[np.arange(len(y)), y] = 4.
    if move_one:
        logits[3, 1], logits[3, 2] = -4., 5.
    np.savez_compressed(ev / 'validation_predictions.npz', object_ids=np.array([f'obj{i:03d}' for i in range(len(y))]),
                        object_targets=y, object_values=logits)
    (ev / 'evaluation_summary.json').write_text(json.dumps({'checkpoint_path': str(checkpoint),
        'model_class_names': list(CLASS_NAMES), 'splits': [{'split': 'validation', 'object_metrics': {}}]}))
    state = persistent / 'runs' / suite_id / policy / f'seed_{seed}'
    state.mkdir(parents=True, exist_ok=True)
    (state / 'run_state.json').write_text(json.dumps({'last_completed_cycle': 9, 'active_cycle': None}))
    if cycle:
        cdir = state / f'cycle_{cycle:03d}'
        cdir.mkdir(exist_ok=True)
        labels = {f'new{x}': CLASS_NAMES[x % 5] for x in range(10)}
        (cdir / 'cycle_summary.json').write_text(json.dumps({
            'cycle': cycle, 'trained_checkpoint': str(checkpoint),
            'acquisition_size': 10, 'revealed_labels': labels,
            'available_before_selection': 30, 'available_after_selection': 20,
        }))
        manifest = state / 'manifests'
        manifest.mkdir(exist_ok=True)
        (manifest / f'acquired_cycle_{cycle:03d}.csv').write_text(
            'oid,other\n' + ''.join(f'new{x},sample{j}\n' for x in range(10) for j in range(x % 3 + 1)))
    return checkpoint, ev, state


def test_snapshot_and_batch_manifest(tmp_path):
    p, d = tmp_path / 'persistent', tmp_path / 'data'
    create_snapshot(p, d)
    create_snapshot(p, d, cycle=1, move_one=True)
    a = read_snapshot(persistent=p, data=d, suite_key='task4', policy='entropy', seed=135, cycle=0)
    b = read_snapshot(persistent=p, data=d, suite_key='task4', policy='entropy', seed=135, cycle=1)
    ensure_alignment(a, b)
    assert int((a.pred != b.pred).sum()) == 1
    batch = acquisition_diagnostics(persistent=p, suite_key='task4', policy='entropy', seed=135, cycle=1)
    assert batch['n_acquired'] == 10
    assert batch['acquired_TDE'] == 2
    assert batch['n_manifest_rows'] == 19
    assert batch['manifest_status'] == 'verified'


def test_invalid_checkpoint_provenance_rejected(tmp_path):
    p, d = tmp_path / 'p', tmp_path / 'd'
    checkpoint, ev, state = create_snapshot(p, d, cycle=1)
    (state / 'cycle_001' / 'cycle_summary.json').write_text(json.dumps({'cycle': 1, 'trained_checkpoint': 'wrong'}))
    with pytest.raises(ValueError, match='ancestry'):
        read_snapshot(persistent=p, data=d, suite_key='task4', policy='entropy', seed=135, cycle=1)
    (ev / 'evaluation_summary.json').write_text(json.dumps({'checkpoint_path': str(checkpoint),
        'model_class_names': ['CV'] * 5, 'splits': [{'split': 'validation'}]}))
    with pytest.raises(ValueError, match='class-order'):
        read_snapshot(persistent=p, data=d, suite_key='task4', policy='entropy', seed=135, cycle=1)


def test_invalid_alignment_rejected(tmp_path):
    p, d = tmp_path / 'p', tmp_path / 'd'
    create_snapshot(p, d, cycle=0)
    _, ev, _ = create_snapshot(p, d, cycle=1)
    with np.load(ev / 'validation_predictions.npz') as f:
        ids, y, z = (f[key] for key in ('object_ids', 'object_targets', 'object_values'))
    ids[0] = 'different'
    np.savez_compressed(ev / 'validation_predictions.npz', object_ids=ids, object_targets=y, object_values=z)
    a = read_snapshot(persistent=p, data=d, suite_key='task4', policy='entropy', seed=135, cycle=0)
    b = read_snapshot(persistent=p, data=d, suite_key='task4', policy='entropy', seed=135, cycle=1)
    with pytest.raises(ValueError, match='IDs'):
        ensure_alignment(a, b)
