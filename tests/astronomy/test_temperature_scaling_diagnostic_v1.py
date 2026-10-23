"""Pure NumPy checks of the proposed offline diagnostic; no GPU/Delta data."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/adaptive/temperature_scaling_diagnostic_v1.py"
spec = importlib.util.spec_from_file_location("temperature_scaling_diagnostic_v1", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_stratified_split_independent_of_input_row_order():
    ids = np.array([f"id{x}" for x in range(30)])
    y = np.array([x % 5 for x in range(30)])
    a, b = module.split_stratified(ids, y, fraction=.5, seed=135)
    assert len(a) == len(b) == 15
    assert not set(a) & set(b)
    mapping = dict(zip(ids, y))
    rev = np.arange(29, -1, -1)
    c, d = module.split_stratified(ids[rev], y[rev], fraction=.5, seed=135)
    assert set(ids[a]) == set(ids[rev][c])
    assert set(ids[b]) == set(ids[rev][d])
    assert {int(v) for v in mapping.values()} == set(range(5))


def test_split_rejects_rare_singleton():
    with pytest.raises(ValueError, match="fewer than 2"):
        module.split_stratified(np.array(["a", "b", "c"]), np.array([0, 0, 1]), fraction=.5, seed=1)


def test_read_eval_npz_and_duplicate_id_rejection(tmp_path):
    f = tmp_path / 'saved.npz'
    np.savez_compressed(f, object_ids=np.array(['a', 'b']), object_targets=np.array([0, 1]),
                        object_values=np.array([[1,2,3,4,5], [5,4,3,2,1.]], dtype=float))
    ids, y, logits = module.read_validation_npz(f)
    assert len(ids) == len(y) == len(logits) == 2
    with pytest.raises(ValueError, match="unique"):
        module.check_logits(np.array(['a','a']), logits)


def test_temperature_argmax_and_rank_change():
    rng = np.random.default_rng(123)
    logits = rng.normal(size=(48, 5)) * rng.uniform(.1, 9, (48, 1))
    assert np.array_equal(np.argmax(module.softmax(logits), axis=1),
                          np.argmax(module.softmax(logits / 2.3), axis=1))
    x = module.pool_rank_changes(np.array([f"x{i}" for i in range(48)]), logits, 2.3, 10)
    assert 0 <= x['top_k_overlap'] <= 10
    assert x['replaced_objects'] == 10 - x['top_k_overlap']
    assert x['mean_absolute_rank_shift'] >= 0


def test_pool_rejects_probabilities(tmp_path):
    p = tmp_path / 'pool.npz'
    np.savez_compressed(p, object_ids=np.array(['a']), values=np.full((1,5),0.2), output_type=np.array('probabilities'))
    with pytest.raises(ValueError, match="raw logits"):
        module.read_pool_npz(p)


def test_checkpoint_pair_rejects_wrong_identity(tmp_path):
    import json
    e = tmp_path / 'evaluation'; e.mkdir()
    fake = tmp_path / 'one.pth'; fake.write_bytes(b'fake')
    (e/'evaluation_summary.json').write_text(json.dumps({'model_class_names': list(module.ASTROMINN_CLASS_ORDER),
        'checkpoint_path': str(fake), 'checkpoint_sha256': module.file_hash(fake)}))
    s = tmp_path / 'cycle_summary.json'
    s.write_text(json.dumps({'inference_checkpoint': '/tmp/two.pth'}))
    with pytest.raises(ValueError, match="Mismatch"):
        module.verify_checkpoint_pair(e,s)
