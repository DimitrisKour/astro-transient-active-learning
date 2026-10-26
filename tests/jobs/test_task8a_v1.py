"""Task 8A integration tests: read-only, first-rejection truncation, incomplete secondary."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from al_pipeline.astronomy.analysis.checkpoint_audit import CLASS_NAMES, SUITES

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'adaptive' / 'task8a.py'
spec = importlib.util.spec_from_file_location('task8a_audit_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def add_snapshot(p, d, suite_key, seed, policy, cycle, *, regress=False):
    suite = SUITES[suite_key]
    ev = d / 'runs' / suite / policy / f'seed_{seed}' / f'cycle_{cycle:03d}' / 'evaluation'
    ev.mkdir(parents=True, exist_ok=True)
    weight = d / 'weights' / suite / policy / f'seed_{seed}' / f'{cycle}.pth'
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b'model')
    ids = [f'obj{i}' for i in range(20)]
    y = np.repeat(np.arange(5), 4)
    z = np.full((20, 5), -4.)
    z[np.arange(20), y] = 4.
    if regress:
        z[4, 1], z[4, 0] = -4., 5.  # loses one TDE
    np.savez_compressed(ev / 'validation_predictions.npz', object_ids=np.array(ids),
                        object_targets=y, object_values=z)
    (ev / 'evaluation_summary.json').write_text(json.dumps({
        'model_class_names': list(CLASS_NAMES), 'checkpoint_path': str(weight),
        'splits': [{'split': 'validation', 'object_metrics': {}, 'object_count': 20}]}))
    run = p / 'runs' / suite / policy / f'seed_{seed}'
    run.mkdir(parents=True, exist_ok=True)
    (run / 'run_state.json').write_text(json.dumps({'last_completed_cycle': 9, 'active_cycle': None}))
    if cycle:
        c = run / f'cycle_{cycle:03d}'
        c.mkdir(exist_ok=True)
        (c / 'cycle_summary.json').write_text(json.dumps({
            'cycle': cycle, 'trained_checkpoint': str(weight),
            'acquisition_size': 5, 'revealed_labels': {f'new{x}': CLASS_NAMES[x] for x in range(5)},
        }))
    return ev


def populate(p, d, suites=('task4',), missing_last=()):
    for suite in suites:
        for policy in ('entropy', 'entropy_balanced_explore'):
            for seed in (135, 246, 357):
                last = 8 if (suite, policy, seed) in missing_last else 9
                for cycle in range(last+1):
                    add_snapshot(p, d, suite, seed, policy, cycle, regress=(cycle == 2))


def test_audit_exact_first_rejection_and_preserved_sources(tmp_path):
    p, d, out = (tmp_path / x for x in ('persistent', 'data', 'out'))
    populate(p, d)
    before = {str(path): path.read_bytes() for path in (p / 'runs').rglob('*.json')}
    result = runner.audit(persistent=p, data=d, output=out, suites=('task4',),
                          bootstrap=20, allow_incomplete_secondary=False)
    assert result['n_adjacent_comparisons'] == 54
    assert result['n_first_rejection_cases'] >= 24
    assert result['test_accessed'] is False
    assert result['missing_secondary'] == []
    rows = list(__import__('csv').DictReader((out / 'first_rejections.csv').open()))
    strict = [r for r in rows if r['rule'] == 'strict_macro']
    assert len(strict) == 6
    assert {r['first_rejected_cycle'] for r in strict} == {'2'}
    import csv
    decisions = [r for r in csv.DictReader((out / 'exploratory_rules.csv').open())
                 if r['rule'] == 'strict_macro']
    assert len(decisions) == 12  # 6 trajectories x (accept cycle 1, reject cycle 2)
    assert {r['candidate_cycle'] for r in decisions} == {'1', '2'}
    assert before == {str(path): path.read_bytes() for path in (p / 'runs').rglob('*.json')}


def test_missing_secondary_last_cycle_allowed(tmp_path):
    p, d, out = (tmp_path / x for x in ('p', 'd', 'out'))
    populate(p, d, suites=('task4', 'task6c'), missing_last={('task6c', 'entropy', 135)})
    result = runner.audit(persistent=p, data=d, output=out, suites=('task4', 'task6c'),
                          bootstrap=20, allow_incomplete_secondary=True)
    assert result['n_adjacent_comparisons'] == 107
    assert result['missing_secondary'][0]['first_missing_cycle'] == 9
    with pytest.raises(FileNotFoundError):
        runner.audit(persistent=p, data=d, output=out, suites=('task4', 'task6c'),
                     bootstrap=20, allow_incomplete_secondary=False)


def test_missing_interior_cycle_rejected(tmp_path):
    p, d, out = (tmp_path / x for x in ('p', 'd', 'out'))
    populate(p, d, suites=('task4', 'task6c'))
    f = d / 'runs' / SUITES['task6c'] / 'entropy' / 'seed_135' / 'cycle_005' / 'evaluation' / 'validation_predictions.npz'
    f.unlink()
    with pytest.raises(ValueError, match='interior'):
        runner.audit(persistent=p, data=d, output=out, suites=('task4', 'task6c'),
                     bootstrap=20, allow_incomplete_secondary=True)


def test_output_must_not_be_in_source_runs(tmp_path):
    p, d = tmp_path / 'p', tmp_path / 'd'
    with pytest.raises(ValueError, match='source run'):
        runner.audit(persistent=p, data=d, output=d / 'runs' / 'report',
                     suites=('task4',), bootstrap=20, allow_incomplete_secondary=True)
