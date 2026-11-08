"""Three independent Task 8A suite shards reconstruct the combined audit."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'research' / 'historical_replay' / 'task8a'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

runner = load(SCRIPTS / 'task8a.py', 'task8a_runner_shards')
merger = load(SCRIPTS / 'task8a_merge.py', 'task8a_merge_shards')
existing = load(Path(__file__).with_name('test_task8a_v1.py'), 'task8a_existing_tests')


def test_independent_suite_shards_are_equivalent_to_monolithic(tmp_path):
    p, d, root = (tmp_path / x for x in ('persistent', 'data', 'report'))
    existing.populate(p, d, suites=('task4', 'task6b', 'task6c'))
    for key in ('task4', 'task6b', 'task6c'):
        summary = runner.audit(persistent=p, data=d, output=root/'shards'/key,
                               suites=(key,), bootstrap=20, allow_incomplete_secondary=False)
        assert summary['suites'] == [key]
        assert summary['n_adjacent_comparisons'] == 54
    merged = merger.merge(root)
    full = runner.audit(persistent=p, data=d, output=tmp_path/'full',
                        suites=('task4', 'task6b', 'task6c'),
                        bootstrap=20, allow_incomplete_secondary=False)
    assert merged['n_adjacent_comparisons'] == full['n_adjacent_comparisons'] == 162
    assert merged['n_changed_object_records'] == full['n_changed_object_records']
    assert merged['n_first_rejection_cases'] == full['n_first_rejection_cases'] == 72
    assert merged['completed_cycles'] == full['completed_cycles']
    for table in merger.TABLES:
        assert merger.read_rows(root/f'{table}.csv') == merger.read_rows(tmp_path/'full'/f'{table}.csv')


def test_merge_refuses_missing_shard(tmp_path):
    with pytest.raises(FileNotFoundError, match='Incomplete shard'):
        merger.merge(tmp_path)


def test_merge_refuses_tampered_suite(tmp_path):
    root = tmp_path / 'report'
    shard = root / 'shards' / 'task4'
    shard.mkdir(parents=True)
    (shard/'summary.json').write_text(json.dumps({'schema':'task8a_retrospective_acceptance_v1','suites':['task6b']}))
    with pytest.raises(ValueError, match='provenance mismatch'):
        merger.merge(root)
