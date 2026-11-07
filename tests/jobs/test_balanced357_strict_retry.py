"""CPU-only scientific-contract checks for both balanced357 strict-retry tasks."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / 'src/al_pipeline/astronomy/experiments/balanced357_strict_retry.py'
CONTROLLERS = {
    'row_order': REPO / 'scripts/adaptive/balanced357_row_order_retry.py',
    'seed': REPO / 'scripts/adaptive/balanced357_seed_retry.py',
}


def load_controller(mode):
    spec = importlib.util.spec_from_file_location(f'balanced357_{mode}_test',
                                                   CONTROLLERS[mode])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_core_functions():
    """Extract pure functions without importing the GPU-dependent training stack."""
    tree = ast.parse(ENGINE.read_text(encoding='utf-8'))
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {'strict_accept', 'shuffled_manifest'}]
    scope = {'Path': Path, 'csv': csv, 'np': np, 'Counter': __import__('collections').Counter,
             'os': __import__('os')}
    import hashlib
    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    scope['digest'] = digest
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(ENGINE), 'exec'), scope)
    return scope


@pytest.mark.parametrize('mode', ['row_order', 'seed'])
def test_isolated_single_case_and_frozen_controls(mode):
    m = load_controller(mode)
    assert m.MODE == mode
    assert m.SEED == 357
    assert m.POLICY == 'entropy_balanced_explore'
    assert m.JOB.name.endswith(f'{mode}_retention_retry_v1')
    assert len(m.OVERLAYS) == 7
    assert any('balanced357_strict_retry.py' in v for v in m.OVERLAYS)
    assert all('task8c2_worker' not in v for v in m.OVERLAYS)


@pytest.mark.parametrize('mode', ['row_order', 'seed'])
@pytest.mark.parametrize('first', [2, 4, 6])
def test_fixed_four_retries_and_no_new_base_fit_on_first_failure(mode, first):
    m = load_controller(mode)
    cases = [{'first_rejection': first, 'key': 'balanced357'}]
    stages = m.stage_plan(cases, [0])
    assert len(stages) == 4 + (9-first)*5
    assert [(x['cycle'], x['stage'], x['retry_number'])
            for x in stages[:4]] == [(first,'retry',n) for n in range(1,5)]
    assert all(x['cycle'] != first or x['stage'] == 'retry' for x in stages)
    for cycle in range(first+1,10):
        chunk = [x for x in stages if x['cycle'] == cycle]
        assert [x['stage'] for x in chunk] == ['base'] + ['retry']*4


def test_strict_gate_requires_both_metrics_and_disallows_ties():
    f = load_core_functions()['strict_accept']
    base = {'macro_f1': .60, 'balanced_accuracy': .70}
    assert f(base, {'macro_f1': .61, 'balanced_accuracy': .71})
    assert not f(base, {'macro_f1': .61, 'balanced_accuracy': .70})
    assert not f(base, {'macro_f1': .60, 'balanced_accuracy': .71})
    assert not f(base, {'macro_f1': .72, 'balanced_accuracy': .69})


def test_row_permutation_preserves_duplicate_multiplicities(tmp_path):
    f = load_core_functions()['shuffled_manifest']
    src, dst = tmp_path/'src.csv', tmp_path/'permuted.csv'
    with src.open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['object_id','label'])
        w.writerows([('x'+str(i%15),str(i%3)) for i in range(150)])
    sha = f(src,dst,41004+400)
    assert sha == f(src,dst,41004+400)
    with src.open(newline='') as stream: a = list(csv.reader(stream))
    with dst.open(newline='') as stream: b = list(csv.reader(stream))
    assert a[0] == b[0]
    assert sorted(a[1:]) == sorted(b[1:])
    assert a[1:] != b[1:]


def test_historical_first_failure_is_derived_not_hardcoded(tmp_path):
    m = load_controller('row_order')
    source = tmp_path/'source'
    source.mkdir()
    rows = []
    for cycle in range(10):
        # First strict failure at c4: F1 improves but BA ties.
        metrics = {'f1_macro': 0.2 + cycle/100,
                   'balanced_accuracy': 0.3 + (cycle if cycle < 4 else cycle-1)/100}
        rows.append({'cycle': cycle, 'split':'validation', 'metrics': metrics})
    (source/'evaluation_history.json').write_text(json.dumps(rows))
    result = m.cases_from_reference(source)
    assert len(result) == 1
    assert result[0]['first_rejection'] == 4
    assert result[0]['seed'] == 357


def test_retention_and_retry_code_contains_continuation_contract():
    tree = ast.parse(ENGINE.read_text())
    code = ENGINE.read_text()
    assert "self._finalize(cycle,row,False,None)" in code
    assert "self._finalize(cycle,row,True,number)" in code
    assert 'initial_checkpoint=parent' in code
    assert "self.retry_mode == 'row_order'" in code
    assert "replace(self.trainer.policy,loader_seed=seed)" in code
    assert 'training_manifest_sha256' in code
    assert 'validation_predictions.npz' in code
