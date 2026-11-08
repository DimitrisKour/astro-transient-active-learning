"""Read-only recent historical run inventory tests."""
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/examples/inspect_recent_astro_runs.py'
spec = importlib.util.spec_from_file_location('inspect_recent_astro_runs', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_inventory_tracks_missing_and_present_evaluations(tmp_path):
    persistent = tmp_path / 'persistent'
    archive = tmp_path / 'archive'
    run = persistent / 'runs' / 'experiment' / 'entropy' / 'seed_135'
    run.mkdir(parents=True)
    (run / 'run_state.json').write_text(json.dumps({'last_completed_cycle': 2, 'active_cycle': None}))
    evalroot = archive / 'runs' / 'experiment' / 'entropy' / 'seed_135'
    for cycle in (0, 2):
        dest = evalroot / f'cycle_{cycle:03d}' / 'evaluation'
        dest.mkdir(parents=True)
        (dest / 'evaluation_summary.json').write_text(json.dumps({'f1_macro': 0.5, 'split': 'validation'}))
    data = mod.inventory(persistent, archive, ('experiment', 'missing'))
    run = data['suites'][0]['runs'][0]
    assert run['last_completed_cycle'] == 2
    assert run['evaluation_cycles'] == [0, 2]
    assert run['missing_evaluation_cycles'] == [1]
    assert run['evaluation_summary_keys'] == ['f1_macro', 'split']
    assert data['suites'][1]['runs'] == []