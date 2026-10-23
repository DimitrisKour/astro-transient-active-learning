"""Task 5 baseline-restart package: CPU-only isolation and ancestry tests."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/adaptive"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


installer = load(SCRIPTS / "install_task5_restart.py", "install_task5_restart")


def checkout_fixture(tmp_path):
    repo = tmp_path / "repo"
    script_dir = repo / "scripts/adaptive"
    script_dir.mkdir(parents=True)
    for name in installer.SCRIPT_NAMES:
        src = SCRIPTS / f"task4_{name}"
        if not src.exists():
            pytest.fail(f"Required Task 4 source missing: {src}")
        shutil.copy2(src, script_dir / src.name)
    (repo / "scripts/run_astrominn_historical_replay.py").write_bytes(
        (REPO / "scripts/run_astrominn_historical_replay.py").read_bytes()
    )
    (repo / "src/al_pipeline").mkdir(parents=True)
    (repo / "src/al_pipeline/__init__.py").write_text("# source snapshot\n")
    return repo


def test_install_is_isolated_and_idempotent(tmp_path):
    repo = checkout_fixture(tmp_path)
    original = (repo / "scripts/adaptive/task4_common.sh").read_bytes()
    first = installer.install(repo)
    assert len(first) == len(installer.SCRIPT_NAMES)
    assert installer.install(repo) == first
    assert (repo / "scripts/adaptive/task4_common.sh").read_bytes() == original
    common = (repo / "scripts/adaptive/task5_common.sh").read_text()
    assert '--training-initialization restart_from_baseline' in common
    assert 'astrominn_closed_loop_baseline_restart_v1' in common
    assert 'astrominn_closed_loop_warm_v1' not in common
    assert '--training-initialization warm_start' not in common
    builder = (repo / "scripts/adaptive/task5_build_runner.py").read_text()
    assert 'args.training_initialization != "restart_from_baseline"' in builder
    assert 'L0_fixed_baseline_each_cycle' in builder
    assert 'run_astrominn_closed_loop_restart.py' in builder
    assert 'Task 5 must use restart_from_baseline' in builder
    assert 'task5_' in (repo / "scripts/adaptive/task5_prepare.py").read_text()
    assert 'T5_s' in (repo / "scripts/adaptive/task5_enqueue.sh").read_text()
    # No source other than dedicated Task 5 files should change.
    assert not (repo / "scripts/run_astrominn_closed_loop_restart.py").exists()


def test_installer_rejects_modified_generated_script_without_partial_overwrite(tmp_path):
    repo = checkout_fixture(tmp_path)
    installer.install(repo)
    target = repo / "scripts/adaptive/task5_common.sh"
    target.write_text("hand-edited\n")
    other = repo / "scripts/adaptive/task5_schedule.py"
    was = other.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        installer.install(repo)
    assert target.read_text() == "hand-edited\n"
    assert other.read_bytes() == was


def test_dedicated_runner_uses_existing_l0_not_task4_warmstart(tmp_path):
    repo = checkout_fixture(tmp_path)
    installer.install(repo)
    builder = load(repo / "scripts/adaptive/task5_build_runner.py", "task5_builder_fixture")
    source = (repo / "scripts/run_astrominn_historical_replay.py").read_text()
    generated = builder.build_runner(source)
    assert 'if args.training_initialization != "restart_from_baseline"' in generated
    assert 'training_initialization=args.training_initialization' in generated
    assert '"cycle_training_initialization": args.training_initialization' in generated
    assert '"vit_tower": "L0_fixed_baseline_each_cycle"' in generated
    assert 'run_id=f"{args.bootstrap_source_suite_id}_bootstrap_seed{args.seed}"' in generated
    assert '/ args.bootstrap_source_suite_id' in generated
    assert 'if not args.require_existing_bootstrap:' in generated
    assert 'Task 5:' in generated
    subprocess.run([sys.executable, str(repo / "scripts/adaptive/task5_build_runner.py"),
                    '--repo', str(repo)], check=True)
    dest = repo / "scripts/run_astrominn_closed_loop_restart.py"
    assert dest.read_text() == generated
    # The checked-in Experiment B runner is never overwritten.
    assert (repo / "scripts/run_astrominn_historical_replay.py").read_text() == source


def test_freeze_and_routed_registry_do_not_include_task4_jobs(tmp_path):
    repo = checkout_fixture(tmp_path)
    installer.install(repo)
    config = repo / "configs/delta.toml"
    config.parent.mkdir()
    config.write_text('[storage]\npersistent_root = "/old"\ndata_root = "/old-data"\n[case]\nvalue = 1\n')
    prepare = load(repo / "scripts/adaptive/task5_prepare.py", "task5_prepare_fixture")
    persistent = tmp_path / "persistent"
    job = persistent / "closed_loop" / "task5"
    data = tmp_path / "hdd"
    kwargs = dict(repo=repo, job_dir=job, source_registry=config,
                  persistent=persistent, data=data)
    first = prepare.prepare(**kwargs)
    assert prepare.prepare(**kwargs) == first
    assert 'scripts/run_astrominn_closed_loop_restart.py' in first
    assert 'scripts/adaptive/task5_common.sh' in first
    assert 'scripts/adaptive/task4_common.sh' not in first
    assert 'persistent_root = "' + str(persistent) + '"' in (job / 'registry.toml').read_text()
    assert 'data_root = "' + str(data) + '"' in (job / 'registry.toml').read_text()
    assert (job / 'source/scripts/run_astrominn_closed_loop_restart.py').exists()
    (repo / 'src/al_pipeline/__init__.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='Source changed'):
        prepare.prepare(**kwargs)
    (job / 'source/scripts/adaptive/task5_common.sh').write_text('tampered')
    with pytest.raises(ValueError, match='mismatch'):
        prepare.verify_source(job)


def test_scientific_engine_separates_scoring_and_initialization():
    source = (REPO / 'src/al_pipeline/astronomy/experiments/simulated.py').read_text()
    assert 'def _inference_checkpoint(self, cycle: int)' in source
    assert 'def _training_checkpoint(self, cycle: int)' in source
    assert 'if mode == "restart_from_baseline" or cycle == 1:' in source
    assert 'return self.baseline_checkpoint' in source
    assert 'return self._inference_checkpoint(cycle)' in source
    assert 'cycle - 1:03d' in source


def test_report_finds_wrong_checkpoint_at_cycle_two(tmp_path):
    repo = checkout_fixture(tmp_path)
    installer.install(repo)
    report = load(repo / 'scripts/adaptive/task5_report.py', 'task5_report_fixture')
    persistent = tmp_path / 'persist'
    data = tmp_path / 'data'
    suite = 'astrominn_closed_loop_baseline_restart_v1'
    bootstrap = data / 'runs/astrominn_historical_replay_v1/shared/seed_135/bootstrap_training/best_vit_tower.pth'
    bootstrap.parent.mkdir(parents=True)
    bootstrap.touch()
    state = persistent / 'runs' / suite / 'entropy' / 'seed_135'
    artifacts = data / 'runs' / suite / 'entropy' / 'seed_135'
    (state / 'run_state.json').parent.mkdir(parents=True)
    (state / 'run_state.json').write_text(json.dumps({'last_completed_cycle': 2, 'active_cycle': None}))
    previous = bootstrap
    for cycle in (1, 2):
        new = artifacts / f'cycle_{cycle:03d}/training/best_vit_tower.pth'
        new.parent.mkdir(parents=True)
        new.touch()
        (new.parent / 'training_summary.json').write_text(json.dumps({'initial_checkpoint': str(bootstrap)}))
        select = [f'object_{cycle}_{i}' for i in range(473)]
        summary = dict(inference_checkpoint=str(previous),
                       training_initial_checkpoint=str(bootstrap),
                       trained_checkpoint=str(new), selected_ids=select)
        summary_dir = state / f'cycle_{cycle:03d}'
        summary_dir.mkdir(parents=True)
        (summary_dir / 'cycle_summary.json').write_text(json.dumps(summary))
        previous = new
    values, audit = report.collect(persistent, data, suite)
    assert any(a['seed'] == 135 and a['policy'] == 'entropy' and a['cycle'] == 2 for a in audit)
    assert not values
    # Warming from c1 is wrong, even though c2 inference must come from c1.
    c2 = state / 'cycle_002/cycle_summary.json'
    record = json.loads(c2.read_text())
    record['training_initial_checkpoint'] = str(previous)
    c2.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Training baseline ancestry mismatch'):
        report.collect(persistent, data, suite)
    record['training_initial_checkpoint'] = str(bootstrap)
    record['inference_checkpoint'] = str(bootstrap)
    c2.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Inference ancestry mismatch'):
        report.collect(persistent, data, suite)


def test_scheduler_and_shell_scripts(tmp_path):
    repo = checkout_fixture(tmp_path)
    installer.install(repo)
    scheduler = load(repo / 'scripts/adaptive/task5_schedule.py', 'task5_schedule_fixture')
    plan = scheduler.schedule()
    assert len(plan) == 108
    assert len({(j.seed, j.cycle, j.policy) for j in plan}) == 108
    assert {j.walltime for j in plan if j.cycle <= 2} == {'00:15:00'}
    assert {j.walltime for j in plan if j.cycle >= 3} == {'00:20:00'}
    for name in installer.SCRIPT_NAMES:
        if name.endswith(('.sh', '.sbatch')):
            subprocess.run(['bash','-n',str(repo / 'scripts/adaptive' / f'task5_{name}')], check=True)
    assert 'task5_cycle.sbatch' in (repo / 'scripts/adaptive/task5_enqueue.sh').read_text()
