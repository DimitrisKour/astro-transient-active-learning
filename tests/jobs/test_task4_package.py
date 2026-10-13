"""CPU-only safety checks; no datasets, cluster, or GPUs required."""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/adaptive"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


builder = load("task4_build_runner")
prep = load("task4_prepare")
check = load("task4_preflight")


def original_stub():
    return '''"""Prepare or run Experiment B: historical AstroMiNN AL replay."""
    parser.add_argument("--require-existing-bootstrap", action="store_true",
                        help="Reuse a validated bootstrap; fail rather than train a missing one.")
    if args.bootstrap_only and not args.run:
        pass
    bootstrap_dir = (
        resolved.storage.data_root
        / "runs"
        / args.suite_id
        / "shared"
        / f"seed_{args.seed}"
        / "bootstrap_training"
    )
    plan = {
        "suite_id": args.suite_id,
        "cycle_training_initialization": "restart_from_random",
        "vit_tower": "random_initialization_each_training_stage",
    }
    bootstrap_request = TrainingRequest(
        run_id=f"{args.suite_id}_bootstrap_seed{args.seed}",
    )
    suite = build_astrominn_simulated_suite(
        training_initialization="restart_from_random",
    )
'''


def test_build_dedicated_warm_runner_without_touching_original():
    orig = original_stub()
    changed = builder.build_runner(orig)
    assert orig != changed
    assert '--bootstrap-source-suite-id' in changed
    assert 'training_initialization=args.training_initialization' in changed
    assert 'run_id=f"{args.bootstrap_source_suite_id}_bootstrap_seed{args.seed}"' in changed
    assert '/ args.bootstrap_source_suite_id' in changed
    assert 'if not args.require_existing_bootstrap:' in changed
    with pytest.raises(ValueError, match="upstream anchor"):
        builder.build_runner(changed)


def test_source_freeze_is_reusable_and_detects_changes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts/adaptive").mkdir(parents=True)
    (repo / "src/al_pipeline").mkdir(parents=True)
    (repo / "scripts/run_astrominn_historical_replay.py").write_text(original_stub())
    (repo / "src/al_pipeline/__init__.py").write_text("# ready\n")
    for p in SCRIPTS.glob("task4_*.*"):
        (repo / "scripts/adaptive" / p.name).write_bytes(p.read_bytes())
    (repo / "configs").mkdir()
    registry = repo / "configs/delta.toml"
    registry.write_text('[storage]\npersistent_root = "/old"\ndata_root = "/old2"\n[settings]\na = 1\n')
    persistent = tmp_path / "persistent"
    data = tmp_path / "data"
    job = persistent / "job"
    kwargs = dict(repo=repo, job_dir=job, source_registry=registry, persistent=persistent, data=data)
    first = prep.prepare(**kwargs)
    assert prep.prepare(**kwargs) == first
    assert (job / "source/scripts/run_astrominn_closed_loop.py").exists()
    assert f'persistent_root = "{persistent}"' in (job / "registry.toml").read_text()
    (repo / "src/al_pipeline/__init__.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="Source changed"):
        prep.prepare(**kwargs)
    frozen_file = job / "source/src/al_pipeline/__init__.py"
    frozen_file.write_text("# tampered")
    with pytest.raises(ValueError, match="mismatch"):
        prep.verify_source(job)


def write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def test_split_comparison_allows_pool_ids_but_not_data_differences(tmp_path):
    for name in ('baseline','task4'):
        folder = tmp_path / 'runs' / name / 'shared' / 'historical_replay_split'
        folder.mkdir(parents=True)
        summary = dict(split_seed=135,initial_fraction=.1,minimum_initial_per_class=5,
                       historical_objects=20,historical_samples=25,
                       initial_objects=2,initial_samples=3,candidate_objects=18,
                       candidate_samples=22,maximum_alert=10)
        (folder/'split_summary.json').write_text(json.dumps(summary))
        write_csv(folder/'initial_labeled_manifest.csv', ['object_id','label'],
                  [{'object_id':'a','label':'AGN'},{'object_id':'a','label':'AGN'}])
        write_csv(folder/'hidden_labels.csv',['object_id','target_label'],
                  [{'object_id':'b','target_label':'SN I'}])
        write_csv(folder/'candidate_pool/candidate_pool.csv', ['pool_id','object_id','filepath'],
                  [{'pool_id':name,'object_id':'b','filepath':'/somewhere/1.npy'}])
    check.check_equivalent_splits(tmp_path, 'baseline', 'task4')
    modified = tmp_path/'runs/task4/shared/historical_replay_split/hidden_labels.csv'
    write_csv(modified,['object_id','target_label'],[{'object_id':'b','target_label':'TDE'}])
    with pytest.raises(ValueError, match="hidden_labels"):
        check.check_equivalent_splits(tmp_path, 'baseline', 'task4')


def test_shell_syntax():
    for f in SCRIPTS.glob('task4_*.sh'):
        subprocess.run(['bash', '-n', str(f)], check=True)
    for f in SCRIPTS.glob('task4_*.sbatch'):
        subprocess.run(['bash', '-n', str(f)], check=True)


def test_report_audits_checkpoint_chain_and_duplicate_selections(tmp_path):
    report = load('task4_report')
    persist = tmp_path/'persist'
    data = tmp_path/'data'
    suite='astrominn_closed_loop_warm_v1'
    base = data/'runs/astrominn_historical_replay_v1/shared/seed_135/bootstrap_training/best_vit_tower.pth'
    base.parent.mkdir(parents=True)
    base.touch()
    s = persist/'runs'/suite/'entropy'/'seed_135'
    a = data/'runs'/suite/'entropy'/'seed_135'
    (s/'cycle_001').mkdir(parents=True)
    (s/'run_state.json').write_text(json.dumps({'last_completed_cycle':1,'active_cycle':None}))
    trained = a/'cycle_001/training/best_vit_tower.pth'
    trained.parent.mkdir(parents=True)
    trained.touch()
    summary=dict(inference_checkpoint=str(base),training_initial_checkpoint=str(base),
                 trained_checkpoint=str(trained),selected_ids=[f'id{i}' for i in range(473)])
    (s/'cycle_001/cycle_summary.json').write_text(json.dumps(summary))
    (s/'evaluation_history.json').write_text(json.dumps([
        {'cycle':1,'split':'test','n_labeled':949,'n_acquired':473,'metrics':{'f1_macro':.63}}
    ]))
    values,audit=report.collect(persist,data,suite)
    assert len(values)==1
    assert values[0]['f1_macro']==.63
    summary['training_initial_checkpoint']=str(trained)
    (s/'cycle_001/cycle_summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='ancestry mismatch'):
        report.collect(persist,data,suite)


def test_task4_short_job_schedule_is_108_individual_fits():
    schedule = load('task4_schedule')
    jobs = schedule.schedule()
    assert len(jobs) == 108
    assert len(set((x.seed, x.cycle, x.policy) for x in jobs)) == 108
    assert {x.walltime for x in jobs if x.cycle <= 2} == {'00:15:00'}
    assert {x.walltime for x in jobs if x.cycle >= 3} == {'00:20:00'}
    assert [x.seed for x in jobs] == [135] * 36 + [246] * 36 + [357] * 36
    assert set(x.policy for x in jobs) == set(schedule.POLICIES)
    assert len(schedule.schedule(cycles=6)) == 72
    with pytest.raises(ValueError, match='1..9'):
        schedule.schedule(cycles=10)
    with pytest.raises(ValueError, match='walltime'):
        schedule.schedule(late_walltime='not-a-time')


def test_enqueue_slurm_dependencies_and_reverse_release(tmp_path):
    schedule = load('task4_schedule')
    source = tmp_path / 'source'
    script = source / 'scripts/adaptive/task4_cycle.sbatch'
    script.parent.mkdir(parents=True)
    script.write_text('#!/bin/bash\n')
    job_dir = tmp_path / 'job'
    job_dir.mkdir()
    plan = job_dir / 'queue_plan.tsv'
    plan.write_text(''.join(
        f'{x.seed}\t{x.cycle}\t{x.policy}\t{x.walltime}\n' for x in schedule.schedule()
    ))
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    calls = tmp_path / 'calls.jsonl'
    fake = """#!/bin/bash
name="${0##*/}"
printf '%s|%s\\n' "$name" "$*" >> "$MOCK_CALLS"
if [[ "$name" == sbatch ]]; then
  counter="$(cat "$MOCK_COUNTER")"
  if [[ "$FAIL_AFTER" == "$counter" ]]; then exit 1; fi
  echo "$((12000+counter))"
  echo "$((counter+1))" > "$MOCK_COUNTER"
fi
"""
    counter=tmp_path/'counter'
    counter.write_text('0')
    for name in ('sbatch','scontrol','scancel','squeue'):
        p = bindir / name
        p.write_text(fake)
        p.chmod(0o755)
    import os
    env = dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}', MOCK_CALLS=str(calls), MOCK_COUNTER=str(counter), FAIL_AFTER='-1')
    script_path = SCRIPTS / 'task4_enqueue.sh'
    result = subprocess.run(['bash', str(script_path), str(plan), str(source), str(job_dir), '9'],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    actions = [line.split('|',1) for line in calls.read_text().splitlines()]
    submitted = [args.split() for cmd,args in actions if cmd=='sbatch']
    released = [args.split() for cmd,args in actions if cmd=='scontrol']
    assert len(submitted)==108 and len(released)==108
    assert all('--hold' in a for a in submitted)
    # Row order: seed, then cycle, then four policies.
    # Every cycle 1 policy starts independently; no cross-policy dependency.
    for offset in (0, 36, 72):
        for index in range(offset, offset + 4):
            assert not any(a.startswith('--dependency=') for a in submitted[index])
        for index in range(offset + 4, offset + 36):
            assert f'--dependency=afterok:{12000 + index - 4}' in submitted[index]
            assert sum(a.startswith('--dependency=') for a in submitted[index]) == 1
    assert released[0] == ['release','12107']
    assert released[-1] == ['release','12000']
    assert len((job_dir/'submitted.tsv').read_text().splitlines()) == 108
    # An abandoned duplicate would be dangerous; reject it.
    double = subprocess.run(['bash', str(script_path), str(plan), str(source), str(job_dir), '9'],
                            env=env, text=True, capture_output=True)
    assert double.returncode != 0 and 'Submission exists' in double.stderr


def test_enqueue_failure_cancels_new_jobs_and_never_releases(tmp_path):
    schedule = load('task4_schedule')
    source = tmp_path/'source'
    script = source/'scripts/adaptive/task4_cycle.sbatch'
    script.parent.mkdir(parents=True)
    script.touch()
    job_dir = tmp_path/'job'
    job_dir.mkdir()
    plan=job_dir/'plan.tsv'
    plan.write_text(''.join(f'{x.seed}\t{x.cycle}\t{x.policy}\t{x.walltime}\n'
                            for x in schedule.schedule()))
    fake_bin=tmp_path/'bin'
    fake_bin.mkdir()
    mocklog=tmp_path/'mock.jsonl'
    mock="""#!/bin/bash
name="${0##*/}"
printf '%s|%s\\n' "$name" "$*" >> "$MOCK_CALLS"
if [[ "$name" == sbatch ]]; then
  n="$(cat "$MOCK_COUNTER")"
  echo "$((n+1))" > "$MOCK_COUNTER"
  if [[ "$n" -eq 4 ]]; then exit 1; fi
  echo "$((2000+n))"
fi
"""
    counter=tmp_path/'counter'
    counter.write_text('0')
    for x in ('sbatch','scontrol','scancel','squeue'):
        target=fake_bin/x
        target.write_text(mock)
        target.chmod(0o755)
    import os
    env=dict(os.environ,PATH=f'{fake_bin}:{os.environ["PATH"]}',MOCK_CALLS=str(mocklog),MOCK_COUNTER=str(counter))
    result=subprocess.run(['bash',str(SCRIPTS/'task4_enqueue.sh'),str(plan),str(source),str(job_dir),'9'],
                          env=env,text=True,capture_output=True)
    assert result.returncode != 0
    actions=[s.split('|',1) for s in mocklog.read_text().splitlines()]
    assert len([x for x in actions if x[0]=='sbatch'])==5
    assert [x[0] for x in actions].count('scontrol')==0
    cancel=[x[1].split() for x in actions if x[0]=='scancel']
    assert cancel == [['2000','2001','2002','2003']]
