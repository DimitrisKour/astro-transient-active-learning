"""Task6A must fail closed on changed science contracts or source drift."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'research' / 'historical_replay' / 'task6a'
sys.path.insert(0,str(SCRIPTS))
import task6a_build_runner as build
import task6a_prepare as prepare
import task6a


def make_repo(tmp_path: Path) -> Path:
    repo=tmp_path/'repo'
    (repo/'scripts'/'research'/'historical_replay'/'task6a').mkdir(parents=True)
    (repo/'scripts'/'run_astrominn_historical_replay.py').write_text('''
    if args.bootstrap_only and not args.run:
        raise RuntimeError()
    policy = AstroMiNNViTTrainingConfig(
        batch_size=512,
    )
    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))
    plan = {
        "cycle_training_initialization": args.training_initialization,
    }
''')
    (repo/'scripts'/'research'/'historical_replay'/'task6a'/'task4_build_runner.py').write_text('''
def build_runner(source):
    return source
''')
    return repo


def test_build_changes_one_scientific_input(tmp_path):
    repo=make_repo(tmp_path)
    result=build.build_runner(repo)
    assert result.count('class_weight_exponent=0.5')==1
    assert result.count('task6a_class_weight_exponent')==1
    assert 'task6a_checkpoint_selection": "best_weighted_validation_loss' in result
    assert 'task6a_object_weighting": "per_sample' in result
    assert 'training_initialization != "warm_start"' in result
    assert 'Task 6A accepts one planned acquisition policy' in result
    assert 'class_weight_source != "training_manifest"' in result
    generated=build.generate(repo)
    assert generated.exists() and build.generate(repo)==generated


def test_build_fails_if_source_anchor_missing(tmp_path):
    repo=make_repo(tmp_path)
    path=repo/'scripts'/'run_astrominn_historical_replay.py'
    path.write_text(path.read_text().replace('policy = AstroMiNNViTTrainingConfig(', 'policy = OtherTrainingConfig('))
    with pytest.raises(ValueError, match='upstream anchor'):
        build.build_runner(repo)


def test_build_never_replaces_nonmatching_file(tmp_path):
    repo=make_repo(tmp_path)
    dst=repo/'scripts'/'run_task6a_closed_loop.py'
    dst.write_text('DO NOT REPLACE')
    with pytest.raises(ValueError,match='Refusing'):
        build.generate(repo)
    assert dst.read_text()=='DO NOT REPLACE'


def test_time_limit_is_below_25_minutes():
    for t in ('00:24:00','00:10:00','0:00:01'):
        task6a._validate_time(t)
    for t in ('00:25:00','00:00:00','bad','01:00:00'):
        with pytest.raises(ValueError):
            task6a._validate_time(t)


def test_contract_freeze_and_detects_drift(tmp_path):
    repo=make_repo(tmp_path)
    (repo/'src'/'al_pipeline').mkdir(parents=True)
    (repo/'src'/'al_pipeline'/'__init__.py').write_text('# test\n')
    for f in SCRIPTS.glob('task6a_*'):
        (repo/'scripts'/'research'/'historical_replay'/'task6a'/f.name).write_bytes(f.read_bytes())
    persistent=tmp_path/'persistent'
    job=persistent/'closed_loop'/prepare.SUITE
    registry=tmp_path/'registry.toml'
    registry.write_text('[storage]\npersistent_root="x"\ndata_root="y"\n')
    prepare.freeze(repo,job,registry)
    prepare.verify(job)
    prepare.freeze(repo,job,registry)
    frozen_source=job/'source'/'src'/'al_pipeline'/'__init__.py'
    frozen_source.write_text('MODIFIED')
    with pytest.raises(ValueError,match='source mismatch'):
        prepare.verify(job)


def test_no_unrequested_jobs_and_array_concurrency():
    worker=(SCRIPTS/'task6a_cycle.sbatch').read_text()
    common=(SCRIPTS/'task6a_common.sh').read_text()
    assert 'SLURM_ARRAY_TASK_ID' in worker
    assert 'index/2' in worker and 'index%2' in worker
    assert 'class_weight_exponent' not in common
    assert '--training-initialization warm_start' in common
    assert '--max-cycles 9 --complete-pool' in common
    code=(SCRIPTS/'task6a.py').read_text()
    assert "'--array=0-5%3'" in code
    assert "'--dependency=afterok:'" in code
    assert 'if not submit:' in code
    assert 'scontrol' in code and 'scancel' in code


def test_reporter_checks_weighting_and_checkpoint_ancestry():
    code=(SCRIPTS/'task6a_report.py').read_text()
    assert "train['policy']['class_weight_exponent']!=0.5" in code
    assert "'inference_checkpoint','training_initial_checkpoint'" in code
    assert 'paired_validation.csv' in code
    assert 'Test is only descriptive' in code
