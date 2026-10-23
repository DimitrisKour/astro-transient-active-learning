"""Contract tests: L0 .5, six independent warm-start chains, fail-closed DAG."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts/adaptive'
sys.path.insert(0,str(SCRIPTS))
import task6a_l0 as job
import task6a_l0_prepare as pre
import task6a_l0_build as builder


def test_57_jobs_and_shared_seed_dependencies():
    rows=job.plan()
    assert len(rows)==57
    lookup={r['key']:r for r in rows}
    assert len(lookup)==57
    for seed in (135,246,357):
        r=lookup[f'l0:{seed}']
        assert r['cycle']==0 and r['policy']=='bootstrap' and r['parent'] is None
        for policy in ('entropy','entropy_balanced_explore'):
            assert lookup[f'{seed}:{policy}:1']['parent']==f'l0:{seed}'
            for cycle in range(2,10):
                assert lookup[f'{seed}:{policy}:{cycle}']['parent']==f'{seed}:{policy}:{cycle-1}'
    assert sum(r['cycle']==0 for r in rows)==3


def test_submit_commands_per_chain_and_bootstrap():
    root=Path('/tmp/j6')
    source=root/'source'
    for r in job.plan():
        pid=None if r['parent'] is None else '12345'
        cmd=job.sbatch_command(root,source,r,parent_job=pid,walltime='00:20:00',l0_walltime='00:22:00')
        assert '--hold' in cmd and '--parsable' in cmd
        assert '--time=00:22:00' in cmd if r['cycle']==0 else '--time=00:20:00' in cmd
        if pid:
            assert cmd.count('--dependency=afterok:12345')==1
        else:
            assert not any(arg.startswith('--dependency=') for arg in cmd)
        assert cmd[-3:]==[str(r['seed']),str(r['cycle']),r['policy']]


def test_time_restricted_and_not_silently_25_or_more():
    for s in ('00:20:00','00:24:59','00:10:00'):job.validate_walltime(s)
    for s in ('00:25:00','00:30:00','garbage','00:00:00'):
        with pytest.raises(ValueError):job.validate_walltime(s)


def test_snapshot_uses_distinct_suite_and_own_l0():
    assert pre.SUITE=='astrominn_closed_loop_warm_weight05_newl0_v1'
    assert job.POLICIES==('entropy','entropy_balanced_explore')
    assert pre.SUITE != 'astrominn_closed_loop_warm_weight05_v1'
    common=(SCRIPTS/'task6a_l0_common.sh').read_text()
    worker=(SCRIPTS/'task6a_l0_worker.sbatch').read_text()
    assert '--bootstrap-source-suite-id "$TASK6A_L0_SUITE_ID"' in common
    assert '--require-existing-bootstrap' not in common
    assert '--bootstrap-only --run' in worker
    assert '--require-existing-bootstrap --cycle "$cycle" --run' in worker
    assert 'SLURM_ARRAY_TASK_ID' not in worker


def test_builder_rejects_bad_anchor():
    with pytest.raises(ValueError,match='upstream anchor'):
        builder.replace_one('abc','MISSING','zzz')
    assert builder.replace_one('abc','b','B')=='aBc'


def test_builder_full_script_science_contract(tmp_path):
    repo=tmp_path/'repo'
    (repo/'scripts/adaptive').mkdir(parents=True)
    (repo/'scripts/run_astrominn_historical_replay.py').write_text('ORIGINAL')
    (repo/'scripts/adaptive/task4_build_runner.py').write_text('''
def build_runner(source):
    return """    if args.bootstrap_source_suite_id == args.suite_id:
        raise ValueError("Use a distinct suite ID for Task 4.")
    if not args.require_existing_bootstrap:
        raise ValueError("Task 4 must reuse an existing, validated L0 bootstrap.")
        / args.bootstrap_source_suite_id
        / "shared"
        / f"seed_{args.seed}"
        / "bootstrap_training"
        run_id=f"{args.bootstrap_source_suite_id}_bootstrap_seed{args.seed}",
    policy = AstroMiNNViTTrainingConfig(
        "cycle_training_initialization": args.training_initialization,
    if args.bootstrap_only and not args.run:
    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))
    vit_tower: L0_then_previous_loss_best_checkpoint
"""
''')
    text=builder.build_runner(repo)
    assert 'class_weight_exponent=0.5' in text
    assert 'task6a_bootstrap_weight_exponent": 0.5' in text
    assert 'bootstrap_source_suite_id != args.suite_id' in text
    assert 'if args.run and not args.bootstrap_only and not args.require_existing_bootstrap' in text
    assert 'run_id=f"{args.suite_id}_bootstrap_seed{args.seed}"' in text
    assert text.count('class_weight_exponent=0.5')==1


def test_report_checks_own_bootstrap():
    s=(SCRIPTS/'task6a_l0_report.py').read_text()
    assert "BASE = NEW" in s
    assert "boot['policy']['class_weight_exponent']!=0.5" in s
    assert 'paired_validation.csv' in s
    assert 'per_class' in s


def test_plan_dry_run_does_not_call_sbatch(monkeypatch,tmp_path):
    monkeypatch.setattr(job,'_check_pre_submission',lambda _:None)
    monkeypatch.setattr(job,'_run',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('sbatch called')))
    job.submit(tmp_path,through=2,actually_submit=False)
