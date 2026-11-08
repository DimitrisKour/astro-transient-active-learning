from __future__ import annotations
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts/research/historical_replay/task6c'
sys.path.insert(0,str(SCRIPTS))
import task6c
import task6c_build as b
import task6c_prepare as prep


def test_distinct_modes_and_references():
    assert set(b.MODES)=={'loss','macro_f1'}
    assert b.MODES['loss']['bootstrap']=='astrominn_historical_replay_v1'
    assert b.MODES['macro_f1']['bootstrap']=='astrominn_closed_loop_warm_f1best_085_v1'
    assert b.MODES['loss']['suite'] != b.MODES['macro_f1']['suite']
    assert b.MODES['macro_f1']['reference']==b.MODES['macro_f1']['bootstrap']


def test_exact_six_independent_dependency_chains():
    rows=task6c.plan()
    assert len(rows)==54
    assert set(x['seed'] for x in rows)=={135,246,357}
    assert set(x['policy'] for x in rows)=={'entropy','entropy_balanced_explore'}
    seen=set()
    for r in rows:
        assert r['key'] not in seen
        if r['cycle']==1: assert r['parent'] is None
        else: assert r['parent']==f'{r["seed"]}:{r["policy"]}:{r["cycle"]-1}'
        seen.add(r['key'])
    assert len(task6c.plan(through=5))==30
    for n in [0,10]:
        with pytest.raises(ValueError):task6c.plan(n)


def test_under_25_minute_walltime():
    for t in ['00:15:00','00:20:00','00:24:00']:task6c.validate_walltime(t)
    for t in ['00:00:00','00:25:00','foo','01:00:00']:
        with pytest.raises(ValueError):task6c.validate_walltime(t)


def test_sbatch_dependency_is_own_predecessor():
    rows=task6c.plan(through=2)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        for mode in b.MODES:
            for r in rows:
                parent=None if r['cycle']==1 else '12345'
                cmd=task6c.sbatch_command(r,mode,path,path,parent,'00:20:00')
                assert '--hold' in cmd
                if r['cycle']==1:assert not any(x.startswith('--dependency=') for x in cmd)
                else:assert '--dependency=afterok:12345' in cmd
                assert cmd[-5:] == [str(path), mode, str(r["seed"]), str(r["cycle"]), r["policy"]]
            with pytest.raises(ValueError): task6c.sbatch_command(rows[0],mode,path,path,'1','00:20:00')


def test_common_shell_arguments():
    import subprocess
    for mode in b.MODES:
        code='source "$1"; task6c_arguments "$2" 135 entropy; printf "%s\\n" "${TASK6C_ARGS[@]}"'
        out=subprocess.run(['bash','-c',code,'_',str(SCRIPTS/'task6c_common.sh'),mode],
                           text=True,capture_output=True,check=True).stdout.splitlines()
        assert b.MODES[mode]['suite'] in out
        assert b.MODES[mode]['bootstrap'] in out
        assert '--require-existing-bootstrap' in out
        assert '--max-cycles' in out
        assert '--bootstrap-only' not in out


def test_checked_replacements_fail_closed():
    with pytest.raises(ValueError):b.once('abc','missing','changed')
    with pytest.raises(ValueError):b.once('a-a','a','b')


def test_science_runner_uses_per_sample_bootstrap_and_equal_object_cycles(tmp_path):
    # Mock builders emit their EXACT documented runner anchors; the upstream
    # builders are independently tested in their corresponding Task4/6B suite.
    t=tmp_path/'repo';(t/'scripts/research/historical_replay/task6c').mkdir(parents=True)
    (t/'src/al_pipeline/astronomy/astrominn').mkdir(parents=True)
    (t/'src/al_pipeline/astronomy/astrominn/training.py').write_text('training original')
    (t/'scripts/run_astrominn_historical_replay.py').write_text('historical source')
    core='''    if args.bootstrap_only and not args.run:\n    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))\n    suite = build_astrominn_simulated_suite(\n        training_policy=policy,\n        "cycle_training_initialization": args.training_initialization,\n'''
    loss='''    if args.bootstrap_source_suite_id == args.suite_id:\n        raise ValueError("Use a distinct suite ID for Task 4.")\n    if not args.require_existing_bootstrap:\n        raise ValueError("Task 4 must reuse an existing, validated L0 bootstrap.")\nL0_then_previous_loss_best_checkpoint\n'''+core
    macro='''    if args.bootstrap_source_suite_id != args.suite_id:\n        raise ValueError("Task6B requires its OWN L0 training")\n    if args.run and not args.bootstrap_only and not args.require_existing_bootstrap:\n        raise ValueError("Task6B cycle jobs require their completed own L0 bootstrap")\n        / args.suite_id\n        / "shared"\n        / f"seed_{args.seed}"\n        / "bootstrap_training"\n        run_id=f"{args.suite_id}_bootstrap_seed{args.seed}",\n    if args.suite_id != "astrominn_closed_loop_warm_f1best_085_v1":\n        "task6b_object_weighting": "per_sample",\nL0_then_previous_object_macro_f1_best_checkpoint\n'''+core
    class Loss: 
        @staticmethod
        def build_runner(text):return loss
    class Macro:
        @staticmethod
        def build_runner(repo):return macro
        @staticmethod
        def patch_training(s):return s+' macro_f1_support'
    def load(path,name):return Loss if 'task4' in str(path) else Macro
    with patch.object(b,'_load',side_effect=load):
        for mode in b.MODES:
            runner,trainer=b.build_snapshot(t,mode)
            assert 'cycle_policy = replace(policy, new_object_weighting="equal_object")' in runner
            assert 'training_policy=cycle_policy' in runner
            assert 'training_policy=policy' not in runner
            assert 'task6c_equal_new_object_weighting' in runner
            assert b.MODES[mode]['bootstrap'] in runner
            assert b.MODES[mode]['suite'] in runner
            assert 'task6c_checkpoint_mode' in runner
            if mode=='loss': assert trainer=='training original'
            else: assert 'macro_f1_support' in trainer


def test_summary_contract_rejects_bootstrap_mode_mismatch(tmp_path,monkeypatch):
    data=tmp_path/'data';monkeypatch.setattr(prep,'DATA',data)
    for mode in b.MODES:
        suite=b.MODES[mode]['bootstrap']
        for seed in prep.SEEDS:
            folder=data/'runs'/suite/'shared'/f'seed_{seed}'/'bootstrap_training'
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'best_vit_tower.pth').write_bytes(b'pretend')
            summary={'seed':seed,'cycle':0,'run_id':f'{suite}_bootstrap_seed{seed}',
                     'initial_checkpoint':None,'initialization':'random_vit',
                     'checkpoint_path':str(folder/'best_vit_tower.pth'),
                     'policy':{'class_weight_exponent':.85,'class_weight_source':'training_manifest',
                               'new_object_weighting':'per_sample'}}
            if mode=='macro_f1':
                summary['policy']['checkpoint_selection']='object_macro_f1_best'
                summary['checkpoint_selection']='object_macro_f1_best'
                (folder/'best_loss_vit_tower.pth').write_bytes(b'loss')
            (folder/'training_summary.json').write_text(json.dumps(summary))
        assert prep.bootstrap_check(mode,strict=True)
        f=data/'runs'/suite/'shared'/'seed_135'/'bootstrap_training'/'training_summary.json'
        wrong=json.loads(f.read_text());wrong['policy']['new_object_weighting']='equal_object'
        f.write_text(json.dumps(wrong))
        with pytest.raises(ValueError):prep.bootstrap_check(mode,strict=True)


def test_verifier_refuses_changed_snapshot(tmp_path):
    mode='loss';dest=tmp_path/'job';(dest/'source').mkdir(parents=True)
    data=b'original';(dest/'source'/'a.py').write_bytes(data)
    contract={'selection':mode,'suite_id':b.MODES[mode]['suite'],
              'checkpoint_selection':'loss_best','cycle_object_weighting':'equal_object',
              'bootstrap_suite':b.MODES[mode]['bootstrap'],'class_weight_exponent':.85,
              'registry_sha256':prep.digest_bytes(b'registry')}
    (dest/'registry.toml').write_bytes(b'registry')
    (dest/'contract.json').write_text(json.dumps(contract))
    (dest/'source_snapshot.json').write_text(json.dumps({'a.py':prep.digest_bytes(data)}))
    prep.verify(mode,dest)
    (dest/'source'/'a.py').write_bytes(b'changed')
    with pytest.raises(ValueError):prep.verify(mode,dest)


def test_submission_fail_closed_before_any_sbatch(tmp_path,monkeypatch):
    mode='loss';d=tmp_path/'closed_loop'/b.MODES[mode]['suite']
    d.mkdir(parents=True)
    (d/'submitted.json').write_text('{}')
    monkeypatch.setattr(task6c,'job_dir',lambda m:d)
    with pytest.raises(RuntimeError,match='submission marker'):
        task6c.submit(mode,walltime='00:20:00',really=True)

