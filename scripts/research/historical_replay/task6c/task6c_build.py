#!/usr/bin/env python3
"""Build a standalone equal-new-object-weighting closed-loop runner.

The upstream code is not modified: this generates an isolated frozen snapshot.
Both modes reuse their matching, originally per-sample-weighted L0 checkpoint.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

MODES={
    'loss': {'suite':'astrominn_closed_loop_equal_object_loss_085_v1',
             'reference':'astrominn_closed_loop_warm_v1',
             'bootstrap':'astrominn_historical_replay_v1',
             'checkpoint':'loss_best'},
    'macro_f1':{'suite':'astrominn_closed_loop_equal_object_f1best_085_v1',
                'reference':'astrominn_closed_loop_warm_f1best_085_v1',
                'bootstrap':'astrominn_closed_loop_warm_f1best_085_v1',
                'checkpoint':'object_macro_f1_best'},
}

def once(s:str,old:str,new:str)->str:
    n=s.count(old)
    if n!=1: raise ValueError(f'Expected one audited anchor, found {n}: {old[:110]!r}')
    return s.replace(old,new,1)

def _load(path:Path,name:str):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import {path}')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def build_snapshot(repo:Path,mode:str)->tuple[str,str]:
    if mode not in MODES: raise ValueError(f'Unsupported checkpoint choice: {mode}')
    cfg=MODES[mode]
    original_training=(repo/'src/al_pipeline/astronomy/astrominn/training.py').read_text()
    if mode=='loss':
        builder=_load(repo/'scripts/research/historical_replay/task4/task4_build_runner.py','task6c_task4_builder')
        runner=builder.build_runner((repo/'scripts/run_astrominn_historical_replay.py').read_text())
        training=original_training
        runner=once(runner,
            '    if args.bootstrap_source_suite_id == args.suite_id:\n'
            '        raise ValueError("Use a distinct suite ID for Task 4.")',
            f'    if args.bootstrap_source_suite_id != "{cfg["bootstrap"]}":\n'
            '        raise ValueError("Task6C loss mode requires the original .85 L0")')
        runner=once(runner,
            '    if not args.require_existing_bootstrap:\n'
            '        raise ValueError("Task 4 must reuse an existing, validated L0 bootstrap.")',
            '    if not args.require_existing_bootstrap:\n'
            '        raise ValueError("Task6C must reuse a verified matching baseline")')
    else:
        b=_load(repo/'scripts/research/historical_replay/task6b/task6b_build.py','task6c_task6b_builder')
        runner=b.build_runner(repo)
        training=b.patch_training(original_training)
        runner=once(runner,
            '    if args.bootstrap_source_suite_id != args.suite_id:\n'
            '        raise ValueError("Task6B requires its OWN L0 training")',
            f'    if args.bootstrap_source_suite_id != "{cfg["bootstrap"]}":\n'
            '        raise ValueError("Task6C macro-F1 mode requires Task6B L0")')
        runner=once(runner,
            '    if args.run and not args.bootstrap_only and not args.require_existing_bootstrap:\n'
            '        raise ValueError("Task6B cycle jobs require their completed own L0 bootstrap")',
            '    if not args.require_existing_bootstrap:\n'
            '        raise ValueError("Task6C must reuse a verified matching baseline")')
        runner=once(runner,
            '        / args.suite_id\n        / "shared"\n        / f"seed_{args.seed}"\n        / "bootstrap_training"',
            '        / args.bootstrap_source_suite_id\n        / "shared"\n        / f"seed_{args.seed}"\n        / "bootstrap_training"')
        runner=once(runner,
            '        run_id=f"{args.suite_id}_bootstrap_seed{args.seed}",',
            '        run_id=f"{args.bootstrap_source_suite_id}_bootstrap_seed{args.seed}",')
        runner=once(runner,
            '    if args.suite_id != "astrominn_closed_loop_warm_f1best_085_v1":',
            f'    if args.suite_id != "{cfg["suite"]}":')
        runner=once(runner,
            '        "task6b_object_weighting": "per_sample",',
            '        "task6c_object_weighting": "equal_object",')
    # Both modes preserve bootstrap validation with the original per-sample policy,
    # then use equal_object ONLY for training on the cumulative acquired manifest.
    runner=once(runner,
        '    suite = build_astrominn_simulated_suite(\n',
        '    cycle_policy = replace(policy, new_object_weighting="equal_object")\n'
        '    suite = build_astrominn_simulated_suite(\n')
    runner=once(runner,
        '        training_policy=policy,',
        '        training_policy=cycle_policy,')
    runner=once(runner,
        '    if args.bootstrap_only and not args.run:\n',
        f'    if args.suite_id != "{cfg["suite"]}":\n'
        '        raise ValueError("Task6C suite ID mismatch")\n'
        '    if args.seed not in (135, 246, 357):\n'
        '        raise ValueError("Task6C seed out of scope")\n'
        '    if args.strategies not in (["entropy"], ["entropy_balanced_explore"]):\n'
        '        raise ValueError("Task6C acquisition policy out of scope")\n'
        '    if args.bootstrap_only:\n'
        '        raise ValueError("Task6C never trains a new bootstrap")\n'
        '    if args.bootstrap_only and not args.run:\n')
    runner=once(runner,
        '    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))',
        f'    if (policy.class_weight_exponent != 0.85 or\n'
        f'            policy.class_weight_source != "training_manifest" or\n'
        f'            policy.new_object_weighting != "per_sample"):\n'
        f'        raise AssertionError("Unexpected Task6C baseline policy")\n'
        f'    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))') if mode=='loss' else runner
    runner=once(runner,
        '        "cycle_training_initialization": args.training_initialization,',
        '        "cycle_training_initialization": args.training_initialization,\n'
        '        "task6c_equal_new_object_weighting": True,\n'
        f'        "task6c_checkpoint_mode": "{mode}",\n'
        f'        "task6c_baseline_suite": "{cfg["bootstrap"]}",')
    # The Task6B parent runner already includes a strict checkpoint policy and
    # always initializes the model from the predecessor's selected checkpoint.
    if mode=='loss':
        assert 'L0_then_previous_loss_best_checkpoint' in runner
    else:
        assert 'L0_then_previous_object_macro_f1_best_checkpoint' in runner
    assert 'cycle_policy = replace(policy, new_object_weighting="equal_object")' in runner
    return runner,training
