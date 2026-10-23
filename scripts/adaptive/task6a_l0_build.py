#!/usr/bin/env python3
"""Generate a strict new Task6A runner: L0 and all AL cycles use exponent .5.

The Task4 builder is reused to preserve the original historical split and
warm-start semantics. Each upstream replacement is anchored and fail-closed.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path

SUITE = 'astrominn_closed_loop_warm_weight05_newl0_v1'


def replace_one(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f'Expected exactly one upstream anchor, got {count}: {old[:110]!r}')
    return text.replace(old, new, 1)


def build_runner(repo: Path) -> str:
    upstream_builder = repo/'scripts/adaptive/task4_build_runner.py'
    original_runner = repo/'scripts/run_astrominn_historical_replay.py'
    if not upstream_builder.is_file() or not original_runner.is_file():
        raise FileNotFoundError('Requires Task4 generator and original historical replay runner')
    spec = importlib.util.spec_from_file_location('task6a_l0_upstream', upstream_builder)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load Task4 runner generator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.build_runner(original_runner.read_text(encoding='utf-8'))
    # Task4 required a separate bootstrap source. This new experiment explicitly
    # trains and uses its OWN three bootstrap checkpoints, so these must match.
    result = replace_one(result,
        '    if args.bootstrap_source_suite_id == args.suite_id:\n'
        '        raise ValueError("Use a distinct suite ID for Task 4.")',
        '    if args.bootstrap_source_suite_id != args.suite_id:\n'
        '        raise ValueError("Task6A requires its own freshly trained L0 bootstrap")')
    result = replace_one(result,
        '    if not args.require_existing_bootstrap:\n'
        '        raise ValueError("Task 4 must reuse an existing, validated L0 bootstrap.")',
        '    if args.run and not args.bootstrap_only and not args.require_existing_bootstrap:\n'
        '        raise ValueError("Task6A cycle workers must require an existing completed .5 L0 bootstrap")')
    result = replace_one(result,
        '        / args.bootstrap_source_suite_id\n        / "shared"\n        / f"seed_{args.seed}"\n        / "bootstrap_training"',
        '        / args.suite_id\n        / "shared"\n        / f"seed_{args.seed}"\n        / "bootstrap_training"')
    result = replace_one(result,
        '        run_id=f"{args.bootstrap_source_suite_id}_bootstrap_seed{args.seed}",',
        '        run_id=f"{args.suite_id}_bootstrap_seed{args.seed}",')
    result = replace_one(result,
        '    policy = AstroMiNNViTTrainingConfig(\n',
        '    policy = AstroMiNNViTTrainingConfig(\n'
        '        class_weight_exponent=0.5,  # same scientific setting for new L0 and all AL cycles\n')
    result = replace_one(result,
        '        "cycle_training_initialization": args.training_initialization,',
        '        "cycle_training_initialization": args.training_initialization,\n'
        '        "task6a_bootstrap_weight_exponent": 0.5,\n'
        '        "task6a_cycle_weight_exponent": 0.5,\n'
        '        "task6a_checkpoint_selection": "best_weighted_validation_loss",\n'
        '        "task6a_object_weighting": "per_sample",')
    result = replace_one(result,
        '    if args.bootstrap_only and not args.run:\n',
        f'    if args.suite_id != "{SUITE}":\n'
        '        raise ValueError("Task6A requires its dedicated NEW L0 suite ID")\n'
        '    if args.seed not in (135, 246, 357):\n'
        '        raise ValueError("Task6A requires one of three fixed model seeds")\n'
        '    if args.strategies not in (["entropy"], ["entropy_balanced_explore"]):\n'
        '        raise ValueError("Task6A requires a single planned AL policy")\n'
        '    if args.bootstrap_only and args.require_existing_bootstrap:\n'
        '        raise ValueError("Bootstrap fit cannot require an existing bootstrap")\n'
        '    if args.bootstrap_only and not args.run:\n')
    result = replace_one(result,
        '    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))',
        '    if (policy.class_weight_exponent != 0.5 or\n'
        '            policy.class_weight_source != "training_manifest" or\n'
        '            policy.new_object_weighting != "per_sample"):\n'
        '        raise AssertionError("Task6A training policy unexpectedly changed")\n'
        '    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))')
    if result.count('class_weight_exponent=0.5') != 1:
        raise AssertionError('Must configure .5 exactly once in shared L0/cycle policy')
    if 'L0_then_previous_loss_best_checkpoint' not in result:
        raise AssertionError('Task6A must preserve warm-start semantics')
    return result


def generate(repo: Path) -> Path:
    result = build_runner(repo)
    output = repo/'scripts/run_task6a_l0_closed_loop.py'
    if output.exists():
        if output.read_text(encoding='utf-8') != result:
            raise ValueError('Existing generated Task6A L0 runner differs; refusing overwrite')
    else:
        output.write_text(result, encoding='utf-8')
    return output


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    a = ap.parse_args()
    print('Generated Task6A new-bootstrap runner:', generate(a.repo.resolve()))
