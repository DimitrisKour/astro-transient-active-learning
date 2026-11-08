#!/usr/bin/env python3
"""Generate an isolated Task 6A runner from the existing, anchored Task 4 generator.

Change ONLY the class weight exponent from the Task 4 runner, keeping
within-cycle loss-based checkpoint selection and all other training policy fixed.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path

EXPONENT = 0.5


def replace_exact(text: str, old: str, new: str) -> str:
    observed = text.count(old)
    if observed != 1:
        raise ValueError(f"Expected exactly one upstream anchor, got {observed}: {old[:90]!r}")
    return text.replace(old, new, 1)


def build_runner(repo: Path) -> str:
    upstream_builder = repo / "scripts" / "research" / "historical_replay" / "task4" / "task4_build_runner.py"
    source = repo / "scripts" / "run_astrominn_historical_replay.py"
    if not upstream_builder.is_file() or not source.is_file():
        raise FileNotFoundError("Task 6A requires original Task 4 builder and historical replay runner")
    spec = importlib.util.spec_from_file_location("task6a_upstream_task4_builder", upstream_builder)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import Task4 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated: str = module.build_runner(source.read_text(encoding="utf-8"))
    generated = replace_exact(
        generated,
        '    policy = AstroMiNNViTTrainingConfig(\n',
        '    policy = AstroMiNNViTTrainingConfig(\n'
        '        class_weight_exponent=0.5,  # TASK6A SINGLE SCIENTIFIC CHANGE\n',
    )
    generated = replace_exact(
        generated,
        '        "cycle_training_initialization": args.training_initialization,',
        '        "cycle_training_initialization": args.training_initialization,\n'
        '        "task6a_class_weight_exponent": 0.5,\n'
        '        "task6a_checkpoint_selection": "best_weighted_validation_loss",\n'
        '        "task6a_object_weighting": "per_sample",',
    )
    generated = replace_exact(
        generated,
        '    if args.bootstrap_only and not args.run:\n',
        '    if args.suite_id != "astrominn_closed_loop_warm_weight05_v1":\n'
        '        raise ValueError("Task 6A requires its dedicated suite ID")\n'
        '    if args.training_initialization != "warm_start":\n'
        '        raise ValueError("Task 6A must use warm_start")\n'
        '    if args.bootstrap_source_suite_id != "astrominn_historical_replay_v1":\n'
        '        raise ValueError("Task 6A must reuse original L0 bootstrap")\n'
        '    if args.strategies not in (["entropy"], ["entropy_balanced_explore"]):\n'
        '        raise ValueError("Task 6A accepts one planned acquisition policy per job")\n'
        '    if args.bootstrap_only and not args.run:\n',
    )
    generated = replace_exact(
        generated,
        '    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))',
        '    if (policy.class_weight_exponent != 0.5 or '\
        'policy.new_object_weighting != "per_sample" or '\
        'policy.class_weight_source != "training_manifest"):\n'
        '        raise AssertionError("Task6A training contract changed")\n'
        '    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))',
    )
    return generated


def generate(repo: Path) -> Path:
    generated = build_runner(repo)
    target = repo / "scripts" / "run_task6a_closed_loop.py"
    if target.exists():
        if target.read_text(encoding="utf-8") != generated:
            raise ValueError(f"Refusing to overwrite nonmatching generated runner: {target}")
    else:
        target.write_text(generated, encoding="utf-8")
    return target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    args = p.parse_args()
    print(f"Generated, verified Task6A runner: {generate(args.repo)}")

if __name__ == "__main__":
    main()
