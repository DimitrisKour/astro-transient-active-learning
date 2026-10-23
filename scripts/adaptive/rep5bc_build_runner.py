#!/usr/bin/env python3
"""Generate a Task 5C-specific closed-loop runner from the immutable Experiment B runner.

Uses the known Task4 builder as a text-level transform, WITHOUT writing the
Task4 runner. Every additional edit checks a unique upstream anchor.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"5C incompatible upstream runner: expected 1 anchor, found {count}: {old[:90]!r}")
    return source.replace(old, new, 1)


def generate(repo: Path) -> str:
    script = repo / "scripts/adaptive/task4_build_runner.py"
    spec = importlib.util.spec_from_file_location("rep5bc_task4_builder", script)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = (repo / "scripts/run_astrominn_historical_replay.py").read_text(encoding="utf-8")
    runner = module.build_runner(original)
    runner = replace_once(
        runner,
        'from al_pipeline.core.experiments.budget import plan_cycle_budget',
        'from al_pipeline.core.experiments.budget import plan_cycle_budget\n'
        'from al_pipeline.astronomy.representation.vi_tuning import make_vit_factory, MODES',
    )
    runner = replace_once(
        runner,
        '                        default="warm_start")',
        '                        default="warm_start")\n'
        '    parser.add_argument("--representation-mode", choices=MODES, required=True)',
    )
    runner = replace_once(
        runner,
        '    if args.training_initialization != "warm_start":\n'
        '        raise ValueError("Task 4 must use warm_start; do not silently change the experiment.")',
        '    if args.training_initialization not in ("warm_start", "restart_from_baseline"):\n'
        '        raise ValueError("Task 5C requires warm_start or restart_from_baseline.")',
    )
    runner = replace_once(runner, 'Task 4 requires --bootstrap-source-suite-id.', 'Task 5C requires --bootstrap-source-suite-id.')
    runner = replace_once(runner, 'Use a distinct suite ID for Task 4.', 'Use a distinct suite ID for Task 5C.')
    runner = replace_once(runner, 'Task 4 must reuse an existing, validated L0 bootstrap.', 'Task 5C must reuse an existing, validated L0 bootstrap.')
    runner = replace_once(
        runner, '        "bootstrap_source_suite_id": args.bootstrap_source_suite_id,',
        '        "bootstrap_source_suite_id": args.bootstrap_source_suite_id,\n'
        '        "representation_mode": args.representation_mode,',
    )
    runner = replace_once(runner, '"vit_tower": "L0_then_previous_loss_best_checkpoint",',
                          '"vit_tower": "5c_" + args.representation_mode,')
    runner = replace_once(
        runner, '    if args.extend_budget:\n',
        '    # New training policy is assigned BEFORE starting/resuming any cycle.\n'
        '    # The standard trainer still stores only ViT checkpoints; experts stay frozen.\n'
        '    for experiment in suite.experiments.values():\n'
        '        experiment.trainer.model_factory = make_vit_factory(\n'
        '            args.representation_mode, artifact_dir=model_artifact_dir,\n'
        '            legacy_code_dir=DEFAULT_LEGACY_CODE_DIR,\n'
        '        )\n'
        '    if args.extend_budget:\n',
    )
    runner = replace_once(runner,
        '"""Task 4: closed-loop historical replay with warm-started AstroMiNN ViT."""',
        '"""Task 5C: closed-loop comparison of full ViT vs head-only adaptation."""')
    compile(runner, "scripts/run_representation_5c.py", "exec")
    return runner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    a = p.parse_args()
    destination = a.repo / "scripts/run_representation_5c.py"
    payload = generate(a.repo)
    if destination.exists() and destination.read_text(encoding="utf-8") != payload:
        raise ValueError(f"Refusing to overwrite a modified 5C runner: {destination}")
    destination.write_text(payload, encoding="utf-8")
    print(f"Validated dedicated 5C runner: {destination}")


if __name__ == "__main__":
    main()
