#!/usr/bin/env python3
"""Build a dedicated Task-5 runner from the checked-in Experiment-B runner.

Deliberately *does not* modify the existing historical-replay runner. Each
replacement checks an exact upstream fragment and refuses ambiguous changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one upstream anchor, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def build_runner(source: str) -> str:
    source = replace_once(
        source,
        '    parser.add_argument("--require-existing-bootstrap", action="store_true",\n'
        '                        help="Reuse a validated bootstrap; fail rather than train a missing one.")',
        '    parser.add_argument("--require-existing-bootstrap", action="store_true",\n'
        '                        help="Reuse a validated bootstrap; fail rather than train a missing one.")\n'
        '    parser.add_argument("--bootstrap-source-suite-id",\n'
        '                        help="Reuse the completed L0 fit from a separate suite without copying it.")\n'
        '    parser.add_argument("--training-initialization",\n'
        '                        choices=("warm_start", "restart_from_random", "restart_from_baseline"),\n'
        '                        default="restart_from_baseline")',
    )
    source = replace_once(
        source,
        '    if args.bootstrap_only and not args.run:\n',
        '    if not args.bootstrap_source_suite_id:\n'
        '        raise ValueError("Task 5 requires --bootstrap-source-suite-id.")\n'
        '    if args.bootstrap_source_suite_id == args.suite_id:\n'
        '        raise ValueError("Use a distinct suite ID for Task 5.")\n'
        '    if args.training_initialization != "restart_from_baseline":\n'
        '        raise ValueError("Task 5 must use restart_from_baseline; do not silently change the experiment.")\n'
        '    if not args.require_existing_bootstrap:\n'
        '        raise ValueError("Task 5 must reuse an existing, validated L0 bootstrap.")\n'
        '    if args.bootstrap_only and not args.run:\n',
    )
    source = replace_once(
        source,
        '        / args.suite_id\n        / "shared"\n        / f"seed_{args.seed}"\n        / "bootstrap_training"',
        '        / args.bootstrap_source_suite_id\n        / "shared"\n        / f"seed_{args.seed}"\n        / "bootstrap_training"',
    )
    source = replace_once(
        source,
        '        "suite_id": args.suite_id,\n',
        '        "suite_id": args.suite_id,\n'
        '        "bootstrap_source_suite_id": args.bootstrap_source_suite_id,\n',
    )
    source = replace_once(
        source,
        '        "cycle_training_initialization": "restart_from_random",',
        '        "cycle_training_initialization": args.training_initialization,',
    )
    source = replace_once(
        source,
        '        "vit_tower": "random_initialization_each_training_stage",',
        '        "vit_tower": "L0_fixed_baseline_each_cycle",',
    )
    source = replace_once(
        source,
        '        run_id=f"{args.suite_id}_bootstrap_seed{args.seed}",',
        '        run_id=f"{args.bootstrap_source_suite_id}_bootstrap_seed{args.seed}",',
    )
    source = replace_once(
        source,
        '        training_initialization="restart_from_random",',
        '        training_initialization=args.training_initialization,',
    )
    return source.replace(
        '"""Prepare or run Experiment B: historical AstroMiNN AL replay."""',
        '"""Task 5: closed-loop historical replay with baseline-restarted AstroMiNN ViT."""', 1,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    args = p.parse_args()
    src = args.repo / "scripts/run_astrominn_historical_replay.py"
    dst = args.repo / "scripts/run_astrominn_closed_loop_restart.py"
    generated = build_runner(src.read_text(encoding="utf-8"))
    if dst.exists() and dst.read_text(encoding="utf-8") != generated:
        raise ValueError(f"Refusing to overwrite a modified Task 5 runner: {dst}")
    dst.write_text(generated, encoding="utf-8")
    print(f"Generated {dst}")


if __name__ == "__main__":
    main()
