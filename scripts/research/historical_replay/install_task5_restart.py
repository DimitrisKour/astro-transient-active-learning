#!/usr/bin/env python3
"""Materialize an isolated Task 5 baseline-restart closed-loop package from Task 4.

Run in a checkout containing the complete Task 4 v2 scripts. No Task 4 code,
source trees, runs, or checkpoints are changed. Existing DIFFERENT Task 5 files
are never overwritten; identical files are reused.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_NAMES = (
    "build_runner.py", "prepare.py", "preflight.py", "schedule.py",
    "cycle.sbatch", "common.sh", "submit.sh", "enqueue.sh", "report.py",
)


def replace_exact(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected exactly one anchor {old!r}, found {count}")
    return source.replace(old, new, 1)


def remake(name: str, original: str) -> str:
    """Convert a checked Task 4 component to its Task 5 counterpart."""
    if name not in SCRIPT_NAMES:
        raise ValueError(f"Unexpected source file: {name}")
    text = original
    # Retain the genuine Task 4 v2 implementation; only isolate its wiring.
    for old, new in (
        ("task4_", "task5_"), ("TASK4_", "TASK5_"),
        ("task4", "task5"), ("TASK4", "TASK5"),
        ("Task 4", "Task 5"), ("Task4", "Task5"), ("Task-4", "Task-5"), ("T4_", "T5_"),
    ):
        text = text.replace(old, new)

    if name == "common.sh":
        text = replace_exact(text,
            'TASK5_SUITE_ID="astrominn_closed_loop_warm_v1"',
            'TASK5_SUITE_ID="astrominn_closed_loop_baseline_restart_v1"', label=name)
        text = replace_exact(text,
            '--require-existing-bootstrap --training-initialization warm_start',
            '--require-existing-bootstrap --training-initialization restart_from_baseline', label=name)
    elif name == "build_runner.py":
        text = replace_exact(text,
            '"scripts/run_astrominn_closed_loop.py"',
            '"scripts/run_astrominn_closed_loop_restart.py"', label=name)
        text = replace_exact(text, 'default="warm_start")',
            'default="restart_from_baseline")', label=name)
        text = replace_exact(text, 'args.training_initialization != "warm_start"',
            'args.training_initialization != "restart_from_baseline"', label=name)
        text = replace_exact(text,
            'Task 5 must use warm_start; do not silently change the experiment.',
            'Task 5 must use restart_from_baseline; do not silently change the experiment.', label=name)
        text = replace_exact(text, 'L0_then_previous_loss_best_checkpoint',
            'L0_fixed_baseline_each_cycle', label=name)
        text = text.replace('with warm-started AstroMiNN ViT.',
            'with baseline-restarted AstroMiNN ViT.')
    elif name == "prepare.py":
        text = replace_exact(text, '"run_astrominn_closed_loop.py"',
            '"run_astrominn_closed_loop_restart.py"', label=name)
    elif name in ("submit.sh", "cycle.sbatch"):
        text = replace_exact(text, 'scripts/run_astrominn_closed_loop.py',
            'scripts/run_astrominn_closed_loop_restart.py', label=name)
    elif name == "report.py":
        text = replace_exact(text, "'astrominn_closed_loop_warm_v1'",
            "'astrominn_closed_loop_baseline_restart_v1'", label=name)
        previous_block = '''                for field in ("inference_checkpoint", "training_initial_checkpoint"):
                    if Path(summary[field]) != previous:
                        raise ValueError(f"Checkpoint ancestry mismatch at {policy} seed {seed} c{c}: {field}")'''
        restart_block = '''                if Path(summary["inference_checkpoint"]) != previous:
                    raise ValueError(f"Inference ancestry mismatch at {policy} seed {seed} c{c}")
                if Path(summary["training_initial_checkpoint"]) != bootstrap:
                    raise ValueError(f"Training baseline ancestry mismatch at {policy} seed {seed} c{c}")
                training_summary_file = artifact_root / f"cycle_{c:03d}" / "training" / "training_summary.json"
                if not training_summary_file.is_file():
                    raise ValueError(f"Missing training summary at {policy} seed {seed} c{c}")
                training_summary = json.loads(training_summary_file.read_text())
                if Path(training_summary["initial_checkpoint"]) != bootstrap:
                    raise ValueError(f"Trainer used wrong initialization at {policy} seed {seed} c{c}")'''
        text = replace_exact(text, previous_block, restart_block, label=name)
    # Keep the existing Task 4 scheduler and enqueue behavior otherwise EXACT.
    return text


def planned_changes(repo: Path) -> dict[Path, str]:
    src = repo / "scripts/research/historical_replay/task4"
    dest = repo / "scripts/research/historical_replay/task5"
    changes: dict[Path, str] = {}
    for name in SCRIPT_NAMES:
        source = src / f"task4_{name}"
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing {source}; check out feat/entropy-batch-policies with Task 4 v2 first.")
        original = source.read_text(encoding="utf-8")
        generated = remake(name, original)
        target = dest / f"task5_{name}"
        changes[target] = generated
    # Sanity-check the three key scientific constraints BEFORE writing files.
    assert '--training-initialization restart_from_baseline' in changes[src / "task5_common.sh"]
    assert 'L0_fixed_baseline_each_cycle' in changes[src / "task5_build_runner.py"]
    assert 'summary["inference_checkpoint"]' in changes[src / "task5_report.py"]
    assert 'summary["training_initial_checkpoint"]' in changes[src / "task5_report.py"]
    assert 'training_summary["initial_checkpoint"]' in changes[src / "task5_report.py"]
    return changes


def install(repo: Path) -> list[Path]:
    """All-or-nothing conflict validation, followed by idempotent creation."""
    changes = planned_changes(repo)
    conflicts = [p for p, v in changes.items() if p.exists() and p.read_text(encoding="utf-8") != v]
    if conflicts:
        raise FileExistsError("Refusing to overwrite modified Task 5 files:\n" + "\n".join(map(str, conflicts)))
    for path, text in changes.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            if path.suffix in (".sh", ".sbatch"):
                path.chmod(0o755)
    return list(changes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    files = install(args.repo.resolve())
    print(f"Installed/verified {len(files)} isolated Task 5 scripts in {args.repo}")
    print("Next: python -m pytest tests/jobs/test_task5_restart.py -q")
    print("Then: bash scripts/research/historical_replay/task5/task5_submit.sh --prepare-only")
    print("No GPU jobs have been submitted.")


if __name__ == "__main__":
    main()
