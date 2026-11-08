#!/usr/bin/env python3
"""Execute the CPU-only training-school digits notebook and validate exercises.

From the repository root after `pip install -e ".[teaching]"`:
    python scripts/examples/execute_digits_notebook.py --mode core --output ~/digits_core.ipynb
    python scripts/examples/execute_digits_notebook.py --mode optional --output ~/digits_optional.ipynb

Optional instructor-only solutions mode requires a *local* file at
examples/instructor/01_active_learning_digits_instructor.ipynb. Never publish
the instructor notebook in the participant repository.

The participant source is never overwritten; only embedded figures and
exercise completion markers are checked.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

for env_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(env_name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
PARTICIPANT_SOURCE = ROOT / "examples" / "01_active_learning_digits.ipynb"
INSTRUCTOR_SOURCE = ROOT / "examples" / "instructor" / "01_active_learning_digits_instructor.ipynb"


def set_flags(nb, *, exercises: bool, optional: bool):
    setup = nb.cells[2]
    original = setup.source
    for name, enabled in (("RUN_EXERCISES", exercises),
                          ("RUN_OPTIONAL_EXPERIMENTS", optional)):
        old = f"{name} = {not enabled}"
        new = f"{name} = {enabled}"
        if old in original:
            original = original.replace(old, new, 1)
        elif new not in original:
            raise RuntimeError(f"Missing configuration flag {name}")
    setup.source = original


def text_outputs(nb):
    return "\n".join(
        str(output.get("text", ""))
        for cell in nb.cells if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def validate_results(nb, mode):
    text = text_outputs(nb)
    image_count = sum(
        "image/png" in part.get("data", {})
        for cell in nb.cells if cell.cell_type == "code"
        for part in cell.get("outputs", [])
        if part.get("output_type") in ("display_data", "execute_result")
    )
    required = ["Selected from VALIDATION:"]
    if mode in ("optional", "solutions"):
        required += ["TODO 3 passed:", "TODO 4 passed:", "TODO 5 passed:",
                     "Entropy curves vary with initial L0: True"]
    if mode == "solutions":
        required += ["Exercise 1 passed!", "Exercise 2 passed!"]
    missing = [marker for marker in required if marker not in text]
    min_images = 8 if mode != "core" else 5
    if image_count < min_images or missing:
        raise RuntimeError(
            f"Incomplete {mode} notebook: found {image_count} embedded images, "
            f"expected >= {min_images}; missing markers={missing}"
        )
    return image_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("core", "optional", "solutions"),
                        default="core")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    source = INSTRUCTOR_SOURCE if args.mode == "solutions" else PARTICIPANT_SOURCE
    if not source.is_file():
        parser.error(f"Notebook source not found: {source}")
    out = (args.output or (Path.cwd() / f"digits_v3_{args.mode}_executed.ipynb")).expanduser().resolve()
    if out == source.resolve() or out.suffix.lower() != ".ipynb":
        parser.error("Output must be a distinct .ipynb file")
    if out.exists():
        parser.error(f"Output already exists (delete explicitly to rerun): {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(source, as_version=4)
    nbformat.validate(nb)
    set_flags(nb, exercises=args.mode == "solutions", optional=args.mode != "core")
    try:
        executed = NotebookClient(
            nb, timeout=600, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False,
        ).execute()
        figures = validate_results(executed, args.mode)
    except Exception:
        nbformat.write(nb, out)
        print(f"FAIL ({args.mode}): diagnostic notebook saved to {out}", flush=True)
        raise
    nbformat.write(executed, out)
    print(f"PASS ({args.mode}): {out}")
    print(f"Embedded figures: {figures}; cells: {len(executed.cells)}")
    print("Verified execution markers; no GPU, astronomical data or Slurm job needed.")


if __name__ == "__main__":
    main()
