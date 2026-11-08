#!/usr/bin/env python3
"""Execute the CPU-only teaching notebook and save its figures and tables.

Run from any working directory with:
  python scripts/examples/execute_digits_notebook.py --output /path/to/result.ipynb
Never overwrites the source notebook or submits scheduler jobs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep small scikit-learn fits and BLAS bounded on shared CPU login nodes.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples" / "01_active_learning_digits.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path.cwd() / "digits_active_learning_executed.ipynb",
        help="Writable destination for the executed notebook; source is not modified.",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == SOURCE.resolve():
        parser.error("The output cannot overwrite the original teaching notebook.")
    if output.suffix.lower() != ".ipynb":
        parser.error("The output file must have a .ipynb extension.")
    if output.exists():
        parser.error(f"Output already exists: {output} (remove explicitly to rerun)")
    output.parent.mkdir(parents=True, exist_ok=True)

    notebook = nbformat.read(SOURCE, as_version=4)
    nbformat.validate(notebook)
    executed = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    ).execute()
    figure_count = sum(
        "image/png" in part.get("data", {})
        for cell in executed.cells if cell.cell_type == "code"
        for part in cell.get("outputs", [])
        if part.get("output_type") in ("display_data", "execute_result")
    )
    if figure_count < 5:
        raise RuntimeError(
            f"Expected at least five embedded figures, found {figure_count}; "
            "refusing to save an incomplete teaching notebook."
        )

    nbformat.write(executed, output)
    print(f"Executed notebook saved: {output}")
    print(f"Embedded figures: {figure_count}")
    print(f"Cells: {len(executed.cells)}, "
          f"code cells: {sum(c.cell_type == 'code' for c in executed.cells)}")
    print("No astronomical data, GPU, or Slurm submission was used.")


if __name__ == "__main__":
    main()
