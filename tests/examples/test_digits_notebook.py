"""Standalone offline CPU execution checks for the teaching notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples" / "01_active_learning_digits.ipynb"


def test_notebook_format_all_code_parses_and_exercises_are_opt_in():
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert raw["nbformat"] == 4
    cells = raw["cells"]
    assert len(cells) >= 30
    assert len({cell["id"] for cell in cells}) == len(cells)
    codes = [cell["source"] for cell in cells if cell["cell_type"] == "code"]
    assert len(codes) >= 15
    for code in codes:
        ast.parse(code)
    assert "RUN_EXERCISES = False" in codes[0]
    assert "RUN_OPTIONAL_EXPERIMENTS = False" in codes[0]
    joined = "\n".join(codes)
    assert "ExistingLabelsOracle" in joined
    assert "ActiveLearningLoop" in joined
    assert "EntropyScorer" in joined
    assert "MarginScorer" in joined
    assert "EvaluationHistory" in joined
    assert "compute_classification_metrics" in joined
    assert "plot_strategy_comparison" in joined
    assert "load_digits" in joined


def test_notebook_executes_without_gpu_network_or_private_data():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    client = nbclient.NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    finished = client.execute()
    failures = [
        (i, output.get("ename"), output.get("evalue"))
        for i, cell in enumerate(finished.cells)
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not failures, failures
    assert all(cell.get("execution_count") is not None
               for cell in finished.cells if cell.cell_type == "code")

    # Matplotlib may silently suppress figures under MPLBACKEND=Agg.
    # Require the teaching visuals to be embedded in the executed notebook.
    png_outputs = [
        output["data"]["image/png"]
        for cell in finished.cells if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") in ("display_data", "execute_result")
        and "image/png" in output.get("data", {})
    ]
    assert len(png_outputs) >= 5, (
        f"Expected five or more embedded figures; got {len(png_outputs)}"
    )
    assert all(len(png) > 1000 for png in png_outputs)
