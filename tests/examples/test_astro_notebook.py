"""Validate the astronomy notebook's real-data, read-only CPU teaching flow."""
from __future__ import annotations

import ast
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples/02_applecider_astronomy_active_learning.ipynb"


def test_historical_notebook_executes_without_images_or_model_training():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    codes = [c for c in nb.cells if c.cell_type == "code"]
    assert len(codes) >= 9
    for code in codes:
        ast.parse(code.source)
    text = "\n".join(c.source for c in nb.cells)
    assert "astro_recent_validation.csv" in text
    assert "examples/data" in text
    assert "make_synthetic_demo" not in text
    assert "Felipe" not in text
    assert "TODO:" in text
    assert "len(policies) <= 3" in text
    result = NotebookClient(nb, timeout=180, kernel_name="python3",
                            allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}}).execute()
    images = [o["data"]["image/png"] for c in result.cells
              if c.cell_type == "code" for o in c.get("outputs", ())
              if o.get("data", {}).get("image/png")]
    assert len(images) == 4
    streams = "\n".join(str(o.get("text", ""))
                        for c in result.cells for o in c.get("outputs", ())
                        if o.get("output_type") == "stream")
    assert "51 trajectories" in streams
    assert "Contract checks passed" in streams