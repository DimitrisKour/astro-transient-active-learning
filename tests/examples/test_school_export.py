"""Check exported Rubin school materials are standalone and versioned."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = SOURCE_ROOT / "scripts/examples/export_rubin_school.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_rubin_school", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_school_export_has_executable_notebooks_and_bundled_data(tmp_path):
    module = _load()
    school = tmp_path / "Rubin-ML-Training-School"
    (school / ".git").mkdir(parents=True)
    session = school / "day3/01_active_learning"
    session.mkdir(parents=True)
    (session / "README.md").write_text("School placeholder\n", encoding="utf-8")

    module.export(school, overwrite=True)
    manifest = json.loads((session / "source_manifest.json").read_text())
    assert len(manifest["source_commit"]) == 40
    assert "__SOURCE_COMMIT__" not in (session / "README.md").read_text()
    assert "al-pipeline[teaching]" in (session / "README.md").read_text()
    assert (session / "LICENSE_AL_PIPELINE").read_text().startswith("MIT License")
    assert (session / "run_notebooks.py").is_file()

    for name in ("01_active_learning_digits.ipynb",
                 "02_applecider_astronomy_active_learning.ipynb"):
        notebook = json.loads((session / name).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["school_export"]["source_commit"]
        code = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        for source in code:
            ast.parse(source)
        assert any(cell.get("outputs") for cell in notebook["cells"]
                   if cell.get("cell_type") == "code")

    astronomy = json.loads((session / "02_applecider_astronomy_active_learning.ipynb").read_text())
    code = "\n".join("".join(cell.get("source", [])) for cell in astronomy["cells"]
                     if cell["cell_type"] == "code")
    assert "Path('teaching_data')" in code
    assert "Path('day3/01_active_learning/teaching_data')" in code
    assert "Path('examples/data')" not in code
    for filename in module.DATA_FILES:
        file = session / "teaching_data" / filename
        assert file.is_file()
        assert f"teaching_data/{filename}" in manifest["content"]
    for file in manifest["content"]:
        assert (session / file).is_file()

    try:
        module.export(school, overwrite=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Exporter should refuse silent overwrites")


def test_school_runner_is_standalone_and_refuses_overwriting():
    runner = (SOURCE_ROOT / "scripts/examples/school_templates/run_notebooks.py").read_text()
    ast.parse(runner)
    assert "NotebookClient" in runner
    assert "output.exists()" in runner
    assert "kernel_name=kernel" in runner
