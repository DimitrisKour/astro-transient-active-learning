"""Regression: Task13's generated frozen runner must import its identity settings dependency."""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research/historical_replay/task13"))


def test_task13_frozen_patch_includes_batch_acquisition_config(tmp_path, task11_frozen_source):
    from task13_patch import patch_source

    staged = tmp_path / "source"
    shutil.copytree(task11_frozen_source, staged)
    package = staged / "src/al_pipeline/core/meta_acquisition"
    package.mkdir(parents=True, exist_ok=True)
    for name in ("tailor.py", "mixed_selection.py"):
        shutil.copy2(ROOT / "src/al_pipeline/core/meta_acquisition" / name, package / name)
    patch_source(staged)

    simulated = staged / "src/al_pipeline/astronomy/experiments/simulated.py"
    content = simulated.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(simulated))
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "al_pipeline.core.experiments.batch_policy"
        for alias in node.names
    }
    assert "BatchAcquisitionConfig" in imports
    assert '"batch_policy": asdict(BatchAcquisitionConfig())' in content
    compile(tree, str(simulated), "exec")
