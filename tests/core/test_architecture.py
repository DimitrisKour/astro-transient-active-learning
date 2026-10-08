"""Verify isolation promised by the standalone package."""

import ast
import subprocess
import sys
from pathlib import Path

import al_pipeline


def test_core_has_no_astronomy_or_old_package_imports():
    core = Path(al_pipeline.__file__).parent / "core"
    for path in core.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any(
                name.startswith(("al_pipeline.astronomy", "applecider", "torch", "hyrax", "flare"))
                for name in names
            ), path


def test_basic_workflow_imports_with_optional_integrations_blocked():
    script = """
import importlib.abc
import sys

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("al_pipeline.astronomy", "applecider", "torch", "hyrax", "flare", "astropy", "pandas", "matplotlib", "sklearn")):
            raise AssertionError("Unexpected optional import: " + fullname)

sys.meta_path.insert(0, BlockOptionalImports())
from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.workflow import ActiveLearningResolutionWorkflow
from al_pipeline.core.state import SQLiteObjectHistoryStore
from al_pipeline.core.training import TrainingRequest
from al_pipeline.core.evaluation import ModelEvaluator
from al_pipeline.core.experiments import SimulatedALConfig
"""
    subprocess.run([sys.executable, "-c", script], check=True)
