"""Explicit fixture for tests that reproduce frozen historical experiments.

Portable tests do not request this fixture. Historical tests requiring Task 11
must opt in with APPLECIDER_TASK11_SOURCE; no cluster/user-specific default.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def task11_frozen_source() -> Path:
    configured = os.environ.get("APPLECIDER_TASK11_SOURCE")
    if not configured:
        pytest.fail(
            "The historical Task 11 source fixture requires "
            "APPLECIDER_TASK11_SOURCE to be set explicitly. "
            "The portable suite runs with python -m pytest; "
            "for historical reproduction configure this variable and run "
            "python -m pytest tests."
        )
    candidate = Path(configured).expanduser().resolve()
    source = candidate if (candidate / "src/al_pipeline").is_dir() else candidate / "source"
    required = (
        source / "scripts/run_astrominn_closed_loop.py",
        source / "src/al_pipeline/astronomy/experiments/simulated.py",
    )
    if not all(path.is_file() for path in required):
        pytest.fail(
            "Frozen Task 11 source not found at " + str(source) + ". "
            "APPLECIDER_TASK11_SOURCE must point to a frozen source directory "
            "(or its parent) containing both expected files."
        )
    return source
