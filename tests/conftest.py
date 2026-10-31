"""Portable integration fixture for the immutable Task 11 frozen source.

On Delta this resolves the real frozen Task11 directory. For development/CI,
set APPLECIDER_TASK11_SOURCE to a verified Task11 source directory (or its parent).
An absent fixture is an error, never a passing-but-skipped integration check.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest


DEFAULT_TASK11 = Path(
    "/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/"
    "astrominn_task11_advanced_acquisition_v1/source"
)


@pytest.fixture(scope="session")
def task11_frozen_source() -> Path:
    configured = os.environ.get("APPLECIDER_TASK11_SOURCE")
    candidate = Path(configured).expanduser() if configured else DEFAULT_TASK11
    candidate = candidate.resolve()
    source = candidate if (candidate / "src/al_pipeline").is_dir() else candidate / "source"
    if not ((source / "scripts/run_astrominn_closed_loop.py").is_file()
            and (source / "src/al_pipeline/astronomy/experiments/simulated.py").is_file()):
        pytest.fail(
            "Frozen Task11 source not found at " + str(source) + ". "
            "Set APPLECIDER_TASK11_SOURCE to the Task11 frozen source directory "
            "(or its parent); these integration tests must not silently skip."
        )
    return source
