#!/usr/bin/env python3
"""Compatibility import and CLI for the relocated Task5 bootstrap installer."""
from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
namespace = runpy.run_path(
    str(ROOT / "scripts/research/historical_replay/install_task5_restart.py")
)
SCRIPT_NAMES = namespace["SCRIPT_NAMES"]
replace_exact = namespace["replace_exact"]
remake = namespace["remake"]
planned_changes = namespace["planned_changes"]
install = namespace["install"]
main = namespace["main"]

if __name__ == "__main__":
    main()
