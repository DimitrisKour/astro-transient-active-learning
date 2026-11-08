# Task 11 packaging repair v1.1

Issue: importing `al_pipeline.core.advanced_acquisition.badge` initializes the parent `advanced_acquisition` package and eagerly imports the optional `ensemble_disagreement` scorer. That scorer depends on `al_pipeline.core.prediction`, which is not included in the deliberately minimal Task 4-derived source snapshot. This causes ModuleNotFoundError before the real-data preflight can proceed.

Repair: remove only the unrelated eager `ensemble_disagreement` import and corresponding `__all__` entry from the **Task 11 frozen copy** of `src/al_pipeline/core/advanced_acquisition/__init__.py`. The full Task 4 baseline, Task 11 BADGE/core-set/Fisher algorithms, training setup, scientific plan, checkpoint paths, and datasets remain unchanged.

The repair verifies all frozen source hashes, refuses if an acquisition pilot or training submission has begun, backs up the original `__init__.py`, refreshes the frozen hash manifest, and records the exact old/new file hashes in `import_dependency_repair_v1_1.json` outside the source directory. It is idempotent and auditable. This is not a change to acquisition-policy design.

Usage from checkout after transferring/unzipping:

```bash
python -m pytest tests/jobs/test_task11_repair_import.py -q
python scripts/research/historical_replay/task11/task11_repair_import.py
python scripts/research/historical_replay/task11/task11.py verify
python scripts/research/historical_replay/task11/task11.py preflight
```

Keep the Conda `LD_PRELOAD` workaround set in the interactive shell. Do not run pilot jobs until the preflight passes.
