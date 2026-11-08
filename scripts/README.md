# Script navigation

This repository contains a reusable active-learning library **and** the
reproducibility code for individual AppleCiDEr experiments. These are not the
same user interface.

## For new users

Start with the [root README](../README.md), the
[testing guide](../docs/testing.md), and the installable library in
`src/al_pipeline/`. Its general selection, pool, scoring, budget, oracle and
checkpoint APIs are in `src/al_pipeline/core/`. The
`src/al_pipeline/astronomy/` package contains astronomy-specific adapters.

The experimental files below are **not** generic public quickstarts:
they can assume historic data, model weights, external upstream packages or
Delta/HPC-specific settings. Portable training-school notebooks and their CPU runners are already included;
see [examples](../examples/README.md).

## Portable teaching examples

- [Notebook 01: handwritten digits](../examples/01_active_learning_digits.ipynb) — an interactive CPU acquisition loop, with guided exercises.
- [Notebook 02: astronomical transients](../examples/02_applecider_astronomy_active_learning.ipynb) — offline analysis of real archived AstroMiNN learning curves, class scores and cumulative acquisition overlap.

After installing `.[teaching]` from the repository root, these commands execute the student notebooks without changing any research experiment:

```bash
python scripts/examples/execute_digits_notebook.py --mode core --output digits_core_local.ipynb
python scripts/examples/execute_astro_notebook.py --output astro_school_local.ipynb
```

The read-only teaching-data exporters are under `scripts/examples/`; unlike the notebooks, re-exporting source records requires access to the original research artifacts.

## Research and historical runs

- [Original frozen-expert AstroMiNN experiments](research/historical_replay/README.md):
  Tasks 4–18, with experiment-family subdirectories and separate source
  snapshots. The advanced acquisition experiments (TAILOR, SIMILAR, DIRECT,
  BALD, etc.) are preserved as distinct scientific comparisons.
- [Later ConvNeXt image+metadata historical model](research/fusion_image/README.md):
  cache and resident-memory training-recipe workflows with validation-only
  recipe extensions.
- [Task 0 / Task 1 grouped entry points](research/fusion_image/task0_task1_README.md):
  later image+metadata initialization comparison and historical active learning.
  Their original `adaptive/` scripts remain byte-for-byte intact so previously
  queued Slurm jobs and frozen source hashes are not broken. New grouped
  launchers use separate output identities, not the original suite IDs.
- `adaptive/rep5bc*`: separate representation diagnostics of the earlier
  model; not a replacement or alternate implementation of fusion Task 1.

## Archived, not deleted

[Earlier fusion-image prototypes](archive/fusion_image/README.md)
preserve original L0-only cache/benchmark/matrix launchers, original
pre-Task0 sharded and fastcache training variants, and exploratory diagnostics.
They are not recommended current commands. Scientific recipe modules
and their regression tests remain available in `src/` and `tests/`.

Old frozen studies may contain absolute `scripts/adaptive/` paths with
source hashes. Do **not** overwrite old sources or reuse their suite IDs with
scripts moved since their snapshots were created. In particular, archived
launchers still have hardcoded cluster assumptions and require independent
historical environment/provenance verification.

For a detailed file-by-file classification, see
[script_inventory.csv](../docs/maintenance/script_inventory.csv).
For a safe staged migration methodology, see
[script_cleanup_plan.md](../docs/maintenance/script_cleanup_plan.md).

## Testing contract

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

This portable default tests the library without Delta data. Run a focused
research-family test module explicitly to validate an experiment's synthetic
contracts. The full historical suite may require the optional
`APPLECIDER_TASK11_SOURCE` fixture, PyTorch and other pinned dependencies,
and is not a clean-machine installation requirement.
