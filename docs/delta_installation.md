# Delta installation for historical research

This guide describes the existing Delta/HPC setup for the **historical astronomical experiments**. A local clone can run the [core and teaching notebooks](../README.md) without the steps below. The original research checkout remains under the directory name `active-learning`; that on-disk name is independent of the GitHub repository name. Do not move it while frozen jobs, environments, or absolute artifact paths still refer to it.

The historical checkout, upstream clones, datasets, checkpoints, and experiment artifacts are preserved.

## Environment and editable installs

These are first-time setup commands. If the cloned `al-pipeline` environment
and editable installs already exist, skip this section and use the runtime
setup below. Conda cloning copies the environment's packages, not repository
source, datasets, or model weights. The editable installs use the sibling
upstream AppleCiDEr checkout and this extracted repository.

Clone the working environment so the current experiments keep their dependencies:

```bash
conda create --name al-pipeline --clone applecider
conda activate al-pipeline
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning

# Guard against running package changes in the original environment.
test "$CONDA_DEFAULT_ENV" = al-pipeline

# Remove the old fork's editable package from this cloned environment only.
python -m pip uninstall -y applecider
python -m pip install --no-deps --no-build-isolation -e ../applecider
python -m pip install --no-deps --no-build-isolation -e .

```

After installation, use the runtime setup below for import checks, then run
`al_python -m pytest -q`. If pytest is absent, install it in the cloned
environment. The extraction itself
does not require reinstalling or upgrading PyTorch, Hyrax, or FLARE. FLARE can
be installed from `../flare` when the corresponding notebooks are migrated.

`configs/delta.toml` is ignored by Git. The bundle's apply script can copy the
existing local registry into that path without changing its contents. Review
that it still points to the current HDD/NVMe locations and historical artifacts.
The new example config is documentation, not a replacement for the live registry.

## Runtime setup in each Delta shell

Activate the extracted environment after entering a CPU or GPU allocation.
The tested Delta session otherwise selected `/lib64/libstdc++.so.6` before
Conda's runtime and failed while importing SQLite/ICU with
`CXXABI_1.3.15 not found`. The helper below loads this environment's C++ runtime
first for each Python process. It does not set a global `LD_PRELOAD`, modify
shell startup files, or change either environment's packages.

```bash
conda activate al-pipeline
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning

al_python() {
  env LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" python "$@"
}
```

First check imports and argument parsing:

```bash
al_python -c 'import torch, sqlite3, al_pipeline; print(al_pipeline.__file__); print("SQLite", sqlite3.sqlite_version)'
al_python scripts/run_astrominn_simulated_al.py --help
al_python scripts/run_astrominn_historical_replay.py --help
al_python scripts/report_astrominn_multiseed.py --help
```

Define `al_python` again in each new shell, including the shell opened by
`srun`. Use it in place of `python` for the launch examples in the
[A guide](simulated_active_learning_runs.md) and
[B guide](astrominn_historical_replay_experiment_b.md). For a batch job, put
this activation, working directory, and helper definition in the job script.

Omitting `--run` prepares and prints a plan without model inference or training.
A can write its small holdout manifest; B writes and audits its small split
manifests. Existing dataset packets and model weights remain at their current
Delta paths. GPU training examples must run inside a GPU allocation.

## Suite IDs and resuming

The guides reserve `al_pipeline_experiment_a_v1` and
`al_pipeline_experiment_b_v1` for full-budget runs, and
`al_pipeline_check_a_v1` and `al_pipeline_check_b_v1` for one-epoch checks.
Use an unused ID if any example ID already has run products. Keep the same ID
and all input/training settings when resuming an interrupted run.

Use a new suite ID when changing the epoch budget, patience, initialization,
strategy set, seed, acquisition budget, or input data. In particular, do not
promote a one-epoch smoke suite into a full-budget experiment by changing its
arguments: it may reuse completed checkpoints. The existing original runs,
`al_pipeline_migration_a_v1`, `al_pipeline_migration_b_v1`, and
`al_pipeline_smoke_a_v1` / `al_pipeline_smoke_b_v1` remain separate.

## Checks after a documentation patch

```bash
git diff --check
al_python scripts/run_astrominn_simulated_al.py --help >/dev/null
al_python scripts/run_astrominn_historical_replay.py --help >/dev/null
al_python scripts/report_astrominn_multiseed.py --help >/dev/null
git check-ignore configs/delta.toml
```

After verification, commit the extracted repository and choose its GitHub remote.
No GitHub repository or remote is created by the extraction bundle.
