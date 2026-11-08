# Archived fusion-image L0 experiments

This directory preserves **early exploratory and L0-only pipelines**, not the
recommended image+metadata AstroMiNN research workflow. All original scientific
comparison variants remain available, but their launchers have been relocated
here so they do not appear alongside the later full-history Task 0 / Task 1
controllers.

## Provenance and scope

- `fusion_cache_v2.py`, its shard/finalize/benchmark workers, and
  `submit_fusion_cache_v3.sh` create the **earlier initial L0 + validation**
  cache. It is not the complete historical training/candidate pool.
- `fusion_l0_matrix.py` and its workers, coordinated by
  `submit_fusion_l0_matrix.sh`, compare initial L0 models and initialization.
  This is **not** the full-history Task 0 comparison.
- `fusion_cached_l0_benchmark_v2.py` and
  `fusion_cached_l0_full_seed135.py` (plus worker) are isolated performance /
  baseline probes with old cache identities.

This is an archive of **distinct studies**, not a claim that their model
training methods are mathematically identical to the final recipe.

## Later research path

For the full historical 4,729-object training and fixed validation data,
the later data pipeline is `fusion_full_historical_cache_v1.py` and
`submit_fusion_full_historical_cache_v1.sh`, followed by resident-cache
training-recipe experiments, `task0_full_history.py` and the Task 1
closed-loop controls. Those remain in `scripts/adaptive/` until their own
dependency-tested migration.

The recommended public **package entry points** will be clarified by the
training-school examples. None of these archived scripts is currently
advertised as a portable/standalone command.

## Historical compatibility warning

Original completed or held experiments may have immutable manifests with
`scripts/adaptive/` paths and SHA-256 checksums. Moving tracked scripts in
Git does **not** modify those old snapshots, and archived checkout scripts
**must not be used to resume or overwrite them**. Run any original frozen job
using its own verified snapshot; for a new reproduction, use a separate suite
identifier and verify external data, dependencies, source identities and
Delta-specific directories.

The original source files remain recoverable from the parent Git history
even when downstream historical reports reference their former location.
The archive launchers retain Delta-specific paths and are **not** tested as
standalone cross-machine workflows.

## Validation

`tests/astronomy/fusion_image/test_fusion_l0_matrix.py` now validates
the archived L0-matrix controller. The fusion image cache and recipe
regression tests remain in `tests/astronomy/fusion_image/` (opt-in astronomy
suite). The archive migration should not change production
`src/al_pipeline/astronomy/fusion_image/` code.

Do not submit Slurm jobs as a cleanup test.
