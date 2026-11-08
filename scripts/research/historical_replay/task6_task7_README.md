# Historical AstroMiNN experiment families: Tasks 6–7

These scripts are **research reproduction controls**, not supported public
entry points. They require historical AstroMiNN/BOOM data, frozen reference
checkpoints and the Delta environment for real training. The scripts below
remain source-frozen: their worker paths, derivation scripts, registries and
checkpoint provenance are part of reproducibility.

| Family | Main scientific comparison |
| --- | --- |
| `task6a/` | Warm-start weight exponent 0.50 versus Task 4's exponent 0.85; separately repeated with fresh 0.50-weighted L0 |
| `task6b/` | Macro-F1-best checkpoint selection versus validation-loss-best |
| `task6c/` | Equal-object versus per-sample weighting with loss and macro-F1 checkpoint modes |
| `task7a/` | Temperature-scaled entropy acquisition using validation-only calibration |

The **canonical Task 4 runner generator** is now
`scripts/research/historical_replay/task4/task4_build_runner.py`.
Task 6A/6B/6C/7A builders explicitly reuse it; Task 6C also reuses the Task 6B
builder for its macro-F1 variant.

## Contract tests (local; no real training)

```bash
python -m pip install -e ".[dev]"
python -m pytest -q \
  tests/jobs/test_task6a_weight05.py \
  tests/jobs/test_task6a_independent_chains.py \
  tests/jobs/test_task6a_l0_v2.py \
  tests/jobs/test_task6b_v1.py \
  tests/jobs/test_task6c_v1.py \
  tests/jobs/test_task7a_v1.py \
  tests/astronomy/test_task6a_temperature_audit.py
```

The dedicated GitHub Actions workflow
`.github/workflows/task6-task7-packaging.yml` executes this test selection
without cluster data or GPU. Do not submit `sbatch` as a cleanup check.

## Frozen snapshots and old commands

Previously prepared experiments still have their **original**
`<suite>/source/scripts/adaptive/task6a_*`, `task6b_*`, `task6c_*`,
or `task7a_*` scripts and snapshot identities. Those frozen copies are not
modified and should be used to resume their original runs.

The reorganized paths are for fresh research reproduction under **new suite
IDs**. Do not mix the new checkout's source with the old frozen suite.
This migration does not change acquisition methods, training policy, or
checkpoint semantics, but it does change file hashes and paths, which matters
for frozen-source verification.

The portable public framework tests are documented separately in
`docs/testing.md`. Task 8/9 and balanced-357 controllers are **outside this
migration** and remain in their original locations until independently tested.
