# Tasks 8–9: saved-checkpoint analysis, recovery and retention

These folders contain **historical reproduction scripts**, not the public
active-learning API. The commands assume the historical Task4/6B/6C and Task8B
artifacts and, for training, a configured Delta environment.

| Directory | Research responsibility |
| --- | --- |
| `task8a/` | Retrospective, validation-only checkpoint audits and CPU sharding |
| `task8b/` | First-rejection training-seed recovery and reporting |
| `task8c1/` | Closed-loop checkpoint retention, without retries |
| `task8c2/` | Closed-loop checkpoint retention with bounded training-seed retries |
| `task9/` | Shared validation-prediction reader, fixed ensembles, SWA continuation and BN refresh |

The Task8C.1 and Task8C.2 programs have different scientific acceptance rules.
Neither replaces the other. Task9B SWA and Task9B BN-refresh are separate
controls; do not remove the original variant simply because the latter is
labelled `bn_v2`.

## Safe local contract tests

No Slurm submissions, private datasets or real astrophysical model training:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q \
  tests/jobs/test_task8a_v1.py \
  tests/jobs/test_task8a_shards.py \
  tests/jobs/test_task8b.py \
  tests/jobs/test_task8c1_preparation.py \
  tests/jobs/test_task8c1_retention_decisions.py \
  tests/jobs/test_task8c2_retry_gate.py \
  tests/jobs/test_task8c2_schedule.py \
  tests/jobs/test_task9a_ensemble.py
```

Additional PyTorch-only SWA tests:

```bash
python -m pip install -e ".[dev]"
python -m pip install torch
python -m pytest -q \
  tests/jobs/test_task9b_swa.py \
  tests/jobs/test_task9b_swa_bn_v2.py
```

The dedicated CI workflow provides a test gate for the path migration.

## Frozen scientific source compatibility

Existing completed Delta snapshots keep their original
`<suite>/source/scripts/adaptive/task8*` or `task9*` paths. Their source
hashes and reference provenance have **not** been modified.

The regrouped locations belong to newly prepared research reproductions
under new suite IDs. Do not resume an already frozen suite with the new
checkout's source or try to override source hash mismatch checks.

The balanced-357 strict-retry V2 controllers are **not** part of this
migration. They remain at `scripts/adaptive/balanced357_*` until their active
research trajectories and provenance have been audited independently.
