# Task 4 / Task 5 — historical closed-loop AstroMiNN comparison

These files reproduce the **original historical replay** comparison, not the
recommended public API or an up-to-date broker-level implementation.

| Experiment | Training initialization | Runner |
| --- | --- | --- |
| Task 4 | Continue from previously deployed checkpoint (warm start) | `scripts/run_astrominn_closed_loop.py` |
| Task 5 | Restart from the same seed's original L0 checkpoint each cycle | `scripts/run_astrominn_closed_loop_restart.py` |

Both experiments acquire new historical objects, evaluate on a fixed validation
set, and preserve a frozen source tree and registry per suite. The two runners
are intentionally separate to preserve their original recorded provenance.

## Directory structure

```text
scripts/research/historical_replay/
  README.md                     # this guide
  install_task5_restart.py      # reproducibly derive Task5 from Task4
  task4/                        # controller, frozen-source builder, Slurm, report
    task4_submit.sh             # Task4 entry point
    task4_prepare.py
    task4_preflight.py
    task4_schedule.py
    task4_enqueue.sh
    task4_cycle.sbatch
    task4_common.sh
    task4_build_runner.py
    task4_report.py
  task5/                        # analogous baseline-restart workflow
    task5_submit.sh
    ...
```

Only the original Task4 `task4_extend.sh` was disabled by design; it was moved
to `scripts/archive/historical_replay/task4_extend_disabled.sh` rather than
presenting it as an active launcher.

## Validate locally (no Delta, GPU or data)

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/jobs/test_task4_package.py tests/jobs/test_task5_restart.py -q
```

These tests verify runner derivation, frozen-source source hashes, checkpoint
ancestry, deterministic job planning, mocked Slurm dependency/release logic,
and shell syntax. They must pass before changing either script family's layout.

## Historical Delta submission

The following paths are retained for **research reproduction**, not a public
quickstart. They rely on the historical registry, Task4/Task5 bootstrap and data
layout, and the Delta Slurm scheduler. `--prepare-only` never submits jobs:

```bash
bash scripts/research/historical_replay/task4/task4_submit.sh --prepare-only
bash scripts/research/historical_replay/task5/task5_submit.sh --prepare-only
```

Do **not** submit these experiments merely to validate the cleanup. If
intentionally rerunning historical research on Delta, consult
`docs/task4_closed_loop_warm_start.md` and
`docs/task5_closed_loop_baseline_restart.md`; their explicit submission
commands enqueue GPU jobs.

For older notes or scripts, these two paths continue to work as forwarding
entry points (no scientific logic lives in them):

```bash
bash scripts/adaptive/task4_submit.sh --prepare-only
bash scripts/adaptive/task5_submit.sh --prepare-only
```

Likewise, `scripts/adaptive/install_task5_restart.py` remains a compatibility
adapter for the canonical `scripts/research/historical_replay/install_task5_restart.py`.

## Frozen source compatibility

Existing completed Delta suites remain under their **original** frozen
`<suite>/source/scripts/adaptive/task4_*` and
`<suite>/source/scripts/adaptive/task5_*` paths; none have been edited.

Fresh **reproduction** suites built from this reorganized checkout freeze
the new `scripts/research/historical_replay/task4/` and
`task5/` paths instead. Therefore, do not mix a new checkout's preparation
or resumption commands with a previously frozen source, even if the underlying
scientific policy is identical. Existing suite hash verification takes
precedence over convenience. Coordinate a separate suite ID for new
comparisons; reusing the original suite directory may rightly fail its
immutability checks.

The public-facing portable API is documented in the repository README and
`docs/testing.md`; this family is intentionally non-default.
