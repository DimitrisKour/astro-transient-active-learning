# Delta migration validation — 2026-10-08

- Test suite: 316 passed in 29.38 seconds.
- Experiment A: original and extracted holdout rows and order match.
- Experiment B: initial labeled rows, hidden labels, candidate rows,
  and row order match; only the suite-specific pool_id differs.
- B historical training, validation, and test source hashes match.
- GPU smoke A: entropy, 50 acquisitions, one cycle, one epoch.
- GPU smoke B: one-epoch bootstrap, entropy, 473 acquisitions,
  one cycle, one training epoch.
- Both smoke launchers completed cycle 1.
- Full-budget numerical reproduction remains unverified.

Smoke suites: al_pipeline_smoke_a_v1 and al_pipeline_smoke_b_v1.
Logs: ../migration-validation/smoke_a_v1.log and smoke_b_v1.log.

The al-pipeline environment requires its C++ runtime to be loaded first
in the tested Delta session. Apply this inside a launch subshell:

    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

Keep existing dataset, model, and legacy-code paths for current Delta work.
