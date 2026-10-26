Task 8C.1 registry [storage] scope correction (2026-10-24)

Cause:
  scripts/adaptive/task8c1.py:routed_registry used a whole-file regex for
  persistent_root and data_root, but Task 4 registry has multiple data_root
  fields in dataset/profile tables. This made 'prepare' fail before writing
  plan.json. The subsequent 'verify' FileNotFoundError is a consequence.

Patch changes ONLY:
  scripts/adaptive/task8c1.py
  tests/jobs/test_task8c1_preparation.py

Effect:
  Scope replacements strictly to the [storage] table; parse the source and
  rewritten TOML and verify every non-storage table is identical. Test the
  real failure pattern with four data_root assignments plus indented tables.
  No scientific settings, training data, reference snapshots or SLURM code
  are modified. No GPU work or jobs are triggered by installation/prepare.

Apply in the active-learning repository:
  unzip -o /projects/bcrv/dkourtesis/task8c1_registry_storage_scope_fix_v1.zip -d .
  python -m pytest tests/core/test_retention_gate_v1.py tests/jobs/test_task8c1_preparation.py tests/jobs/test_task8c1_retention_decisions.py -q
  python scripts/adaptive/task8c1.py prepare
  python scripts/adaptive/task8c1.py verify

Expected: 22 local tests pass; successful prepare writes the immutable plan.
Only after these succeed should you run submit-prefill. Keep the CPU replay
and GPU submission in their original, separately verified stages.
