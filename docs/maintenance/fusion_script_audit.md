# Fusion-image scripts: archival decisions and live-checkout migration gate

Audit date: 2026-11-08.

## What counts as the later reference image model

The later standalone AstroMiNN experiment uses a **BTSbot/ConvNeXt image
encoder and 55 metadata features**, mapped to the original five-class
AppleCiDEr taxonomy (AGN, TDE, SN I, SN II, CV). This is not the older
frozen-expert ViT and is not the upstream TEMPO+AstroMiNN full fusion system.

Historical engineering stages remain scientifically distinct:

1. Full-historical input preprocessing / label-free train+validation cache.
2. Training recipe comparisons; the later reference adopts resident-memory
   `cosine_aug` (weighted CE, D4 augmentation, warmup/cosine, loss-best
   validation checkpoint).
3. Task 0 pretrained-vs-random ConvNeXt initialization at fixed full-history
   train/validation membership, with separate sealed test comparison.
4. Task 1 historical active learning (random, entropy, entropy-diversity,
   balanced exploration), multiple initialization modes and seed replicates.
   This is an offline historical oracle simulation, not online broker work.

A later protocol superseding an earlier one **operationally** does not make
their scientific comparisons duplicates.

## Completed organizational changes

- [PR #8](https://github.com/DimitrisKour/al-pipeline/pull/8):
  12 early L0-only cache, performance benchmark, and initialization-matrix
  scripts moved to `scripts/archive/fusion_image/early_l0/`; original L0
  scientific contract and shell checks passed.
- [PR #9](https://github.com/DimitrisKour/al-pipeline/pull/9):
  11 original sharded-cache recipe v1, fastcache training v2 and bounded
  exploratory probes moved to `scripts/archive/fusion_image/`; original
  recipe numerical and path tests passed.
- [PR #10](https://github.com/DimitrisKour/al-pipeline/pull/10):
  14 full-history cache/HDD and resident training/recipe-extension files
  grouped in `scripts/research/fusion_image/cache/` and `recipe/`;
  cache, training and shell tests passed.

**Total: 37 scripts relocated, 23 archived and 14 grouped.**
Production source in `src/al_pipeline/astronomy/fusion_image/` and its
scientifically distinct training variants remain intact.

## Why 20 fusion-related scripts are still at scripts/adaptive/

The following are preserved in place for operational safety:

**Task 0 (9 files)**
- `task0_full_history.py`, `task0_full_history.sbatch`,
  `submit_task0_full_history.sh`
- `task0_test_cache_v1.py`, `task0_test_cache_shard.sbatch`,
  `task0_test_cache_consolidate.sbatch`, `submit_task0_test_cache.sh`
- `task0_vit_test_compare.py`, `task0_final_test_eval.sbatch`,
  `submit_task0_final_test.sh`

**Task 1 (4 files)**
- `task1_closed_loop.py`, `task1_schedule.py`,
  `task1_cycle.sbatch`, `submit_task1_closed_loop.sh`

The counts above are **10 Task 0 files plus 4 Task 1 files** (14 total),
not 9+4. The additional six fusion-specific diagnostics remaining at
`scripts/adaptive/` are:
`audit_fusion_image.py`, `check_fusion_image_model.py`,
`fusion_gpu_compute_probe.sbatch`, `fusion_stage_audit_full.sbatch`,
`profile_fusion_gpu_compute.py`, `run_fusion_image_smoke.py`.

Separately, the three `rep5bc*` scripts concern earlier-model
representation checks and are reserved for a representation-family audit.

### Critical operational hazard

Several already authored Slurm workers invoke the **live checkout**, not a
self-contained frozen source copy. Confirmed examples:

- `task0_full_history.sbatch` changes directory into the project's Delta
  checkout and executes `scripts/adaptive/task0_full_history.py`.
- `task0_test_cache_shard.sbatch` and its consolidation worker similarly
  invoke `scripts/adaptive/task0_test_cache_v1.py`.
- `task0_final_test_eval.sbatch` executes
  `scripts/adaptive/task0_vit_test_compare.py`.
- `task1_cycle.sbatch` executes `scripts/adaptive/task1_closed_loop.py`.
  Its scheduler submits the path
  `scripts/adaptive/task1_cycle.sbatch`.
- `task1_closed_loop.py` pins its own path in a source-file SHA256 list.

**Consequently, a path-changing Git pull could break queued/pending Slurm
jobs even if the job was already submitted.** GitHub CI cannot establish
whether the actual Delta queue is drained.

## Required gate before moving Task 0/Task 1

1. On Delta, inspect `squeue -u "$USER"`, and `sacct` for the current
   Task 0/Task 1 array and job IDs. Identify any pending or held children.
2. Inspect source hashes/manifest roots and decide whether those scripts
   belong to completed immutable research snapshots or unfinished jobs.
3. Confirm that no queued/pending worker will need the old live-checkout
   paths. If uncertain, do **not** move the scripts.
4. Group Task 0 and Task 1 independently, with versioned new frozen plans,
   updated internal script and worker paths, scientific identity hashes,
   corresponding docs and direct synthetic tests.
5. Add compatibility entry points **where necessary**, then validate
   without submitting GPU jobs. The old source snapshots remain the authority
   for their original completed trajectories.

This safety gate is not evidence that there are currently any such jobs;
their queue state has **not been checked** in this GitHub-only audit.

## Later publication decisions

- Make the **portable core** and simple synthetic/notebook examples the
  documented default, not a Task 0/Task 1 Slurm launcher.
- Separate all remaining Delta-specific paths, cache/source locations,
  model assets and secrets from public example TOML/configuration.
- Keep scientific variants/reproduction evidence, but archive operationally
  superseded pilots and one-time repairs with explicit provenance.
- Run both clean-install portable tests and an explicit review of the
  **entire Git history** before making the repo public.
