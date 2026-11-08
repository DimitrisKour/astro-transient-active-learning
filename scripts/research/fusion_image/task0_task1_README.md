# Fusion Task 0 / Task 1 — canonical grouped research entry points

The canonical source now lives in `scripts/research/fusion_image/task0/`
and `task1/`. These studies reproduce the standalone 5-class BTSbot
ConvNeXt plus 55-feature metadata experiment, not the original frozen-expert
ViT experiments and not the upstream TEMPO fusion model.

## Task 0 — full historical initialization comparison

Six fits of one architecture across pretrained/randomized image initialization
and seeds 135/246/357, with validation-only primary checkpoint selection.
The *separate* test-cache and final-test scripts are tightly gated: the test
cache contains **no labels**, test evaluation needs all six completed fits and
a verified train/validation/test overlap audit.

Controller and launcher:

```text
task0/task0_full_history.py
task0/submit_task0_full_history.sh
task0/task0_test_cache_v1.py
task0/submit_task0_test_cache.sh
task0/task0_vit_test_compare.py
task0/submit_task0_final_test.sh
```

Task 0 retains the same historical *input cache*, fixed upstream revision,
model architecture and original resident `cosine_aug` recipe baseline.

## Task 1 — offline historical object-level acquisition

Nine cycles and four policies (random, entropy, entropy-diversity, balanced
exploration), with warm-start and L0-restart comparisons. Training and
acquisition see no test targets and do not use test metrics for model
selection. New Task 1 uses the **new grouped Task 0 plan** and the **new
grouped test cache**, never the original Task 0 source hashes.

```text
task1/task1_closed_loop.py
task1/task1_schedule.py
task1/task1_cycle.sbatch
task1/submit_task1_closed_loop.sh
```

## Why the old scripts still exist

The original 14 files under `scripts/adaptive/task0_*`,
`scripts/adaptive/task1_*` and `submit_task0_*/submit_task1_*` remain
**byte-for-byte unchanged**. This is intentional and tested: existing
scheduled Slurm workers may load them from the live checkout, and original
Task 1 `plan.json` contains the old controller's file hash.

Those original paths are **legacy operational compatibility**, not the
recommended new research source. They may only be archived/removed after
checking the actual Delta queue and the frozen-plan registry. We have not
verified the live Delta queue from GitHub.

## New experiments are isolated, not an in-place resume

The new controllers use distinct output identities to prevent accidental
reuse of existing historical suites:

| Artifact | New grouped directory basename |
| --- | --- |
| Task 0 outputs | `new_image_task0_full_historical_grouped_v1` |
| Task 0 independent test cache | `new_image_historical_test_fastcache_grouped_v1` |
| Task 0 test-cache submission registry | `new_image_test_cache_pipeline_grouped_v1` |
| Task 1 outputs | `new_image_task1_closed_loop_grouped_v1` |

These are *new* research identities; their results have **not** been generated.
The new Task 0 still consumes the previously verified, read-only full-history
training cache and resident recipe baseline; the new test cache must be built
and validated independently. Do **not** reuse the original Task 1 plan or old
Slurm submission registry.

The launchers and workers resolve the repository using `AL_PIPELINE_REPO`
(where set), the script's own repository-relative path for launchers, or
`SLURM_SUBMIT_DIR` for workers. The scripts still require the exact historical
data/model dependencies and Delta environment. This is **not** a portable
training quickstart and no experiment is automatically submitted.

## Validation

```bash
python -m pip install -e ".[dev]"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q \
  tests/jobs/test_fusion_task01_safe_migration.py \
  tests/astronomy/fusion_image/test_task0_full_history_v1.py \
  tests/astronomy/fusion_image/test_task1_closed_loop_v1.py
```

CI checks both old scientific contracts and new path/hash/isolation
invariants. The tests do not use the private historical dataset, a GPU,
network broker calls, or Slurm. Original historical source files remain in
place until the operator confirms it is safe to retire them.
