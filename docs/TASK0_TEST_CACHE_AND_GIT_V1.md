# Task 0 final-test cache: independent, immutable, label-free (v1)

## Why this exists

The Task 0 full-historical train+validation resident cache is 2.07 GB and performs
well because its files are buffered into RAM once. The current final-test evaluator
uses slow per-alert NPZ reads, which would repeat the original bottleneck.
This patch adds **TEST-ONLY** NPZ preprocessing and an identical resident-memory
inference path, without changing training code or completed checkpoint hashes.

## What is frozen and what isn't

- **UNCHANGED:** `scripts/adaptive/task0_full_history.py`,
  `recipe_training_resident.py`, `cache_v2.py`, `fast_cache.py`,
  `resident_cache.py`, `backends.py`, `model.py` and every Task 0 checkpoint.
  These are training inputs with SHA-256 hashes recorded in the frozen Task 0 plan.
- **NEW:** `test_fastcache_v1.py`, cache driver, 24 CPU-only shard jobs, CPU
  consolidation job and tests.
- **CHANGED, test-only:** `task0_vit_test_compare.py`,
  `submit_task0_final_test.sh`, `task0_final_test_eval.sbatch`. Existing
  evaluation summary format remains compatible with the older ViT reporter.
- **TEST DATA:** cached tensors and indexes contain no labels and no predictions;
  label authority remains the audited test CSV. NO test inference happens during
  cache preparation. The six-fit gate and strict old-ViT split audit still apply.

Data source is the *frozen* historical registry and original per-object NPZ index.
Exactly the same BTSbot ConvNeXt preprocessing/config is checked against the
existing full-history training cache identity. No test data are copied to the
training cache or to the training directory. Cache storage is on HDD (not NVMe:
Delta's project NVMe soft quota was previously exceeded).

## Install (on Delta) after transferring ZIP via Windows PowerShell

```powershell
scp "$HOME\Downloads\fusion_task0_test_cache_resident_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_task0_test_cache_resident_v1.zip -d .
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/astronomy/fusion_image/test_test_fastcache_v1.py -q
```

## One-time **CPU-only**, label-free cache build; may run alongside Task 0 fits

```bash
bash scripts/adaptive/submit_task0_test_cache.sh --check
bash scripts/adaptive/submit_task0_test_cache.sh --prepare
bash scripts/adaptive/submit_task0_test_cache.sh --submit
bash scripts/adaptive/submit_task0_test_cache.sh --status
```

This schedules 24 restartable CPU shard tasks at most six concurrently and one
CPU consolidation/verification job dependent on successful completion of the
array. Each shard has a 30-minute **maximum allocation**; the consolidation
requests 20 minutes. No GPU is requested and no final-test metrics are computed.

Cache location:
`/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_historical_test_fastcache_v1`.
Shard workspace:
`/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_historical_test_fastcache_v1_shards`.

After successful consolidation:

```bash
bash scripts/adaptive/submit_task0_test_cache.sh --validate
```

`--validate` checks file checksums, complete manifest stage coverage, finite
preprocessed inputs, source parity, and random access performance without labels.
If a shard or consolidation fails, inspect CPU logs in
`/projects/bcrv/dkourtesis/applecider_adaptive/new_image_test_cache_pipeline_v1/logs`
then use `--retry` when previous jobs have fully stopped. Completed shards are
never overwritten. Failures retain diagnostic temporary files.

## Later: **strictly after** Task 0 fits, frozen selection and ViT test audit

This cache construction does **NOT** unlock final test scoring automatically.
This document supersedes the direct-NPZ performance warning in the original
Task 0 instructions. The old-ViT/new-model `audit` still requires the old ViT's actual full training,
validation and test manifests and verifies exact object/alert-stage/label multisets.
The old ViT cycle-9 experiment/checkpoint provenance is checked in `compare`.

Only after all six Task 0 fits and this audit:

```bash
bash scripts/adaptive/submit_task0_final_test.sh --submit-case pretrained_s135
```

The new `task0_vit_test_compare.py` uses `TestFastDataset` and
`attach_resident`, then the existing classification/aggregation metric
functions to reproduce the existing `evaluation_summary.json` schema, including
per-class metrics. The final evaluation launcher requests 15 minutes instead
of two hours; unlike the old direct-NPZ loader, the cache needs only one
sequential HDD read into memory per inference job.

Do not compare test metrics across models if their test objects leak into either
model's training/validation data. Do not select recipes/checkpoints using test
results. For repeated work with the same preprocessing, the cache is reusable.

## Git / repo hygiene

The untracked `core.python.*` file is a **core dump**, not Python source: do
not add or push it. Review its size before deciding to remove it (`ls -lh`).
Similarly, inspect the root `README.txt` before deciding if it is a genuine
repository document or a temporary extraction artifact.

Stage source/scripts/docs/tests in logically reviewable commits, and use
`git diff --cached --name-status`, `git diff --cached --check`, and the test suite
before pushing. Avoid `git add -A` while the core dump is untracked.

If the Task 0 plan contains SHA hashes of files, committing the same bytes does
not change their hashes; but reformatting or modifying those frozen sources
would invalidate already queued fits. Do not run automated formatters on them.

## Conservative Git commands (run only after inspecting status)

These paths are **only source files**; this excludes the root core dump, root
README.txt, raw NPZ/NPY files, cache data, ZIPs and checkpoints:

```bash
git status -sb
git ls-files -z --others --exclude-standard -- \
  'src/al_pipeline/astronomy/fusion_image/*.py' \
  'scripts/adaptive/*.py' 'scripts/adaptive/*.sh' 'scripts/adaptive/*.sbatch' \
  'tests/astronomy/fusion_image/*.py' 'docs/*.md' | xargs -0 -r git add --
git diff --cached --name-status
git diff --cached --check
```

Review those staged paths for obsolete diagnostic probes. If one should not
be committed, use `git restore --staged -- path/to/file`. The ignored root
`README.txt` is **not** automatically ignored; just not selected by this command.
After tests pass and changes are satisfactory:

```bash
git commit -m "Add reproducible historical fusion caches, Task 0, and fast test inference"
git push -u origin "$(git branch --show-current)"
```

The user controls the actual push from Delta; this ZIP cannot push files that
currently exist only on the cluster's working tree.
