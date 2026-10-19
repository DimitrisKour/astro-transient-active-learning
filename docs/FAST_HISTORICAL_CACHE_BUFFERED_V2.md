# Fast historical cache — buffered build fix v2

## Why

The first NVMe consolidation (job 22985825) terminated with SIGBUS (exit 135) while building a roughly 2.06 GB writable memory-mapped `images.npy`. On the Lustre-backed `/work/nvme` filesystem this is consistent with a backing-storage/mapped-page failure. The specific read or write that caused SIGBUS is unknown. Filesystem free space and reported user quota don't support a straightforward global/full-user-quota explanation; a project/group quota is not independently ruled out.

This patch changes only the fast-cache **build/validate** functions in `src/al_pipeline/astronomy/fusion_image/fast_cache.py`. It keeps the existing `FastHistoricalDataset` read-only memory-mapped lookup API for separate training jobs. It does **not** alter the original validated sharded cache, the pre-Task 0 trainer, experiment manifests, or checkpoints.

## Technical changes

- No memory-mapped source arrays or writable destination arrays during cache consolidation. Each source shard is read as ordinary numpy arrays, and image/metadata tensors are written sequentially in buffered `.npy` streams, with valid NumPy headers.
- Per-shard progress logging **before** and after each source shard, to identify the last successful shard if a storage read/write fails.
- Flushes output buffers after each shard and calls `fsync` before publication.
- Rejects incorrect source identity, duplicate/missing stage positions, missing/invalid arrays, or a modified source index.
- Validates output against 3 original samples per shard before publishing the new cache directory. The stand-alone `check` phase repeats the validation.
- On failure, preserves a hidden `.building_*` directory for investigation; never overwrites/publishes an incomplete final cache.

Normal buffered writing avoids the writable-mmap failure mode, but no user-level code can guarantee that a malfunctioning remote filesystem won't fail again. GPU training is not submitted by this package.

## Installation on Delta

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_fastcache_buffered_v2.zip -d .
python -m pytest \
  tests/astronomy/fusion_image/test_fast_cache_v1.py \
  tests/astronomy/fusion_image/test_fast_cache_buffered_v2.py -q
```

The code uses the original `scripts/adaptive/fusion_fastcache.py` CLI from `fusion_fastcache_v1.zip`, which must already be installed (it is in the user's current Delta setup).

## Submit one CPU-only job

```bash
REPORT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_fastcache_cpu_v2
mkdir -p "$REPORT/logs"
sbatch --output="$REPORT/logs/fastcache_%j.out" \
       --error="$REPORT/logs/fastcache_%j.err" \
       scripts/adaptive/fusion_fastcache_cpu_safe.sbatch
```

This does one buffered build of the previously verified full-history cache, checks exact array shapes and sampled numerical equality, and benchmarks lazy mapping and a full train+validation pass; **no GPU**. Requests 2 CPUs, 12 GB RAM, 30-minute wall limit. The published path remains `/work/nvme/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_v1`, which does not yet exist. If this final directory exists, the build refuses to overwrite it.

### Check results

```bash
REPORT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_fastcache_cpu_v2
cat "$REPORT"/logs/fastcache_*.out
cat "$REPORT"/logs/fastcache_*.err

CACHE=/work/nvme/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_v1
cat "$CACHE/fast_cache_manifest.json"
```

The successful log should contain 24 completed-shard progress lines, `verified: true`, and `io_cases` with `dataset_init_seconds`, `sample_seconds`, and `full_scan_seconds` for train/validation. The returned fast-cache manifest must indicate 7,094 objects and 43,309 stages and no labels.

The **previous** failed hidden `.building_tzaefms4` directory should remain untouched until this CPU job succeeds. Then reclaim it manually using the exact path; do not wildcard-delete active build directories. The size shown by `ls` is the logical size, not guaranteed actually allocated blocks; `du -sh` shows allocated size.

## After validation

Do **not** automatically resume the old three GPU jobs: they are chained to a failed prerequisite, and the pre-Task 0 trainer still uses `ShardedHistoricalNPZDataset(preload=True)`. Before new GPU submissions, integrate `FastHistoricalDataset` into a fresh training-run identity, profile model construction + data mapping + checkpoint reporting separately, and run one GPU smoke/pilot fit. Compare training metrics and raw input tensors to the old loader.
