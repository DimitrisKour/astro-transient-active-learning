# Fast historical cache diagnostic — no GPU jobs or existing experiment modifications

## Why

The original shard cache stores exactly the correct 4,729 training objects and 2,365 validation objects. But `ShardedHistoricalNPZDataset(preload=True)` copies the 24 shards into newly allocated arrays on every training job. That approach is expensive on a shared HDD and could dominate startup in multi-cycle AL.

This package **does not replace** the original immutable cache, the trained model, or any old scripts. It consolidates the **already validated, label-free** full-history cache into two contiguous `.npy` arrays on Delta's `/work/nvme`, plus an object/alert lookup index. All original training and validation objects/stages are copied exactly once per cache materialization. There is no new NPZ preprocessing.

`FastHistoricalDataset` opens these `.npy` arrays with `np.load(mmap_mode='r')`, avoiding a full per-job RAM copy. It supports the full training manifest or arbitrary AL subsets of training rows; passing `label_column=None` omits ground-truth labels. Caching objects does not mean leaking historical labels into selection.

**This package measures, but does not yet claim, a specific speedup.** Separate SLURM jobs remain supported. Confirm this cache's full-split DataLoader scan is fast before switching pre–Task 0 training to it. Model import/init/final-evaluation overhead also needs profiling.

## Installation

From the Delta active-learning checkout:

```bash
unzip -o /projects/bcrv/dkourtesis/fusion_fastcache_v1.zip -d .
python -m pytest tests/astronomy/fusion_image/test_fast_cache_v1.py -q
```

Run first-failure diagnosis (no changes):

```bash
bash scripts/adaptive/diagnose_fusion_pre_task0.sh
```

## CPU-only consolidation and benchmark

```bash
REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
REPORT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_fastcache_cpu_v1
mkdir -p "$REPORT/logs"
cd "$REPO"
sbatch --output="$REPORT/logs/fastcache_%j.out" \
       --error="$REPORT/logs/fastcache_%j.err" \
       scripts/adaptive/fusion_fastcache_cpu.sbatch
```

This runs one CPU job (2 CPUs, 12 GiB, 30 minute limit). It:
1. Verifies the immutable source-cache identity and full-history counts.
2. Copies all 43,309 stages sequentially into a consolidated, **label-free** cache under `/work/nvme/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_v1`.
3. Checks all source-to-target stage indices and compares 3 stage tensors per source shard with exact numerical equality.
4. Benchmarks lazy initialization, 1,024 random row accesses, and complete sequential DataLoader scans for both train and validation.

**Do not run this expensive data movement on the Delta login node.** On a failed build, partial files are retained in a hidden `.building_*` directory for inspection; the finalized target is never created until the last step succeeds. Before submission check NVMe available/quota; images take ~2.07 GB plus metadata and filesystem overhead.

When completed:

```bash
REPORT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_fastcache_cpu_v1
cat "$REPORT"/logs/fastcache_*.out
cat "$REPORT"/logs/fastcache_*.err
```

**Do not submit new GPU fits yet**. Inspect the original failed job's traceback and the real CPU timings first. Based on those, we can release a v2 training runner that profiles separately model setup, dataset construction, epoch training, and final evaluation while preserving the previous run artifacts.
