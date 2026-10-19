# Full-history fast cache: HDD quota-safe target

The prior build to `/work/nvme/bcrv` failed with `EDQUOT` / quota exceeded. Allocation usage is 512.9 GB against a 500 GB soft quota, whereas `/work/hdd/bcrv` is at 2.296 TB versus 2.93 TB soft quota. Filesystem-level `df` is not an allocation quota check. This patch changes *only the scheduled destination* for the already installed buffered-v2 builder. It does not modify the original full-history source cache, training code, or older failed caches.

## Install

Extract at the root of the `active-learning` checkout:

```bash
unzip -o /projects/bcrv/dkourtesis/fusion_fastcache_hdd_retry_v1.zip -d .
bash -n scripts/adaptive/fusion_fastcache_cpu_hdd.sbatch
```

The separate `fusion_fastcache_buffered_v2.zip` and `fusion_fastcache_v1.zip` files must already have been installed; the new sbatch calls their existing `fusion_fastcache.py` CLI.

## Submit a single CPU verification job

```bash
REPORT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_fastcache_cpu_hdd_v1
mkdir -p "$REPORT/logs"
sbatch --output="$REPORT/logs/fastcache_%j.out" --error="$REPORT/logs/fastcache_%j.err" scripts/adaptive/fusion_fastcache_cpu_hdd.sbatch
```

This builds and validates 43,309 stage inputs for the 29,161-row training and 14,148-row validation splits; no training or test access. Files are published only after internal validation. The job also benchmarks independent dataset startup, 1024 random accesses, and one complete DataLoader pass of each split. It uses 2 CPU, 12 GB, max 30 minutes. The output target is `/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_hdd_v1`.

## Report

```bash
REPORT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_fastcache_cpu_hdd_v1
tail -120 "$REPORT"/logs/fastcache_*.out
tail -80 "$REPORT"/logs/fastcache_*.err
```

Compare `dataset_init_seconds`, `sample_seconds`, and `full_scan_seconds` with the earlier sharded-cache benchmark. Successful HDD consolidation **does not by itself mean fast independent GPU jobs**; next instrument model init, data access, and final eval before changing training. Continue keeping the original historical cache as source of truth.

## Optional cleanup after confirming all old build jobs stopped

To locate incomplete NVMe builds and assess their space usage:

```bash
find /work/nvme/bcrv/dkourtesis/applecider_adaptive -maxdepth 1 -type d -name '.new_image_full_historical_fastcache_v1.building_*' -print
```

Only after verifying each printed directory is from a failed attempt, remove those exact incomplete directories (not the entire NVMe workspace). This can restore allocation quota headroom; it is not needed to run the new HDD-targeted job.
