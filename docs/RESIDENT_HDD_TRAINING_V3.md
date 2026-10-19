# Pre–Task 0 v3: resident-RAM training from verified consolidated HDD cache

## Diagnosis
The prior HDD mmap-only GPU probe showed first eight shuffled train-batch fetches of 7–46 seconds, although GPU forward/backward was ~0.04 seconds after warmup. The CPU sequential scan masked the random Lustre page-fault problem. Avoid Lustre mmap during training: **sequentially read the 2.07 GB cache once into host RAM per job**.

## Design
- Immutable source cache remains `/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_hdd_v1`.
- Separate v3 code and immutable plan; all old v2 files, frozen output, and failed job logs preserved.
- Load float32 images and metadata with `np.load(..., mmap_mode=None)` ONCE, share a single read-only NumPy array between training and validation datasets.
- Existing `FastHistoricalDataset` continues manifest/stage indexing and produces exactly the same tensors; labels still come only from the train/validation manifest.
- 4,729 historical training objects / 29,161 rows; 2,365 validation objects / 14,148 rows; no test access.
- No new NVMe writes, no new cache consolidation, no persistent multi-cycle job, no source-data edits.
- Writes detailed `resident_load_seconds`, `resident_cache_bytes`, and ordinary epoch timings. Per-epoch history and checkpoints remain identical to the previous recipe design.
- 32 GiB host RAM requested for independent A40 fits; measured resident arrays total 2,072,249,288 bytes (~1.93 GiB).

## Install
```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_resident_loader_v3.zip -d .
python -m pytest tests/astronomy/fusion_image/test_resident_cache_v3.py -q
python -m pytest tests/astronomy/fusion_image -q
bash scripts/adaptive/submit_fusion_pre_task0_resident.sh --check
bash scripts/adaptive/submit_fusion_pre_task0_resident.sh --prepare
```

## 10-minute diagnostic before full training
```bash
bash scripts/adaptive/submit_fusion_pre_task0_resident.sh --probe
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_resident_v3
cat "$ROOT/probe_job_id.txt"
# after job completion:
cat "$ROOT/resident_probe/probe.json"
tail -70 "$ROOT"/logs/probe_*.out
```

Success means (a) roughly 2 GB were loaded and shared once, (b) random batch fetch times fell from tens of seconds to small fractions of a second, (c) GPU forward/backward remains fast. `--submit-case` blocks unless probe completed with credible fast access.

## Run independent training jobs (after checking probe)
```bash
bash scripts/adaptive/submit_fusion_pre_task0_resident.sh --submit-case constant
# after constant finishes, inspect setup, epoch curve and training summary:
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_resident_v3
cat "$ROOT/constant/setup_timing.json"
cat "$ROOT/constant/training_summary.json"
# then, if constant is good:
bash scripts/adaptive/submit_fusion_pre_task0_resident.sh --submit-case cosine
bash scripts/adaptive/submit_fusion_pre_task0_resident.sh --submit-case cosine_aug
```
One A40 with maximum 60-minute time request for full fit; use actual run timings to reduce later requests. `--prepare` and `--check` never submit GPU jobs. The original v2 failed attempt and output remain intact.

## Limitations / follow-ups
- If the buffered sequential read itself is slow/fails, capture console logs; filesystem variability still matters.
- RAM per independent job is ~2 GB, safe within requested 32 GB. **RAM residency holds only for that job**; no claim of cross-job caching.
- For live new unlabeled objects, extend the immutable cache with new stages or a separate delta-cache, rather than rebuilding historical arrays.
