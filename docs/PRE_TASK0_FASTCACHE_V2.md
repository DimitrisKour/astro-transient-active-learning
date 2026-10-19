# Fast consolidated-cache pre-Task 0 training (v2)

Does not modify the previously prepared `new_image_pre_task0_recipe_v1` plan or failed jobs.
Uses immutable `/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_hdd_v1`.

## Changes
- Only new files; keeps v1 runner/model and historical cache immutable.
- Uses `FastHistoricalDataset` read-only memory mapping, no `preload=True`.
- Full dataset: 4,729 train objects / 29,161 rows; 2,365 validation objects / 14,148 rows.
- Cache contains NO labels; labels come from the original fixed manifests.
- Checks fast-cache manifest against verified sharded-cache manifest, stage index checksum, preprocessing, array schema, and immutable source plan.
- Retries only transient source-index EIO up to 3 times, WITHOUT skipping integrity validation.
- Saves `setup_timing.json` immediately before first epoch: model init, manifest load, dataset init and other preparation.
- Retains checkpoint formats and both loss-best and F1-best checkpoints.
- New frozen plan identity: `/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_fastcache_v2`.
- One A40 per independent fit; 60min maximum; submit only constant first.

## Install
```
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_pre_task0_fastcache_v2.zip -d .
python -m pytest tests/astronomy/fusion_image/test_pre_task0_fastcache_v2.py -q
bash scripts/adaptive/submit_fusion_pre_task0_fast.sh --check
bash scripts/adaptive/submit_fusion_pre_task0_fast.sh --prepare
bash scripts/adaptive/submit_fusion_pre_task0_fast.sh --submit-case constant
```

After the first fit: `cat /projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_fastcache_v2/constant/setup_timing.json`; check training summary & logs. Then submit `cosine`, `cosine_aug` with `--submit-case` separately.

**Do not run old `submit_fusion_pre_task0.sh --submit`**: that targets the sharded cache and old frozen plan. The previously failed constant job left its old plan untouched.
