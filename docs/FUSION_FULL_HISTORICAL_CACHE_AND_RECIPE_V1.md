> **Historical/archived workflow:** The initial L0-only cache/benchmark scripts have been moved to `scripts/archive/fusion_image/early_l0/`. They are retained for reproduction, not recommended as the current full-historical Task 0/Task 1 workflow. See `scripts/archive/fusion_image/early_l0/README.md` before running any old submission commands.

# Full historical ZTF cache: data gate before Task 0

**2026-10-18. No GPU training is launched by this package.**

## Naming convention

- `Historical-full-train` / `L0_full`: the *complete* historical train manifest from the fixed `astrominn_ztf` registry: expected 4,729 objects / 29,161 sample rows. This is Task 0's training population.
- `AL-L0`: the original Experiment B **small** initial labeled subset: previously measured 2,890 sample rows. Not Task 0.
- `AL-U0`: all other historical training objects, whose labels must be hidden from an acquisition algorithm until oracle resolution.
- `validation`: the unchanged historical fixed validation manifest, previously measured 14,148 rows.
- `test`: NEVER used for training, cache compilation, checkpoint selection or policy development.

The earlier cache `new_image_l0_cache_v3` covers only AL-L0 + validation and remains **unaltered**. The new full-history cache contains all historical train objects (including the future AL-U0) and validation. Therefore one full cache serves Task 0, AL-L0, and all historical AL-U0 acquisition cycles. It does not incorporate unrelated genuinely new/unlabeled observations outside the original indexed historical pool.

The cache saves **no class labels**, and the runtime can instantiate `ShardedHistoricalNPZDataset(label_column=None)` for unlabeled inference. The offline preflight *does* read historical train/validation labels to validate provenance, class consistency, and that the old small AL-L0 is a subset; those labels must never be passed to AL scorers. Training and oracle resolution must continue to use the correctly gated manifest/label interface.

## Contents and compatibility

The overlay introduces:
- `scripts/adaptive/fusion_full_historical_cache_v1.py`: strict source/audit guard and full train+validation runner.
- `scripts/adaptive/submit_fusion_full_historical_cache_v1.sh`: prepare, submit, status, retry; NO GPU jobs.
- Two SLURM job scripts: 24 CPU shards (max 6 simultaneous) then one dependent CPU finalization.
- `tests/astronomy/fusion_image/test_full_historical_cache_v1.py`: test expected behavior with synthetic compressed object NPZs.
- A *backward-compatible two-line* update to `src/al_pipeline/astronomy/fusion_image/cache_v2.py`: `finalize(root, *, source='L0+validation only')`. This lets the new manifest truthfully record `source='historical_full_train_plus_fixed_validation'`. Previous v3 runs retain the original behavior.

This is designed for the **already installed cache-v3 modules** (`cache_helpers.py`, `cache_v2.py`). Do not use if those are absent; the existing v3 benchmark already demonstrates they are installed in the working checkout.

## Transfer / install on Delta

From Windows PowerShell:

```powershell
scp "$HOME\Downloads\fusion_full_historical_cache_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

Then on Delta:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_full_historical_cache_v1.zip -d .
python -m pytest tests/astronomy/fusion_image/test_full_historical_cache_v1.py \
  tests/astronomy/fusion_image/test_sharded_cache_v2.py -q
python -m pytest tests/astronomy/fusion_image -q
bash scripts/adaptive/submit_fusion_full_historical_cache_v1.sh --prepare
```

**Check the printed scope carefully**: `training_objects` should be 4,729, `training_rows` should be 29,161, and `previous_small_l0_rows` should be 2,890. Confirm `validation_rows` is 14,148. Stop and inspect if the values differ from the historical audit. Note this depends on the older pinned audit and original CSVs being present/unchanged.

To launch CPU preprocessing only:

```bash
bash scripts/adaptive/submit_fusion_full_historical_cache_v1.sh --submit
```

Monitor:

```bash
bash scripts/adaptive/submit_fusion_full_historical_cache_v1.sh --status
squeue -u "$USER"
PIPE=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_cache_pipeline_v1
CACHE=/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_train_cache_v1
cat "$PIPE/submitted_jobs.tsv"
find "$CACHE" -maxdepth 2 -name done.json | wc -l
tail -n 35 "$PIPE"/logs/*.err
```

The final file is:

```text
/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_train_cache_v1/cache_manifest.json
```

Retry only after earlier shard/finalization jobs have stopped, and inspect any `.shard_NNN_*` incomplete temporary directories first:

```bash
bash scripts/adaptive/submit_fusion_full_historical_cache_v1.sh --retry
```

No existing cache or model checkpoint is overwritten. No full-data training, test evaluation or active learning is submitted. In particular **do not run** `submit_fusion_l0_matrix.sh --submit` until a new full-history-aware training matrix is separately verified: the old one is bound to a different cache and dataset plan.

## Proposed training recipe review; not activated by this cache package

The published `frenbox/fusion_model_basics` run3 is *TEMPO + AstroMiNN fusion with a different hierarchy and different labels*, not a standalone, five-class image+metadata baseline. Use its engineering lessons, not its raw metrics or entire loss definition.

| Feature | Upstream run3 | Current historical standalone trainer | Task 0 approach |
|---|---|---|---|
| Backbone | BTSbot ConvNeXt | Same, pretrained vs random | Keep both conditions |
| Batch | 64 | 64 | Keep |
| Optimizer | AdamW | AdamW | Keep |
| Weight decay | 0.01 | 0.01 | Keep |
| AMP | bf16 | bf16 on CUDA | Keep |
| LR | Separate encoder/fusion LRs | One 1e-4 LR | Benchmark a discriminating, fairly matched LR design, then freeze it |
| Schedule | Warmup 3 epochs + cosine | Constant LR | Implement as opt-in then make primary candidate |
| Image augmentation | Flips/90-degree rotations via upstream preprocessing | None | Candidate on-the-fly train-only on cached images; fixed validation unaffected |
| EMA | 0.999 decay | None | Add optional raw/EMA dual validation later, with BN buffer policy |
| Epochs/patience | 80 / 15 | 30 / 6 pilot | Use 80/15 as a ceiling, inspect curves/early-stop |
| Weighting | Effective-number loss balancing + sampler | Inverse-frequency exponent 0.85 by sample rows | Test controlled object-level effective-number alternative, **not** simultaneous new sampler |
| Loss | Hierarchical evidential (for another taxonomy) | Weighted softmax CE | Retain weighted CE reference; defer hierarchical objective |

### Specific reasons not to copy everything

1. The pretrained image backbone may merit a lower LR than the freshly initialized coordinate/fusion/head components. Matched arms matter when attributing gains to pretraining. Do not tune only the pretrained condition and claim a purely pretraining-derived gain.
2. EMA averages model parameters over optimizer steps, with exponentially decaying weights; it is not SWA and not a seed ensemble. An upstream 0.999 coefficient may be too inert for short small-AL-L0 trajectories; refresh/reconcile BatchNorm stats before comparing checkpoint predictions.
3. On cached tensors, train-only 90-degree rotations and flips can be applied after deterministic preprocess. Keep the three image channels together, avoid validation transforms and record RNG/state. Confirm physical symmetry and metadata consistency.
4. Our fixed five-label set `AGN, CV, SN I, SN II, TDE` does not share upstream scheme-A hierarchy. Using hierarchical EDL would require a justified taxonomy and separate objective evaluation, not a mechanical port.
5. Oversampling plus inverse/"effective number" class weighting can double-correct minority classes. First investigate **object**-level class counts versus alert/sample rows, and compare effective-number weights *instead of* the previous exponent-weighted CE; add sampling only if still supported by evidence.
6. No hyperparameter comparison may select candidates using test labels. Validation is the sole selection split; test is report-only after recipes are frozen.

### Suggested study after cache finalization

1. Implement optional warmup+cosine and train-only augmentation in the standalone trainer, with explicit provenance and tests. Keep unchanged CE reference.
2. Run a **single** full-history seed-135 learning-curve reference. Use pilot ablations to decide whether augmentation and/or LR schedule improve validation stability and macro-F1; avoid changing several variables at once.
3. Optionally test EMA and effective-number weighting on the same fixed data, as separate variations, paying special attention to TDE/SN II recall and overcorrection.
4. Freeze a defensible recipe and run six full-data Task 0 fits (`pretrained/randomized x seeds 135/246/357`).
5. Only after Task 0, split this same full historical pool into AL-L0/AL-U0 and run closed-loop AL warm-start vs L0-restarter, using the complete cache (which is label-free). Genuinely new observations require a later incremental cache extension.

No experiment submission is included in this overlay on purpose.
