> **Historical/archived workflow:** The initial L0-only cache/benchmark scripts have been moved to `scripts/archive/fusion_image/early_l0/`. They are retained for reproduction, not recommended as the current full-historical Task 0/Task 1 workflow. See `scripts/archive/fusion_image/early_l0/README.md` before running any old submission commands.

# Full cached L0 pretrained seed-135 baseline (validation only)

This overlay DOES NOT modify existing training implementations. It reuses the
L0+validation sharded cache v3 and the existing frozen `new_image_l0_matrix_v1`
training policy, but trains **one complete model** before the six-fit matrix.

* Pretrained GalaxyZoo BTSbot ConvNeXt-pico, seed 135, original five classes.
* Historical L0 2,890 rows; fixed historical validation 14,148 rows.
* Existing weighted sample-level cross-entropy objective and loss-selected
  checkpoint, 30 epochs maximum and patience 6, AdamW, batch 64, LR 1e-4.
* Original B.1, C checkpoint/acquisition features are NOT activated.
* Validation-only model evaluation, including sample/object metrics and
  per-class scores; no test read and no active-learning acquisition.
* Output root is NEW and never overwrites the one-epoch benchmark or matrix.

Prerequisites: completed `new_image_l0_cache_v3/cache_manifest.json` and
`new_image_l0_matrix_v1/plan.json` from previous steps.

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/fusion_cached_l0_full_v1.zip -d .
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/archive/fusion_image/early_l0/fusion_cached_l0_full_seed135.py check
```

If check succeeds, queue one A40 job:

```bash
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_cached_full_pretrained_seed135_v1
mkdir -p "$ROOT-logs"
sbatch --output="$ROOT-logs/job_%j.out" --error="$ROOT-logs/job_%j.err" \
  scripts/archive/fusion_image/early_l0/fusion_cached_l0_full_seed135.sbatch
```

After completion, examine:

```bash
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_cached_full_pretrained_seed135_v1
cat "$ROOT/cached_l0_baseline_report.json"
cat "$ROOT/validation_per_class_metrics.json"
cat "$ROOT/training_history.json"
tail -60 "$ROOT-logs"/job_*.out
```

If a job times out/interrupts after writing partial training files, DO NOT
resubmit blindly: the runner refuses partial outputs. Inspect logs, checkpoint,
history and summary first.

Scientific interpretation: this is a full *pilot baseline*, not yet the
controlled pretraining comparison across seeds. The existing cache includes
L0 and validation ONLY; do not point the old six-fit/evaluation runner at it
without handling the test split and smoke pilot correctly.
