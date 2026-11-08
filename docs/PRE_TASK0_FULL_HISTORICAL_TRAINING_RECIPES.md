> **Archive notice:** This document reproduces an earlier fusion-image prototype or diagnostic. The scripts below are now under `scripts/archive/fusion_image/`, rather than the supported full-history resident-cache/Task 0/Task 1 workflow. Consult the archive README before any historical submission. Existing frozen snapshots must not be overwritten.

# Pre–Task 0: Controlled full-historical training-recipe comparison

**Scope:** Standalone new image+metadata AstroMiNN, old five classes
`AGN, TDE, SN I, SN II, CV`; BTSbot ConvNeXt backbone initialized from the
hashed GalaxyZoo pretrained snapshot. Historical **full training split**:
4,729 objects / 29,161 alert sample rows. Fixed validation: 2,365 objects /
14,148 rows. There is **no test evaluation, no candidate acquisition, no AL
L0 subset**, and no access to a live oracle.

The existing Experiment B AL L0 (`~2,890` alert rows) is **not** used. The
full-historical cache is a separate label-free prepared tensor store. It may
later be used by an AL simulation which reveals only acquired labels.

## Overview

| Case | LR schedule | Training image augmentation | EMA | Class weights | Queued by default? |
|---|---|---|---|---|---|
| `constant` | constant 1e-4 | no | no | n_sample^-0.85 | **yes** |
| `cosine` | 3-epoch warmup + cosine | no | no | n_sample^-0.85 | **yes** |
| `cosine_aug` | same | random D4 rotations/reflections | no | n_sample^-0.85 | **yes** |
| `cosine_aug_ema` | same | yes | 0.999 parameter EMA | n_sample^-0.85 | optional |
| `cosine_aug_effnum` | same | yes | no | beta=0.999 effective object counts | optional |

All cases have the same pretrained BTSbot initialization, seed 135, identical
model architecture and input labels, full 80-epoch upper bound, patience 15,
batch size 64, AdamW, lr 1e-4, weight decay .01, bf16 CUDA, gradient clipping
norm 1, and loss-best checkpoint selection on **fixed validation only**. No
weighted sampling, focal loss, evidential objective, hidden labels or test
access. No pretrained-vs-randomized model comparison yet: that is Task 0.

The schedule ramps epoch 1 at 0.1x LR, epoch 2 at 0.55x, epoch 3 at 1x, then
cosine-decays to 0.05x by epoch 80. The scheduler is epoch-based. Stopping
before epoch 80 means cosine may not reach the floor, an interpretation
caveat for the comparison. The constant control uses the SAME 80/15 stopping
limits, not the earlier 30/6 pilot limits.

Image augmentation applies one random D4 transform **per input image triplet**
(science/reference/difference channels transformed together) after reading
deterministic cached inputs. Validation always uses unaugmented inputs.
EMA averages trainable parameters at every optimizer step; normalization
buffers are synchronized from the ordinary model, not averaged. The EMA
comparison selects and evaluates the EMA model on validation loss. This is a
separate hypothesis, not a validated optimal choice.

The effective-number variant computes counts over unique labeled **objects**,
but retains the existing per-alert-row CE reduction; it is not equivalent to
an object-balanced sampler or equal-object loss. Avoid activating multiple
balancing mechanisms simultaneously.

For every case, the trainer writes `best_vit_tower.pth` (validation-loss
selected) and `best_object_f1.pth` (separate validation macro-F1 selected),
`training_history.json`, `training_summary.json`, and
`validation_per_class_metrics.json`. Both checkpoint files carry a complete
standalone model and the existing model spec/class-order identity. The loss
checkpoint is primary, with independent F1 checkpoint for diagnostic review.

## Install on Delta

Copy the ZIP to `/projects/bcrv/dkourtesis/` from Windows PowerShell, then:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_pre_task0_training_v1.zip -d .
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/astronomy/fusion_image/test_pre_task0_recipe.py -q
python -m pytest tests/astronomy/fusion_image -q
```

Installation creates only **new** files; it does not overwrite `backends.py`,
`model.py`, or the historical cache, and doesn't queue a job.

## AFTER the full historical cache has completed and been inspected

First verify its `--status` output says `finalized: true` and that all shards
succeeded. Then:

```bash
bash scripts/archive/fusion_image/recipe_v1/submit_fusion_pre_task0.sh --prepare
```

This checks all identities: cache `source=historical_full_train_plus_fixed_validation`,
exact historical manifests, full population counts, stage-shard completion,
label-free cache flag, previous small AL L0 non-identity, disjoint train/validation
objects, hashed HF pretrained weights, original source plan, and current
preprocessing. It freezes `plan.json` in:

`/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_v1`

No jobs are submitted by `--prepare`.

If successful, queue the **three primary A40 fits**, serialized through
`afterok` (so a failed fit doesn't automatically spend more GPU time):

```bash
bash scripts/archive/fusion_image/recipe_v1/submit_fusion_pre_task0.sh --submit
```

There are only three GPU jobs; **at most one of these three runs at once**.
Each requests 1 A40, 4 CPUs, 32 GB host RAM, and **90 minutes as a safety
walltime cap**; GPU allocation ends when a job completes, not at the time cap.
Full 29,161-sample training may run considerably longer than the 2,890-row
initial-AL-set benchmark, particularly with 80 epochs and 15 patience.
To request a different cap, edit `#SBATCH --time=01:30:00` in
`scripts/archive/fusion_image/recipe_v1/fusion_pre_task0.sbatch` before submission.

Status and logs:

```bash
bash scripts/archive/fusion_image/recipe_v1/submit_fusion_pre_task0.sh --status
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_v1
cat "$ROOT/submitted_jobs.tsv"
tail -40 "$ROOT"/logs/cosine_aug_*.out
```

After they finish:

```bash
bash scripts/archive/fusion_image/recipe_v1/submit_fusion_pre_task0.sh --report
cat "$ROOT/comparison.csv"
cat "$ROOT/cosine_aug/training_summary.json"
cat "$ROOT/cosine_aug/validation_per_class_metrics.json"
```

**Do not launch optional jobs automatically.** After discussing the first
three, and only when they are complete, submit optional EMA and effective-number
fits via:

```bash
bash scripts/archive/fusion_image/recipe_v1/submit_fusion_pre_task0.sh --submit-optional
```

These two optional fits are also serialized with respect to one another.
Their performance should be compared against `cosine_aug` at the same seed,
with careful attention to per-class recall, macro-F1, overfitting, stability,
and loss–F1 checkpoint disagreement. Three initial runs are not sufficient
to declare a recipe generally optimal; the chosen recipe must later be
frozen and used for the paired Task 0 pretrained/randomized × 3 seeds.

## Safety and known limitations

- A completed finalized cache is mandatory *before* plan creation or GPU job
  submission; the old 2,890-row AL L0 cache cannot pass these checks.
- `--prepare` and jobs refuse changed source code/hashes, manifest, model,
  preprocessing or cache identity. The run directory cannot be overwritten.
- If a job fails midway, preserve its partial folder and log; do not simply
  resubmit. Investigate, then move the failed folder aside and explicitly
  replan/retry with a new identity as appropriate.
- The optional cases are intentionally not part of the default submission.
- The module is independent of the AL backend to avoid unreviewed changes to
  existing AL experiments. We can migrate the winning recipe into a generic
  AL trainer after Task 0. This overlay **does not** implement AL cycles.
- No test evaluation. Keep this strict through the recipe-selection stage.
