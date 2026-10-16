# New AstroMiNN image+metadata model: migration audit and validation gates

Source revisions examined on 2026-10-14:

- `frenbox/fusion_model_basics`, `main` @ `9d8b635a35c9dc4f90271098abdfa494abffc24f`.
- `DimitrisKour/al-pipeline`, `feat/entropy-batch-policies` @ `66142b9260055834a21c1e5d377209055e70308e`.

**Scope**: `AstroMiNNNPZClassifier` from `astrominn_encoder.py`, image + metadata ONLY.
No `TempoFusionModel`, TEMPO encoder, fusion head, or photometry stream is used.
Some upstream helper imports may still require its photometry dependencies.

## Important conclusions from the inspected source

1. **Different input schema**. Original historical ViT ingests 34 metadata
   features and `(3,63,63)` image cutouts. The new standalone model expects 55
   metadata features (its `OLD_METADATA_COLUMNS`) and `(3,63,63)` cutouts.
   Upstream `AlertStreamFusionDataset._build_metadata_tensor` constructs and
   normalizes the 55 features. We reuse that method, rather than invent 21
   missing features or double-normalize historical 34-feature arrays.
2. **New model has several variants**. `metadata_only` (out of scope),
   `patch_images` (random image patch embedding) and `btsbot_images` (BTSbot
   ConvNeXt-pico initialized from GalaxyZoo-pretrained HF weights). The latter
   uses `nabeelr/BTSbot-convnext-pico-galaxyzoo` and its own L2-per-channel
   normalization. The pretrained snapshot is NOT checked into GitHub and must
   be acquired into a readable user-owned directory and hashed before science.
3. **Different taxonomy**. New model default `CLASS_GROUPS` has SIX outputs,
   and its published `scheme_a_5c` fusion recipe has FIVE DIFFERENT broad classes:
   `AGN_like`, `CCSN`, `CV_Nova`, `SN_Ia`, `Stellar`. It excludes TDE and mixes
   Type Ib/Ic and IIn labels into CCSN. **Never compare its scheme-A metrics
   numerically to Experiment B's AGN/TDE/SN I/SN II/CV metrics.** This patch
   explicitly overrides AstroMiNN standalone's class order to original five.
4. **Different population and sampling**. The new repo describes 20,941 objects,
   14,415 train / 3,089 validation / 3,090 test, with train prefix sampling.
   Experiment B has 4,729 historical training objects and a fixed early-alert
   sample-based split. Different splits, object memberships and class labels
   must be audited; equality must not be assumed. The new repo states some
   underlying photometry is ~16 months newer.
5. **Pretraining**. The pretrained GalaxyZoo ConvNeXt is explicitly supported.
   The public repo does not include its binary weights or a downloadable
   standalone five-class AstroMiNN classification checkpoint. The documented
   `run3_bf16_midlr` result is **TEMPO+AstroMiNN fusion**, not a valid direct
   baseline for image+metadata standalone.
6. **Training**. Upstream fusion uses hierarchical evidential Dirichlet loss,
   bf16 (after fp16 numerical collapse), EMA and LR warmup. These cannot be
   imported wholesale into a plain softmax five-class AstroMiNN comparison.
   The adapter implements the previous Experiment B's weighted CE baseline
   with explicit alternatives, and resets the optimizer per cycle.
7. **Initialization**. The AL state manager already supports
   `restart_from_random`, `restart_from_baseline`, and `warm_start`.
   For scientifically meaningful initialization comparisons, hold the image
   pretraining status fixed; compare the SAME architecture under all three.
   The ConvNeXt can be pretrained or replaced with a freshly constructed randomly initialized timm backbone,
   and this choice is tracked separately from AL cycle initialization.

## Installation (Delta)

Clone the upstream repo **separately** from `al-pipeline`:

```bash
cd /projects/bcrv/dkourtesis
# Only if the folder does not already exist:
git clone https://github.com/frenbox/fusion_model_basics.git fusion_model_basics
cd fusion_model_basics
git checkout 9d8b635a35c9dc4f90271098abdfa494abffc24f
git rev-parse HEAD

cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_image_adapter_overlay_v1.zip -d .
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH=/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}
python -m pytest tests/astronomy/fusion_image -q
```

The upstream image encoder `btsbot_images` requires `timm`, `huggingface_hub`
if downloading, and a locally readable HF snapshot containing
`pytorch_model.bin` and `train_config.json`. `patch_images` needs no HF weights.
Check `python -c 'import torch, pandas, yaml, sklearn'` and install compatible
optional dependencies *without replacing the working environment's torch*.
Use a separate environment if necessary.

### Check 1: architecture only; independent of all historical data

```bash
python scripts/adaptive/check_fusion_image_model.py \
  --upstream-repo /projects/bcrv/dkourtesis/fusion_model_basics \
  --out-dir /projects/bcrv/dkourtesis/applecider_adaptive/new_image_checks/architecture \
  --variant patch_images --image-initialization randomized --device cpu
```

This verifies new-source imports, shape `(B,55)` + `(B,3,63,63)`, finite
five-class output and gradients, strict state_dict serialization and reload.
It is NOT a reconstruction of historical accuracy or image pretraining.

For the specifically **astronomically pretrained** model, run a separate
architecture check with `--variant btsbot_images --image-initialization pretrained
--hf-snapshot /YOUR/READABLE/HF/SNAPSHOT` and `--device cuda` on a compute node.
Do not assume collaborator-owned paths are accessible to you.

### Check 2: read-only historical data, labels, and alert-stage audit

Locate the new repo's **actual data files** first: the Git checkout does not
contain the 237 GB NPZ dataset! Its README references a *collaborator-owned*
`data_ztf_preprocessed/built_all.csv` that may be inaccessible. Do not copy
private data or assume permissions. If authorized, point to a readable index
in your own work area, or preprocess with the owner's data permissions.

```bash
NEW_INDEX=/path/to/accessible/data_ztf_preprocessed/built_all.csv
REG=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_v1/registry.toml
SHA=$(git -C /projects/bcrv/dkourtesis/fusion_model_basics rev-parse HEAD)

python scripts/adaptive/audit_fusion_image.py \
 --registry "$REG" --new-index "$NEW_INDEX" \
 --model-sha "$SHA" --check-stages 20 \
 --out-dir /projects/bcrv/dkourtesis/applecider_adaptive/new_image_checks/data_audit
```

Optional `--new-split-dir /path/to/splits_a_min200` reports split overlap,
but these five-class scheme-A splits are **not** a substitute for the original
historical fixed splits. `--check-stages -1` performs a potentially expensive
full stage validation (run in a CPU job, not on login node). The audit refuses split overlap, inconsistent historical labels, new NPZ
duplicate IDs, and unmatched/ambiguous alert-stage IDs. It audits fine
labels from `built_all.csv` against the original taxonomy. Missing coverage or
label disagreements are reported; the AL runner rejects them.

Do not infer that a stage match establishes pixel-level or temporal
measurement equality; original samples and new NPZs may come from different
preprocessing snapshots. Pixel/metadata provenance comparison is needed before
claiming strict paired-data reproduction.

### Check 3: small actual-data CPU/GPU training pilot, if stage-matched

Run audit with `--make-smoke --smoke-per-class 2` to generate five-class
`smoke_train.csv` and `smoke_validation.csv`. The audit must find two eligible
objects per class in each split; it will not relabel/drop nonmatching objects.

```bash
python scripts/adaptive/audit_fusion_image.py \
 --registry "$REG" --new-index "$NEW_INDEX" --model-sha "$SHA" \
 --check-stages 20 --make-smoke --smoke-per-class 2 \
 --out-dir /projects/bcrv/dkourtesis/applecider_adaptive/new_image_checks/data_audit

python scripts/adaptive/run_fusion_image_smoke.py \
 --registry "$REG" --new-index "$NEW_INDEX" \
 --upstream-repo /projects/bcrv/dkourtesis/fusion_model_basics \
 --train-manifest /projects/bcrv/dkourtesis/applecider_adaptive/new_image_checks/data_audit/smoke_train.csv \
 --validation-manifest /projects/bcrv/dkourtesis/applecider_adaptive/new_image_checks/data_audit/smoke_validation.csv \
 --out-dir /projects/bcrv/dkourtesis/applecider_adaptive/new_image_checks/smoke_run \
 --epochs 1 --max-batches 1 --device cpu
```

Results: `best_vit_tower.pth`, `training_summary.json`, `training_history.json`.
The historical filename is retained solely because the shared AL coordinator
hardcodes the predecessor filename. **The file contains the entire new model**,
not a historical ViT-tower-only state dictionary.
Smoke batches are deliberately too small for scientific metrics.

## Later AL experiment integration: prepared, NOT approved to run

`scripts/run_fusion_image_al.py` instantiates the existing generic coordinator
with the new predictor/trainer/evaluator. No changes to `SimulatedALConfig`,
`run_state`, candidate selection, old `AstroMiNNViTTrainer`, or Task 4 outputs.
It supports the same three init policies, four acquisition policies and nine
cycles. The predictor exports the model's CLS embedding for the entropy +
diversity policy. The historical coordinator expects the filename
`best_vit_tower.pth`, so the adapter uses that name for the **full new-model**
checkpoint; checkpoint contents and source version are checked before reuse. It **fails closed** unless supplied an audit asserting full historical
NPZ coverage AND verification of every original alert stage. Do not request
`--run` until the actual-data smoke pilot passes and we confirm scientific
compatibility. For source and checkpoint provenance, each suite must use a
new ID, new checkpoint, and explicit upstream git SHA.

### Technical limitations intentionally NOT hidden

- Full AL compatibility remains conditional on actual data and stage matches.
- `run_fusion_image_al.py` is an opt-in scaffold, not yet validated by a GPU
  end-to-end closed-loop test with this new upstream architecture.
- The pilot trainer currently uses historical weighted cross entropy, bf16 on
  CUDA, no EMA, no LR scheduler or training augmentation. It does not claim
  numerical reproduction of the upstream TEMPO-fusion recipe, nor the original
  historical ViT training recipe's complete augmentation schedule.
- Full NPZ loading may be I/O intensive. Profile CPU memory and throughput
  before scheduling large multi-seed experiments; production caching/sharding
  and input-stage parity audits may be necessary.
- F1 checkpoint selection is available; early stopping remains tied to
  validation loss. Test labels are *never* involved in checkpoint selection.
- The public repo lacks pinned requirements/build files and binary weights.
  Preserve `source_revision`, HF snapshot hashes, `npz` manifest hashes, and
  model configuration for each comparison.

## Science plan after passing all gates

1. Train a new **five-class L0 baseline**, check label order, train/val losses,
   shape, checkpoint reload, all class counts, fixed split/evaluation.
2. Run same-data, same-model three initialization modes: fresh ViT initialization
   (image pretraining held fixed), restart from L0, true previous-cycle warm start.
   Each policy/seed has an independent nine-cycle trajectory.
3. Evaluate class weight exponent and equal-object loss contribution on saved
   cumulative manifests (fixed-data controls). Separately compare loss-best vs
   object-level F1-best checkpoint on those same trajectories.
4. Only afterward evaluate new uncertainty/diversity acquisition approaches.

Never present scheme-A fusion metrics as historic AstroMiNN image-only results.
