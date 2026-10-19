# Task 0 — BTSbot pretraining vs random ConvNeXt (full historical training)

## Scope and scientific comparison

Run **six fully independent fits** in the same standalone five-class AstroMiNN image+metadata model:

| Backbone initialization | Seeds |
|:--|:--|
| `pretrained` (GalaxyZoo pretrained BTSbot ConvNeXt-pico) | 135, 246, 357 |
| `randomized` (fresh timm ConvNeXt-pico, no pretrained image weights) | 135, 246, 357 |

Everything else is identical, based on successful pre–Task 0 `cosine_aug`:

- Full historical training split: **4,729 objects, 29,161 alert rows**.
- Fixed historical validation split: **2,365 objects, 14,148 alert rows**.
- Trainable image+metadata model; **not** the old ViT/expert-tower model.
- AdamW, learning rate `1e-4`, weight decay `.01`, batch size 64.
- First three epochs warmup (`0.1×, 0.55×, 1×`); subsequent cosine decay to `0.05×` over 80 epochs.
- 80 epoch cap, patience 15 based on validation weighted CE. Full-history train alert-row counts raised to `-0.85`, normalized to mean class weight 1.
- Train-only D4 image augmentation (the same random rotation/reflection across all 3 channels of each cutout). No validation/test augmentation.
- No EMA. Save **both** best validation loss checkpoint `best_vit_tower.pth` (historical naming convention; it is **not** a ViT) and best validation object macro-F1 checkpoint `best_object_f1.pth`. **Loss-selected is the frozen primary criterion**.
- BCE? **No**: weighted multiclass cross entropy; original five-class taxonomy `AGN, TDE, SN I, SN II, CV` in that exact order.
- **No test access** during six training fits or recipe selection. No AL pool, oracle or acquisition; **not** the 476-object initial AL L0.

Run folder: `/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task0_full_historical_v1`.
The original pre–Task 0 run folder remains untouched.

The random condition uses `NewImageModelSpec(image_initialization='randomized')`: it verifies the same pinned pretrained snapshot/upstream revision then reconstructs a fresh `timm.create_model(..., pretrained=False, num_classes=0)` for the image backbone. That deliberately preserves the same model architecture and training preprocessing; all other components are initialized normally in both cases. This is **not** a ViT-vs-ConvNeXt architecture test.

## 1. Transfer and install

From **Windows PowerShell** (ZIP must be in Downloads):

```powershell
scp "$HOME\Downloads\fusion_task0_full_history_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

From **Delta login / VS Code terminal**:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_task0_full_history_v1.zip -d .

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/astronomy/fusion_image/test_task0_full_history_v1.py -q
```

These files **add** Task 0 scripts; they do not overwrite the working v3 training runner or core recipe-training code.

## 2. Audit and freeze the immutable Task 0 plan

```bash
bash scripts/adaptive/submit_task0_full_history.sh --check
bash scripts/adaptive/submit_task0_full_history.sh --prepare
```

`--check` verifies all historical manifests, 24 source shards, pinned upstream/HF identities, the 2.07GB resident cache, and that the reference `cosine_aug` training was completed with the expected recipe. Both commands submit **zero GPU jobs**.

If any pinned source file changes after freezing `plan.json`, future job runs fail rather than silently switch implementation/weights/datasets. Use a new run directory/version if legitimate experimental code changes are required.

## 3. Submit six 15-minute fits

```bash
bash scripts/adaptive/submit_task0_full_history.sh --submit-all
```

Or first submit only `pretrained_s135`, then submit remaining cases individually:

```bash
bash scripts/adaptive/submit_task0_full_history.sh --submit-case pretrained_s135
bash scripts/adaptive/submit_task0_full_history.sh --submit-case pretrained_s246
bash scripts/adaptive/submit_task0_full_history.sh --submit-case pretrained_s357
bash scripts/adaptive/submit_task0_full_history.sh --submit-case randomized_s135
bash scripts/adaptive/submit_task0_full_history.sh --submit-case randomized_s246
bash scripts/adaptive/submit_task0_full_history.sh --submit-case randomized_s357
```

**Do not use both submission modes for the same case.** The launcher prevents duplicate submissions and overwriting partial outputs.

```bash
bash scripts/adaptive/submit_task0_full_history.sh --status
sacct -j <job_id> -X --format=JobID,State,ExitCode,Elapsed,Timelimit
```

**15 minutes is an allocation, not a guaranteed duration.** Previous augmented full-data fit completed in ~9 minutes, but randomized initialization may train longer. An interrupted training fit writes `training_history.json` every epoch and preserves any checkpoint; it does **not** count as a completed Task 0 fit until `training_summary.json` exists. Do not infer 80 epochs were run after a TIMEOUT or overwrite the partial output. Inspect early-stop curves before deciding whether to resubmit a distinct retry run with a longer wall limit.

## 4. Report paired validation comparisons (after training)

```bash
bash scripts/adaptive/submit_task0_full_history.sh --report
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task0_full_historical_v1
cat "$ROOT/comparison.csv"
cat "$ROOT/paired_differences.csv"
cat "$ROOT/comparison.json"
```

`paired_differences.csv` is pretrained minus randomized **within the same seed**, for object macro-F1, balanced accuracy, accuracy, ROC-AUC, average precision, ECE and log loss. `comparison.json` provides group means and sample SD. All results are validation-only. Full-data validation object macro-F1 and stability are the primary Task 0 diagnostics.

## 5. OLD ViT comparison — CODE READY, TEST SET LOCKED UNTIL AFTER TRAINING

The old ViT and new ConvNeXt architectures should **not** be evaluated with each other's training/validation objects as if they were independent tests. Our old ViT evaluation `validation_manifest_sha256` was `afc3de9554cf682da586326526b1deba046c52970d84c3692891f1de3eb79d30`, matching the new model historical validation identity. This enables a matched validation population, but it remains a tuning split, **not** a final impartial architecture benchmark.

`task0_vit_test_compare.py` implements a three-step *post-training* workflow. It does not run during Task 0 fits or `--report`. It requires every Task 0 run to have completed **and** an explicit `--confirm-frozen-final-test` flag for any access to test manifests/metrics.

### Step A: Audit identical train/validation/test population

Provide the OLD ViT's **original full historical train, validation and test manifests**, not the small 476-object AL L0 manifest, and not the 4,253-object candidate pool by itself. The old ViT full-data checkpoint should come from **cycle 9** of the nine-cycle historical experiment. The code requires identical object IDs/labels AND a canonical `(object_id, alert, label)` multiset for the old and new train/validation/test manifests, including strict within-model and cross-model leakage checks. It rejects nonidentical test populations rather than presenting misleading head-to-head scores.

Example (verify these real paths on Delta; do not substitute the cycle-0 L0 manifest):

```bash
VIT_HIST=/projects/bcrv/dkourtesis/applecider_adaptive/historical/astrominn_vit135_20260122
python scripts/adaptive/task0_vit_test_compare.py audit \
  --vit-train "$VIT_HIST/train_manifest.csv" \
  --vit-validation "$VIT_HIST/validation_manifest.csv" \
  --vit-test "$VIT_HIST/test_manifest.csv" \
  --confirm-frozen-final-test
```

The audit writes a frozen `final_test_split_audit.json` if and only if all checks pass. This **does not prove from the checkpoint alone** which data trained an arbitrary old ViT; the old experiment provenance is checked separately against the supplied old experiment record and cycle-9 checkpoint in the comparison step.

### Step B: Run *new model* test inference (after recipe/model-selection decisions are frozen)

```bash
bash scripts/adaptive/submit_task0_final_test.sh --submit-case pretrained_s135
```

Repeat for whatever Task 0 cases you **predeclare** for a final test comparison; ideally all six to avoid selection by validation/test performance confounding interpretation. Each result stores checkpoint hash, split hash, object metrics, sample metrics, and per-class metrics in:

`$ROOT/final_test_evaluation/<case>/evaluation_summary.json`.

**PERFORMANCE WARNING:** The existing ~2GB resident fast cache contains TRAIN+VALIDATION **only**. This code deliberately uses the known-correct stage-NPZ loader for TEST rather than silently treating test data as training-cache rows. The separate `submit_task0_final_test.sh` requests up to **2 hours**, not the 15-minute training limit. Direct NPZ test inference may require a substantially longer, separate GPU job (previous direct historical fetching was only ~2.7–8.7 samples/s); it should **not** be allocated the 15-minute *training* default. If this stage is too slow, build an independently audited test-only cache without entering the training directory and rerun this step on that immutable cache. No test evaluation runs are automatically scheduled by the six-fit training launcher; each final test job requires a separate explicit `--submit-case` action.

### Step C: Compare against archived old ViT cycle-9 evaluation

Pass an actual completed **old ViT cycle-9** evaluation summary (`evaluation_summary.json`), the associated seed/policy `experiment.json`, and the **exact** matching old checkpoint. For example:

```bash
VIT_RUN=/work/hdd/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_closed_loop_warm_v1/entropy/seed_135
VIT_SHARED=/projects/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_closed_loop_warm_v1
python scripts/adaptive/task0_vit_test_compare.py compare \
  --vit-evaluation "$VIT_RUN/cycle_009/evaluation/evaluation_summary.json" \
  --vit-experiment "$VIT_SHARED/entropy/seed_135/experiment.json" \
  --vit-checkpoint "$VIT_RUN/cycle_009/training/best_vit_tower.pth" \
  --confirm-frozen-final-test
```

**The above paths are illustrative** based on the past experiment artifact tree; inspect the actual completed artifacts (`find ... -name evaluation_summary.json` or check old experiment records) for the correct evaluation and experiment paths. The original L0/candidate manifest paths referenced inside `experiment.json` must also still exist and match their recorded hashes. The tool verifies checkpoint SHA, five-class order, cycle-9 completion, old experiment validation identity, and the SHA-256 hashes and object-ID union of the **original 476-object L0 plus 4,253-object candidate pool** against the declared 4,729-object full training set, as well as full test object/sample counts and new Task 0 evaluation identities. It refuses to compare if no audited new test results exist.

Outputs: `$ROOT/final_vit_comparison/paired_test_summary.csv` and JSON. Any old ViT seed/policy versus new model comparison with different training protocols is an architectural baseline, not an isolated pretrained-vs-randomized causal comparison; the clean causal comparison is the six **same-architecture** Task 0 fits.

## Notes on reproducibility

- The old `best_vit_tower.pth` filename in the new model is inherited from the current model API. It is a **ConvNeXt image+metadata** checkpoint, not a historical ViT checkpoint.
- The five-class ordering in the Python contracts is `AGN,TDE,SN I,SN II,CV`; never use the upstream repository's unrelated class-order scheme.
- Best-F1 secondary checkpoints are saved for diagnostic comparison. Do not select different checkpoints by test outcomes. Primary selection is validation weighted CE.
- Training uses and preserves the existing resident-HDD cache and no `test_manifest` during Task 0 preparation/fits. The final test-audit stage is intentionally separate.
