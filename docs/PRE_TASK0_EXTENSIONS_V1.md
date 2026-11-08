# Pre–Task 0 extension v1

**Purpose:** two training-only, validation-only follow-up fits after the successful
full historical `cosine_aug` run. Prepares a **new frozen plan and output identity**;
does not modify `new_image_pre_task0_recipe_resident_v3`, its source files, or outputs.

## Models, populations, and controls

- Five-class standalone AstroMiNN image + metadata, GalaxyZoo-pretrained BTSbot
  ConvNeXt-pico, **not** the historical frozen-expert ViT.
- Complete historical training: 4,729 objects / 29,161 alert rows.
- Fixed validation: 2,365 objects / 14,148 rows.
- A40, bf16, AdamW, batch 64, seed 135, peak LR 1e-4, weight decay .01,
  max 80 epochs, patience 15, loss-best primary and macro-F1-best secondary checkpoint.
- Same verified HDD consolidated cache, sequentially loaded once into host RAM.
  No test set, label oracle, or acquisition. Existing original v3 experiments remain unchanged.

| Case | Difference from completed `cosine_aug` reference |
| ---- | ------------------------------------------------ |
| `cosine_aug_short` | 3-epoch warmup; cosine LR reaches floor 0.05× at epoch 25, remains there for epochs 26–80; same exponent 0.85 weights |
| `cosine_aug_effnum` | Keep original 80-epoch cosine and augmentation; use effective-number weighted CE, beta 0.999 on **alert-row counts**, normalized mean weight 1 |

**Count basis matters:** the previous original v3 `cosine_aug_effnum` recipe
calculates effective-number weights on unique **objects**. Here we use **alert rows**
deliberately because the baseline inverse-frequency recipe uses row counts.
This changes one causal factor (the weighting formula) rather than the formula
and counting unit. A future object-level weight comparison would need its own experiment.

Weights are recomputed on *training* rows only. Train/validation membership is audited.
Losses with different weight vectors should **not** be compared directly across
weighting experiments. Prefer macro-F1, balanced accuracy, per-class metrics,
ROC/PR AUC and identical unweighted log loss for model selection.

## Install and run (Delta)

From Windows PowerShell:
```
scp "$HOME\Downloads\fusion_pre_task0_extensions_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

From Delta:
```
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_pre_task0_extensions_v1.zip -d .
python -m pytest tests/astronomy/fusion_image/test_pre_task0_extensions_v1.py -q
bash scripts/research/fusion_image/recipe/submit_fusion_pre_task0_extensions.sh --check
bash scripts/research/fusion_image/recipe/submit_fusion_pre_task0_extensions.sh --prepare
bash scripts/research/fusion_image/recipe/submit_fusion_pre_task0_extensions.sh --submit-case cosine_aug_short
bash scripts/research/fusion_image/recipe/submit_fusion_pre_task0_extensions.sh --submit-case cosine_aug_effnum
bash scripts/research/fusion_image/recipe/submit_fusion_pre_task0_extensions.sh --status
```
Both jobs are independent and request 15 minutes, **not guaranteed sufficient** for
an unusually long 80-epoch run. Training history is saved after each epoch; partial
output is preserved for audit, and retries must use a new experiment identity.
After completion:
```
bash scripts/research/fusion_image/recipe/submit_fusion_pre_task0_extensions.sh --report
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_extensions_v1
cat "$ROOT/comparison.csv"
```

## Pretraining and fair test evaluation

**We already use the GalaxyZoo-pretrained BTSbot ConvNeXt image backbone.**
The rest of this standalone five-class model is not a checkpoint of the historical
ViT with frozen expert towers. Task 0 is the controlled pretrained-vs-randomized
**same ConvNeXt architecture** comparison.

The test split has deliberately not been read during recipe tuning.
After freezing the recipe, compare test metrics only for a given model's strictly
held-out objects, using exactly the same class mapping and object aggregation.
To compare old ViT vs new model head-to-head, audit both train/validation/test
object sets first. Sharing a test set is only safe if all its objects were absent
from **both models' training and tuning/validation**. If the old and new held-out
populations differ, report separate test results as descriptive, *not* a
controlled architecture comparison. Prefer a common independent held-out
intersection, if sufficiently large and representative, or an untouched new test
population. Do not select recipes/checkpoints on test metrics.

## Optional future read-only split overlap audit

Once training-recipe selection is finalized, use
`scripts/adaptive/audit_vit_new_test_membership.py` with **six explicit manifest
paths** (`--old-train`, `--old-validation`, `--old-test`,
`--new-train`, `--new-validation`, `--new-test`).
It audits object IDs and catches cross-model split overlap without touching labels
or computing test metrics. Optional `--old-id-column` / `--new-id-column` can
handle differing manifest formats. Run it before declaring a head-to-head
comparison; it is not invoked by the training jobs.
