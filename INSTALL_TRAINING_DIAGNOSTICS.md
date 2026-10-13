# Launch the training diagnostics

The package adds new diagnostic files to the existing al-pipeline checkout. It does not change the existing trainer. Nothing has been submitted from ChatGPT; submit using your Delta SSH session.

## Default run

- Two original-seed-135 anchor fits: the saved balanced-exploration cycle-6 and cycle-7 datasets, unchanged order.
- Forty paired repetitions: the same two fixed datasets, trained independently under seeds 1001–1020; original loader seed 135 stays fixed.
- Fifteen full-data fits: for each of seeds 135,246,357, train one canonical-order reference and four original-policy-order models. All five use the same full data and settings; only row order changes.
- Total: 57 independent fits. No acquisition cycles or bootstrap training.

## Install on Delta

Upload training_diagnostics_v1.zip to the existing al-pipeline repository folder using VS Code or your usual file-transfer method. From that repository root:

```bash
unzip -o training_diagnostics_v1.zip -d .
bash scripts/adaptive/submit_training_diagnostics.sh --prepare-only
bash scripts/adaptive/submit_training_diagnostics.sh 00:15:00 00:20:00
```

If preparation fails, stop before submission and inspect its error. It checks saved training metadata, dataset content, split disjointness, input files and output locations. No GPU work occurs during prepare-only.

Default limits: 15 minutes per cycle-6/7 fit, 20 minutes per full-data fit. These are limits, not expected runtimes. Recorded training alone reached 6.24 minutes at cycle 7 and 7.26 minutes at full data, so 5–7-minute total-job limits are too tight for some fits plus evaluation. The original 50-epoch maximum and patience 10 are retained.

The two anchor jobs run first. Remaining jobs depend on both completing successfully. Thereafter up to three repetitions and one full-data fit run simultaneously, subject to scheduler availability. Cycle 7 does not depend on cycle 6: both train from scratch on already-saved data. Jobs use frozen source code, allowing later work in the live checkout.

## Results

Small metadata, logs and results:

`/projects/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1`

Checkpoints, predictions and runtime:

`/work/hdd/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1`

Generate the report after jobs finish:

```bash
python scripts/report_training_diagnostics.py \
  --plan /projects/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1/plan.json \
  --output /projects/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1/report
```

Export lightweight results for local inspection:

```bash
python scripts/export_training_diagnostics.py \
  --plan /projects/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1/plan.json \
  --output /tmp/training_diagnostics_v1_results.tar.gz --include-logs
```

Models and raw sample arrays are excluded from the export. A local report can read the extracted plan and task metadata without mounting Delta paths.

## Retry incomplete fits

Only after prior jobs have all stopped:

```bash
bash scripts/adaptive/submit_training_diagnostics.sh 00:25:00 00:30:00 --retry-incomplete
```

Completed fits are skipped. Finished training can be reused if evaluation was interrupted. Interrupted training starts a new attempt, retaining the old partial attempt. The report explicitly lists incomplete fits/pairs; do not interpret only the fastest completed seeds as the final experiment.

## Meaning of the comparisons

Training seeds vary initialization and training randomness, including dropout/augmentation; they are not new acquisition trajectories or independently sampled datasets. The original anchor is excluded from the 20-fresh-seed summary. Paired deltas and their uncertainty quantify training variation on this one fixed cycle-6/7 transition. Full-data initial ViT weights are fingerprinted; the reporter rejects order comparisons with unequal initial weights for the same seed. Validation loss selects checkpoints; test metrics are report-only.

Read docs/training_diagnostics.md for the full design, controls, output tables, overrides and limitations.
