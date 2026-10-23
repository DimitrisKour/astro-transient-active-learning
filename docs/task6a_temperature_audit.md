# Task 6A — read-only temperature-scaling audit

**No new ViT inference, training, or access to test data.** Script compares both completed warm-start suites:

- `.85` — `astrominn_closed_loop_warm_v1`
- `.5` — `astrominn_closed_loop_warm_weight05_newl0_v1`
- Seeds: `135`, `246`, `357`; policies: `entropy`, `entropy_balanced_explore`
- Stages: `0`, `3`, `5`, `9`
- **48 analysis rows**, comprising **42 distinct checkpoints**: for each suite and seed, L0 is the same checkpoint for both policies; we retain both policy rows to assess their separate saved U0 pool inferences. **24 paired rows** (6 per stage) allow model comparisons at matched budgets, but trajectories acquire different objects.

## Calibration procedure

The existing evaluator saves `validation_predictions.npz` per checkpoint, including latest-alert **object-level raw logits** (`object_values`), `object_targets`, and `object_ids`. The script uses those values, and requires the same validation IDs and labels for both experiments. It does not use sample-level output in its primary analysis, to match object-level acquisition.

Five deterministic class-stratified folds (on fixed object IDs/labels) are shared across both suites and all checkpoints. For each held-out fold, temperature is fitted using four folds by minimizing object-level NLL, with a bounded scalar optimizer. Predictions on the held-out fold are then evaluated; combining the five folds gives **out-of-fold calibration metrics**. Temperature fitted to **all validation** is recorded *separately* and used for the prospective ranking overlap diagnostic. Do not compare in-sample post-fit metrics to the out-of-fold results as though they were the same estimand.

Reports include NLL, Brier, 15-bin top-label ECE, mean confidence, accuracy, class-wise one-vs-rest Brier and 10-bin ECE. Temperature scaling never changes argmax predictions; macro-F1 and balanced accuracy therefore cannot improve merely by scaling the logits.

**Limitation:** The existing ViT checkpoints were selected on validation loss using the full validation split. Cross-fitting the temperature avoids fitting it on an evaluated fold but does *not* undo validation reuse during neural checkpoint selection. Results are a retrospective diagnostic, not fully independent generalization evidence. No test data are used.

## Acquisition ranking

The saved `cycle_001/inference.npz` corresponds to the stage-0 checkpoint, `cycle_004/inference.npz` to stage 3, and `cycle_006/inference.npz` to stage 5. Their logits were generated before acquisition using that checkpoint. We compute baseline and full-validation-fitted-temperature entropy rankings and report top-473 overlap, exchanges, and full pool size. We verify `cycle_summary.json`'s inference-checkpoint path. These are **counterfactual top-k rankings**, not actual acquisitions or new labels.

For `entropy_balanced_explore`, the entropy ranking is explicitly labeled a **proxy**, not a reproduction of the balanced-exploration selection algorithm. Stage 9 has no remaining pool because cycle 9 acquired the remainder; report `pool_exhausted`, not a fabricated ranking. Since the two trajectories acquire different objects, we compare each model's *own* unscaled/scaled ranking; there is no false one-to-one pool matching across weighting schemes.

## Install and inspect (no submissions)

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/task6a_temperature_audit_v1.zip -d .
python -m pytest tests/astronomy/test_task6a_temperature_audit.py -q
python scripts/adaptive/task6a_temperature_audit.py plan
```

Run offline after both suites have their checkpoints and evaluations:

```bash
python scripts/adaptive/task6a_temperature_audit.py run
```

Outputs under `/projects/bcrv/dkourtesis/applecider_adaptive/diagnostics/temperature_05_vs_085_v1`:

- `calibration_per_model.csv`: all 48 rows with raw and five-fold out-of-fold calibration metrics
- `paired_05_vs_085.csv`: 24 matched budget / seed / policy comparisons
- `per_class_calibration.csv`: class-specific reliability
- `pool_entropy_overlap.csv`: counterfactual selection stability
- `summary.json`: protocol and missing-data indicators

## Optional: queue a single dependent CPU job while Task6A is running

`submit` **without `--submit` is a dry run**. It reads the 57-job Task6A submission manifest and validates the per-seed/per-policy dependency graph. On actual submit, it queues one CPU job depending on successful completion of all **six final cycle-9 jobs**. It never resubmits any Task6A work.

```bash
python scripts/adaptive/task6a_temperature_audit.py submit
python scripts/adaptive/task6a_temperature_audit.py submit --submit
```

The CPU job calls only the audit script and writes to the diagnostic output folder. If you queue this after Task6A cycle-9 jobs have long since left the scheduler's accounting, dependency resolution can fail. In that case run the `run` command directly (no SLURM dependencies). If the analysis reports missing evaluation artifacts, inspect the relevant Task6A jobs and re-run only the diagnostic after resolving their outputs; do not touch training source.

The CPU job is configured for Delta's CPU partition/account in `task6a_temperature_audit.sbatch`; **check local partition/account availability** (`sinfo` and `sacctmgr`) before submission. No Git changes or editing frozen Task6A scripts are needed.
