# Task 8A — Retrospective checkpoint acceptance audit (v1)

## Scientific objective

Inspect saved, object-level validation predictions from 9-cycle closed-loop
AstroMiNN experiments to determine *where* successive checkpoints regress,
*which validation objects* change prediction, whether class recall changes are
plausibly noisy, and which **illustrative** gate rules would first reject a
candidate. This is **not** an acceptance-controlled closed-loop experiment.

The primary reference is **Task 4** (`astrominn_closed_loop_warm_v1`, exponent
0.85, per-sample weighting, warm-start, best-validation-loss checkpoints).
Tasks 6B (object-macro-F1-best epoch selection) and 6C (equal new-object
weighting, loss-best) are secondary robustness and mechanism checks. These
secondary trajectories have different acquired data and/or model lineages.
Each of 3 suites has 3 seeds × 2 acquisition policies × 9 comparisons, where
all data are present: **162 adjacent saved comparisons**.

The classes are **AGN, TDE, SN I, SN II, CV** in the stored model index order.
Comparisons use identical object IDs and the latest-alert aggregated object
predictions. Each object is included once, not once per alert.

### Stage analyses

1. **Regression map.** Adjacent saved-checkpoint macro-F1, balanced accuracy,
   accuracy, their differences and paired percentile bootstrap intervals.
2. **Paired prediction flips.** Lost vs recovered correct classifications per
   class, the identities of changed validation objects, predicted label changes.
3. **Class-stratified paired bootstrap.** Resample the same object indices for
   incumbent and candidate *within each ground-truth class*. Fixed support per
   class, fixed RNG seeds, default 600 iterations and 2.5/97.5 percentiles.
   Bootstrap frequencies are descriptive, **not** guaranteed posterior
   probabilities or formally calibrated significance tests.
4. **Acquired-batch diagnostics.** Actual revealed class counts, acquisition
   budget, mean alert multiplicity when the object-id manifest column is
   verifiable. No new label resolution or oracle queries.
5. **Exploratory acceptance-rule panel.** These are candidate assumptions,
   **not selected hyperparameters**:
    - `strict_macro`: candidate macro-F1 must be nondecreasing.
    - `macro_tolerance_1pp`: allow at most 1 percentage-point macro-F1 loss.
    - `macro_balanced_guard`: above plus balanced-accuracy loss ≤1.5 pp.
    - `macro_balanced_rare_guard`: above plus net losses of correctly predicted
      validation objects ≤1 TDE, ≤2 SN II and ≤2 CV.

   Recall guards act on *counts of lost minus recovered objects* and do not
   impose the same recall-% floor across classes with different supports. The
   values are strictly illustrative. We will examine whether these constraints
   behave sensibly before choosing any gate. The first-rejection simulation
   terminates immediately after the **first rejected checkpoint** for each
   suite/seed/policy/rule; later saved trajectories are not valid causal
   counterfactuals after a rejection. Per-cycle adjacent comparisons remain
   descriptive even after a hypothetical earlier rejection.

## Architecture

- `src/al_pipeline/core/analysis/checkpoint_acceptance.py`: pure generic
  object-level metrics, paired deltas, bootstrap, rule contracts; no astro or
  filesystem imports.
- `src/al_pipeline/astronomy/analysis/checkpoint_audit.py`: AstroMiNN NPZ
  and JSON readers, snapshot alignment, experiment paths, revealed labels,
  alert multiplicity diagnostics.
- `scripts/adaptive/task8a.py`: audit orchestration and CSV/JSON reporting.
- `scripts/adaptive/task8a_plot.py`: optional PNG plots from generated CSV reports.
- `scripts/adaptive/task8a_worker.sbatch`: single *read-only* reporting job;
  no training, inference, GPU kernels, or modifications to source experiments.
  A GPU is allocated only because the available Delta account is GPU-only.
- Tests: `tests/core/test_checkpoint_acceptance.py`,
  `tests/astronomy/test_task8a_snapshot.py`, `tests/jobs/test_task8a_v1.py`.

## Install and test on Delta

From Windows PowerShell, after saving the supplied ZIP in Downloads:

```powershell
scp "$HOME\Downloads\task8a_checkpoint_acceptance_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

From Delta login:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/task8a_checkpoint_acceptance_v1.zip -d .
conda activate al-pipeline
python -m pytest tests/core/test_checkpoint_acceptance.py \
  tests/astronomy/test_task8a_snapshot.py tests/jobs/test_task8a_v1.py -q
python scripts/adaptive/task8a.py plan
```

If the repository was updated previously, inspect paths before running an
unconditional `unzip -o`; installation only adds Task 8A files in v1.

## Run once on the available GPU partition (CPU analysis)

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
OUT=/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task8a_retrospective_acceptance_v1
mkdir -p "$OUT/logs"
sbatch --output="$OUT/logs/T8A_%j.out" --error="$OUT/logs/T8A_%j.err" \
  scripts/adaptive/task8a_worker.sbatch "$PWD" "$OUT"
```

The source code reads snapshots only under the three fixed suite names, and
writes results only under the requested `--output`. No SLURM dependency chains,
training fits, or model inference are launched. Do **not** run `--output`
inside the experiment's own `runs/` tree. It is acceptable for Tasks 6B/6C
persistent state directories on `/projects` to be symbolic links to HDD.

If Task 6C finishes before the audit runs, all secondary comparisons are
included. If a trailing cycle is missing in Task 6C, default behavior permits
only a contiguous completed prefix. Interior missing cycles fail closed.
Pass `--require-complete-secondary` to fail on any missing secondary cycle.

## Results and transfer

The report output includes:

- `summary.json`: protocol, number of comparisons, completeness, limitations,
  trajectory and rule summaries.
- `adjacent_comparisons.csv`: all saved adjacent validation comparisons and
  bootstrap macro-F1/balanced-accuracy intervals.
- `per_class_changes.csv`: TDE, SN II, CV, AGN, SN I recall/F1 changes and
  lost/recovered counts, classwise bootstrap intervals.
- `prediction_flips.csv`: individual changed validation object IDs, predicted
  classes, correction/loss flags. **Internal research provenance**.
- `acquisition_batches.csv`: actual selected class composition and alert
  multiplicity.
- `exploratory_rules.csv`: decision records up to each hypothetical **first**
  rejection, not beyond.
- `first_rejections.csv`: first rejected cycle per trajectory/rule.
- `trajectory_summary.csv`, `rule_summary.csv`: concise summaries.

When complete, on Delta (figures are optional and generated from saved CSVs):

```bash
OUT=/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task8a_retrospective_acceptance_v1
python scripts/adaptive/task8a_plot.py --report "$OUT"
cat "$OUT/summary.json"
tar -czf /work/hdd/bcrv/dkourtesis/applecider_adaptive/task8a_review_v1.tar.gz \
  -C "$OUT" summary.json adjacent_comparisons.csv per_class_changes.csv \
  prediction_flips.csv acquisition_batches.csv exploratory_rules.csv \
  first_rejections.csv trajectory_summary.csv rule_summary.csv README.md \
  task8a_task4_macro_regression_map.png task8a_task4_rare_recall_changes.png \
  task8a_task4_first_rejections.png
```

PowerShell:

```powershell
scp "dkourtesis@dt-login.delta.ncsa.illinois.edu:/work/hdd/bcrv/dkourtesis/applecider_adaptive/task8a_review_v1.tar.gz" "$HOME\Downloads\"
```

Upload the archive for analysis **before** selecting a checkpoint-acceptance
rule or preparing Task 8B retry training.

## Methodological limitations

- The validation set was also used for within-cycle epoch selection. Reusing
  the same objects for cross-cycle gates can overfit the acceptance rule. The
  bootstrap only estimates paired sampling variation conditional on these
  saved models; it **cannot** remove adaptive selection bias.
- Class-stratified bootstrap preserves minority-class supports (TDE has few
  examples) but cannot invent missing rare-class diversity. Interpretation
  requires inspecting paired lost/recovered object counts.
- The audit does **not** open test prediction NPZ files or use test metrics.
  Provenance requires parsing an `evaluation_summary.json` that may contain
  metadata for other splits, including test. These other split metrics are
  ignored; all comparisons use only `validation_predictions.npz`.
- If no model is rejected by a rule on a trajectory, Task 8B has no first
  rejection to retry for that rule; do not invent one.
- Model-weight hashes are not recomputed; evaluation/checkpoint provenance is
  validated from stored paths, class-order, and cycle summary. Missing or
  inconsistent required artifacts fail closed.
- Trajectory cycles share models, validation objects, and acquisition history.
  Do not treat them as independent observations in statistical comparisons.

## CPU split execution (recommended when long jobs queue slowly)

The three training configurations can be audited independently without altering the
statistical analysis. Run three 20-minute CPU jobs, each with 600 paired,
class-stratified bootstrap resamples per adjacent saved checkpoint:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
OUT=/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task8a_retrospective_acceptance_v1
mkdir -p "$OUT/logs"
for SUITE in task4 task6b task6c; do
  sbatch --job-name="T8A_${SUITE}" \
    --output="$OUT/logs/T8A_${SUITE}_%j.out" \
    --error="$OUT/logs/T8A_${SUITE}_%j.err" \
    scripts/adaptive/task8a_worker_cpu_shard.sbatch \
    "$PWD" "$OUT/shards/$SUITE" "$SUITE"
done
```

Check that all three individual `shards/{task4,task6b,task6c}/summary.json`
files exist. Then merge and plot **without** submitting a fourth job:

```bash
python scripts/adaptive/task8a_merge.py --output "$OUT"
python scripts/adaptive/task8a_plot.py --report "$OUT"
cat "$OUT/summary.json"
```

Merge checks source-suite identity, 6 trajectories per shard, bootstrap settings,
row counts, and class-row counts. Any missing or inconsistent shard causes a
hard failure rather than creating a partial combined report. These CPU jobs do
not run training, change checkpoint ancestry, or access test predictions.

Do **not** run the original one-job wrapper simultaneously with these jobs.
