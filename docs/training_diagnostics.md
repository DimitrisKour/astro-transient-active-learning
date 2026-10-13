# Training instability diagnostics for completed Experiment B

These are **training-only** jobs on saved manifests. They do not query an oracle,
score candidates, retrain a bootstrap, advance a cycle, or change existing AL runs.
The runner uses the existing trainer and fixed-split evaluator.

## Default experimental design

| Part | Fixed data | Seeds / controls | Fits |
| --- | --- | --- | ---: |
| Original anchor | Balanced exploration seed-135 manifests at cycles 6 and 7, in their saved row order | Original training seed 135 and original loader seed 135 | 2 |
| Paired repetitions | The same two saved manifests | Fresh training seeds 1001–1020; loader seed remains 135 | 40 |
| Full-data reference and order test | All 4,729 historical training objects / 29,161 sample rows | For each of 135, 246, 357: one canonical order and four original policy orders | 15 |
| **Total** | | | **57** |

The cycle-6 dataset has 3,314 objects / 20,803 sample rows. Cycle 7 has 3,787
objects / 24,208 rows. Every repetition uses these **fixed selected objects**;
it does not simulate another seed's acquisition trajectory. Cycle 7 is trained
from scratch independently of cycle 6, rather than warm-starting from its model.
The selected data and original row order stay fixed within each cycle. Class
weights continue to be recomputed from that manifest under the original recipe.

`training_seed` resets the global process RNG as in the existing trainer. It
affects initialization and training randomness such as dropout/augmentation;
this is **not an initialization-only ablation**. `loader_seed` remains the saved
value 135 for the repetitions. DataLoader shuffling still differs between
different-sized datasets; matched seeds do not make their batches identical.
The original anchor is reported separately and excluded from the 20-seed mean
and uncertainty interval.

The full-data test does not run acquisition cycles. At a given seed, all five
fits have identical sample content, class weights, loader seed and training
recipe. They differ only in base manifest row order: canonical versus the
original random, entropy, entropy-diversity and balanced-exploration orders.
Canonical sorting uses object ID, numerical alert stage, sample path and a
complete-row tie breaker, retaining duplicate rows. It does not disable training
shuffle. The three canonical models are a common full-data reference, not a
claim of optimal achievable performance.

Initial ViT tensors are fingerprinted before training without changing RNG
state. The reporter rejects order comparisons if initial weights differ for
the same seed. The three expert towers remain the same frozen published models.

## Training settings and time requests

Each task copies the **complete saved training policy**, including the original
maximum 50 epochs, patience 10, batch size 512, learning rate 1e-4, augmentation,
class weighting and weighted sample-level validation-loss checkpoint rule.
Only the fresh repetition's training seed changes by default. No epoch cap is
shortened to squeeze fits into a requested job time.

In the completed archive, training alone took 3.56–5.93 minutes at cycle 6,
4.04–6.24 minutes at cycle 7, and 4.76–7.26 minutes at full data. Loading,
evaluation and a longer early-stopping path can add time. Defaults request
**15 minutes per cycle-6/7 fit** and **20 minutes per full-data fit**. These are
limits, not expected runtimes. Five-to-seven-minute limits risk censoring slower
fits; incomplete fits are never silently counted as completed results.

## Install and launch on Delta

Copy the supplied ZIP into the `al-pipeline` checkout, then extract there. The
package contains only new diagnostic files; no existing trainer or AL scripts
are replaced. From the repository root:

```bash
unzip -o /path/to/training_diagnostics_v1.zip -d .
bash scripts/adaptive/submit_training_diagnostics.sh --prepare-only
bash scripts/adaptive/submit_training_diagnostics.sh 00:15:00 00:20:00
```

The prepare-only command validates the 12 completed full-data sources and the
two saved transition datasets, manifest hashes, matched recipes/loader seeds,
full-data equality including duplicate rows, cycle-6 containment in cycle 7,
evaluation disjointness, sample-file availability, expert assets and write access.
It freezes a copy of the Python source, runner, reporter, job script and registry.
Jobs also verify the frozen source, external legacy code, immutable input files
and Python/package versions. You can edit the working checkout for later tasks
while these jobs use its frozen copy. Do not change the shared historical
artifacts, arrays, legacy model code or conda dependencies during these jobs.

The launcher submits three held arrays, then releases them:

1. Anchor: indices 0–1, up to two simultaneous tasks.
2. Fresh pairs: indices 2–41, up to three simultaneous tasks.
3. Full-data orders: indices 42–56, one task at a time.

The latter arrays depend on successful completion of both anchor tasks. After
that, fits are independent; there is no cycle-6 → cycle-7 dependency. An invalid
anchor dependency cancels its dependent arrays instead of leaving them pending
indefinitely. The gate checks successful execution, not agreement with a test
metric threshold. Inspect the separate anchor reproduction report to assess
agreement with the old results.

The default registry is the original replay registry:
`/projects/bcrv/dkourtesis/applecider_adaptive/cycle_jobs_v1/registry_hdd.toml`.

Small outputs:
`/projects/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1`.
Checkpoints, predictions and runtime directories:
`/work/hdd/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1`.
Historical samples are read in place from their existing NVMe paths; none are copied.

Optional environment settings, chosen **before first preparation**:

```bash
export DIAG_REPEAT_COUNT=20
export DIAG_FIRST_REPEAT_SEED=1001
export DIAG_PARALLEL=3
export DIAG_ORDER_PARALLEL=1
# For a scientifically different plan, choose a NEW ID:
# export DIAG_ID=astrominn_training_diagnostics_v2
```

`DIAG_REGISTRY`, `DIAG_SOURCE_SUITE`, `DIAG_SOURCE_ARTIFACT_ROOT`,
`DIAG_PERSISTENT_ROOT`, `DIAG_DATA_ROOT` and `DIAG_LEGACY_CODE_DIR` allow explicit
path overrides. `DIAG_TRAJECTORY_STRATEGY` and `DIAG_TRAJECTORY_SEED` select the
fixed trajectory. `DIAG_REPEAT_LOADER_SEED` is an optional additional design
change, not part of the default. Prepared settings cannot silently change.
Use `--only transition` or `--only order` to submit just that part plus the
anchor gate. Duplicate submissions are refused.

## Retry and report

After earlier jobs have all stopped, retry only incomplete fits with more time:

```bash
bash scripts/adaptive/submit_training_diagnostics.sh 00:25:00 00:30:00 --retry-incomplete
```

Retry refuses active/pending prior arrays. A finished model whose evaluation was
interrupted is reused. An interrupted training fit starts a new attempt from
scratch; its partial attempt is retained. Epoch/optimizer resume is not added.
Training-task locks prevent simultaneous writes to the same fit. Submission
failures cancel only arrays created by that attempt and retain the submission
record. Old AL jobs are never cancelled.

To create the report, after activating the existing `al-pipeline` environment:

```bash
python scripts/report_training_diagnostics.py \
  --plan /projects/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1/plan.json \
  --output /projects/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1/report
```

Reports contain object-level validation/test accuracy, balanced accuracy and
macro-F1; paired cycle-7-minus-cycle-6 deltas; the number/fraction of negative
pairs; per-class precision/recall/F1; checkpoint epochs and runtime; anchor
reproduction differences; original-order-minus-canonical differences; and
explicit missing/incomplete-task accounting. Test metrics are report-only;
the original validation rule selects checkpoints.

The paired mean's 95% interval is a fixed-seed percentile bootstrap with 5,000
resamples. Its scope is **training variability on one fixed acquisition
trajectory**, not independently sampled datasets or acquisition runs. Full-data
means/SD use the three matched training seeds. A report generated before all
fits finish explicitly identifies itself as incomplete.

Export lightweight results for local review without model weights or arrays:

```bash
python scripts/export_training_diagnostics.py \
  --plan /projects/bcrv/dkourtesis/applecider_adaptive/training_diagnostics/astrominn_training_diagnostics_v1/plan.json \
  --output /tmp/training_diagnostics_v1_results.tar.gz --include-logs
```

The report also runs against a locally extracted `plan.json` and small task
results, without the original Delta paths being mounted. Use `--no-plots` if
matplotlib is unavailable.

## Interpretation

If the original-order full-data fits reproduce substantial score differences
with equal initial weights, training order can influence the final outcome.
That alone does not prove that it caused every early-cycle drop. If cycle 7
usually recovers under fresh training seeds, the original bad result is less
representative of that fixed dataset's performance. If it repeatedly declines,
investigate interactions with composition, class weights, optimization and
checkpoint selection. Neither outcome establishes that the new batch is
intrinsically harmful. These controls precede initialization or loss changes.
