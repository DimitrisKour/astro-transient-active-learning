# Simulated-label AstroMiNN active-learning runs

The retrospective runner treats the audited BOOM/TNS labels as a hidden
oracle. A strategy sees only model predictions until an object is selected.
Each strategy owns an isolated pool state, cumulative labeled set, model
checkpoints, and evaluation history.

## One cycle

For cycle `t`, the runner:

1. reconstructs `U_t` from the immutable candidate manifest and the strategy's
   acquisition ledger;
2. runs object-level inference using the latest model checkpoint;
3. ranks `U_t` with random, margin, entropy, or the optional expected
   last-layer Fisher trace and selects the configured budget;
4. reveals hidden labels only for the selected IDs;
5. appends every physical sample for those objects to the cumulative manifest
   `L_t = L_0 union A_1 ... union A_t`;
6. trains the ViT tower while keeping the three expert towers frozen;
7. evaluates the best checkpoint on fixed validation/test manifests and, when
   configured, a fixed held-out subset of the new BOOM/TNS pool;
8. writes the cycle summary and advances durable run state.

Inference, selection, revealed labels, manifests, training output, evaluation,
and completion are separate durable stages. Re-running the same launch command
resumes an interrupted cycle. Experiment fingerprints prevent a run directory
from being reused with another pool, label mapping, baseline checkpoint, or
historical manifest.

## Storage

Small durable state is written below the registry's `persistent_root`:

- run identity and fingerprints;
- run/cycle state;
- selections and revealed labels;
- acquisition ledger and cumulative manifests;
- evaluation history and cycle summaries.

Large products are written below `data_root`:

- per-cycle inference caches;
- ViT checkpoints and training histories;
- evaluation predictions and confusion matrices.

Every new cycle also persists `candidate_scores.npz` in its durable state
directory. It contains the full object ID/score arrays, not only the selected
objects, so strategy correlations and acquisition overlap can be audited
without repeating inference.

## Fisher acquisition

The optional `fisher` strategy captures the normalized 256-dimensional CLS
representation supplied directly to the AstroMiNN ViT classification layer.
For predictive probabilities `p` and final-layer input `h`, it scores

```text
(||h||^2 + 1) * (1 - sum_c p_c^2)
```

which is the expected Fisher-matrix trace for the final linear classifier,
including its bias parameters. The unknown target is averaged over the
model's predictive distribution. This is explicitly a last-layer
approximation, not a full-network Fisher matrix or batch log-determinant
criterion. The inference adapter captures `h` through a temporary hook on
`vit_tower.fc_out`; the historical model source and checkpoint format remain
unchanged.

Because the CLS representation is layer-normalized, its norm may vary only
weakly. Fisher can therefore correlate strongly with probability-only
uncertainty. Use the persisted full score arrays to measure rank correlation
and selected-set overlap with entropy and margin rather than assuming the
strategies are distinct.

## Training initialization policies

- `warm_start`: cycle 1 starts from the published ViT and cycle `t` starts from
  cycle `t-1`. This is conventional sequential fine-tuning.
- `restart_from_baseline`: every cycle starts from the same published ViT but
  trains on cumulative `L_t`. This removes optimizer-path inheritance while
  retaining transfer learning.
- `restart_from_random`: every cycle initializes a new ViT tower and trains on
  cumulative `L_t`; the frozen expert towers remain pretrained. This is the
  most expensive option and must use a trainer configured for random ViT
  initialization.

All policies reset the optimizer and scheduler for each cycle.

`restart_from_random` alone is not sufficient to define a historical replay:
the first acquisition also needs a model trained on a small initial labelled
set. Experiment B therefore has a separate manifest/bootstrap runner; see
`docs/astrominn_historical_replay_experiment_b.md`.

## Safe launch interface

### Expand the labelled pool to all early-alert stages

Notebook 12 intentionally materialized one latest compatible sample per
object. Before training, expand that audited pool so every acquired object
contributes every reconstructable stage up to its acquisition stage, with the
historical zero-based AstroMiNN limit `alert <= 10`:

```bash
python scripts/materialize_boom_all_early_alert_pool.py \
  --registry configs/delta.toml
```

This command is preflight-only. Review the object count, planned sample count,
samples-per-object distribution, class counts, and the number of objects with
non-contiguous cached stages. Then materialize on a CPU allocation with a
valid `BABAMUL_API_TOKEN`:

```bash
python scripts/materialize_boom_all_early_alert_pool.py \
  --registry configs/delta.toml \
  --materialize
```

The operation is resume-safe. It reuses the 463 already validated latest
samples, retains validated partial earlier-stage samples after an interrupted
BOOM query, and writes `candidate_pool.csv` only after every planned sample is
present. It never uses detections after a reconstructed alert's JD.

The current BOOM audit finds 113 earlier stages whose science cutouts violate
the historical input contract (non-finite pixels or non-63x63 shape). They are
not imputed, padded, or repaired. After the strict run records those failures,
`--finalize-invalid-stages` explicitly certifies their exclusion and writes a
separate rejection table. The resulting pool contains 463 objects and 2,450
model-ready packets.

Finally, on a CPU or GPU allocation, require frozen-model parity for every
object's latest sample:

```bash
python scripts/materialize_boom_all_early_alert_pool.py \
  --registry configs/delta.toml \
  --materialize \
  --run-parity \
  --device cuda
```

Do not start adaptive training unless `parity_summary.json` reports
`"passed": true`, `"mismatches": 0`, and a zero (or tolerance-level) maximum
prediction difference.

### Launch an adaptive suite

The script defaults to preflight-only mode:

```bash
python scripts/run_astrominn_simulated_al.py \
  --registry configs/delta.toml \
  --pool /work/hdd/bcrv/dkourtesis/applecider_adaptive/pools/boom_tns_new_labelled_early10_allavailable_v1 \
  --hidden-labels /projects/bcrv/dkourtesis/applecider_adaptive/runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv \
  --suite-id astrominn_boom_tns_al_v1 \
  --acquisition-size 50 \
  --max-cycles 6 \
  --max-epochs 15 \
  --patience 5 \
  --new-object-weighting equal_object \
  --holdout-fraction 0.2 \
  --holdout-seed 135 \
  --seed 135 \
  --training-initialization warm_start
```

The command validates paths and prints the exact strategy, storage, data,
budget, holdout, and training policy. Add `--run` only after that output has
been reviewed. The three strategies run sequentially to avoid oversubscribing
one GPU.

`fisher` is available through `--strategies` but intentionally excluded from
the default and first policy pilot. After the original three strategies have
completed one clean cycle, it can be launched alone with
`--strategies fisher`, or included in a final comparison with
`--strategies random margin entropy fisher`.

`--holdout-seed` controls only the object IDs in the new-data holdout. Keep it
at 135 for every independent training seed so all runs are compared on exactly
the same objects. `--seed` controls the training and acquisition randomness
and should change between repeated runs.

`--new-object-weighting equal_object` leaves every historical row at unit
weight and gives each newly acquired object a total weight equal to the
historical mean samples per object. Its weight is divided across all of that
object's retrievable alert packets. This keeps all valid early-alert data
without allowing BOOM archival coverage to determine an object's total
training influence.

## Decisions required before the final experiment

The machinery intentionally does not hard-code the following scientific
choices:

- sequential warm-starting versus restarting from the baseline checkpoint;
- epochs/patience and whether 50 newly acquired samples need explicit replay
  or oversampling relative to the 29,161 historical sample rows;
- whether to reserve a fixed new-data holdout in addition to the immutable
  historical test split;
- number of cycles and independent seeds.

The all-stage materialization and fixed-holdout decisions are now explicit.
The remaining policy choice should be made from a one-cycle pilot: inspect the
training/validation loss history, chosen best epoch, historical validation and
test metrics, new-holdout metrics, and wall time before fixing the final epoch
budget and running all strategies.
