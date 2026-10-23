# AstroMiNN: exploratory scalar temperature audit v1

This overlay contains only new files. It does not alter the ongoing closed-loop
runs, training, checkpoint policies, acquisition scores, or scheduler scripts.
Run **after** the corresponding cycle has finished and its immutable evaluation
and pool inference snapshots have been saved. No GPU is needed to analyze them.

The existing evaluator saves object-level latest-alert **raw logits** in
`<evaluation_dir>/validation_predictions.npz` (`object_ids`, `object_targets`,
`object_values`). The closed-loop engine saves object-level candidate logits in
`<cycle>/inference.npz` (`object_ids`, `values`, `output_type`).

This report fits scalar temperature by unweighted NLL on a deterministic
stratified 50% subset of validation objects, then measures NLL, ECE, Brier,
macro-F1 and other metrics on the other 50%. A matching pool prediction can be
used to compute a label-free counterfactual top-473 entropy-acquisition overlap.
No test or candidate labels are used. The evaluated checkpoint is matched to
pool-inference checkpoint by `evaluation_summary.json` and `cycle_summary.json`.

**Scientific limitation:** AstroMiNN historically chose its best checkpoint
using the original full validation set. Thus the calibration/audit partition
is disjoint *for the temperature fit*, but is **not** an independently untouched
validation set for checkpoint selection. Interpret this as exploratory evidence.
A more rigorous future experiment should use a calibration fold reserved before
checkpoint/early-stopping decisions, and evaluate on a distinct selection/audit
population. This tool does not claim to validate final closed-loop gains.

## Install and CPU tests on Delta

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/temperature_scaling_diagnostic_v1.zip -d .
source ~/al_delta_setup.sh
python -m pytest tests/astronomy/test_temperature_scaling_diagnostic_v1.py -q
```

## Example: original seed-135 L0 baseline and matching Task-4 cycle-1 pool

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
P=/projects/bcrv/dkourtesis/applecider_adaptive
D=/work/hdd/bcrv/dkourtesis/applecider_adaptive
S=astrominn_closed_loop_warm_v1
python scripts/adaptive/temperature_scaling_diagnostic_v1.py \
  --evaluation-dir "$D/runs/$S/entropy/seed_135/cycle_000/evaluation" \
  --pool-inference "$D/runs/$S/entropy/seed_135/cycle_001/inference.npz" \
  --cycle-summary "$P/runs/$S/entropy/seed_135/cycle_001/cycle_summary.json" \
  --budget 473 --seed 135 \
  --output "$P/temperature_scaling/offline_v1/seed135_entropy_cycle1.json"
```

For validation-only calibration diagnostics, omit `--pool-inference` and
`--cycle-summary`; only `--evaluation-dir` and `--output` are required. The
script refuses mismatched model checkpoints or duplicate object IDs. Report
output is idempotent and uses a separately named directory, not the AL output.

## Further work (not implemented yet)

- Prefer true held-out calibration data beyond validation used for checkpoint selection.
- Repeat over seeds and acquisition cycles, predefine fit protocol and compare
  raw/calibrated validation audit metrics without test-driven selection.
- Evaluate entropy rankings and pool composition, then frozen-protocol
  closed-loop calibration pilot after initialization choice is settled.
- Measure whether improvements in NLL/ECE correlate with improved label efficiency.
