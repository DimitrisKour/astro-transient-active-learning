# Core prediction utilities: temperature scaling and ensembles

This module is an **opt-in**, model-independent layer. It does not modify the
active-learning loop, trainer, acquisition scorers, or astronomy-specific model.
No scientific inference or data-processing experiment is launched by installing it.

## Existing contracts reused

- `al_pipeline.core.types.ClassificationPredictions`: typed logits/probabilities.
- `al_pipeline.core.evaluation.metrics.compute_classification_metrics`: NLL,
  multiclass Brier, ECE, macro-F1, etc. No duplicate metrics implementations.
- Any model adapter can provide ordered predictions, sample IDs, and class labels.

Only NumPy is needed for calibration and aggregation. `PredictionEnsemble.evaluate`
uses the existing scikit-learn-based evaluation module.

## Example: fit a held-out calibrator

```python
from al_pipeline.core.types import ClassificationPredictions
from al_pipeline.core.prediction import TemperatureScaler

classes = ("AGN", "CV", "SN I", "SN II", "TDE")
validation_logits = ClassificationPredictions(val_logits, "logits")
scaler = TemperatureScaler.fit(
    validation_logits, val_object_labels,
    class_labels=classes, split="validation",
)
scaler.save("run/calibration.json")
scaler = TemperatureScaler.load("run/calibration.json")
calibrated = scaler.transform(
    ClassificationPredictions(candidate_logits, "logits"),
    class_labels=classes,
)
# Explicit opt-in: pass calibrated to your existing entropy scorer.
```

Do not fit on the test set, training labels, or a candidate pool's hidden labels.
Use an independent calibration holdout or a principled validation subdivision
if model/checkpoint selection also depends on the same validation set. Fit a
new calibrator for each *distinct* model checkpoint; do not reuse a temperature
across updated models without evaluating the implications. Temperature scaling
usually preserves argmax class, but can change cross-object entropy rankings.
`fit` minimizes **unweighted** multiclass NLL on held-out labels. It never
optimizes macro-F1 or the downstream acquisition objective.

When passed probabilities, `transform` applies softmax(log(p)/T): this is
explicit **post-ensemble** calibration, *not* the same as calibrating every
member's logits before averaging. Exact zero probabilities remain zero.

## Example: ensemble aligned predictions

```python
from al_pipeline.core.prediction import EnsembleMember, PredictionEnsemble
from al_pipeline.core.types import ClassificationPredictions

members = [
    EnsembleMember(
        "seed-135", tuple(ids_135), classes,
        ClassificationPredictions(logits_135, "logits"),
        checkpoint_ref="runs/seed-135/best_loss.pt",
    ),
    EnsembleMember(
        "seed-246", tuple(ids_246), classes,
        ClassificationPredictions(logits_246, "logits"),
        checkpoint_ref="runs/seed-246/best_loss.pt",
    ),
]
ensemble = PredictionEnsemble(members, aggregation="probability_mean")
ensemble_probs = ensemble.predict()
ensemble_entropy = ensemble.predictive_entropy()
disagreement = ensemble.disagreement()
report = ensemble.evaluate(truth_in_ensemble_object_id_order)
ensemble.save_definition("run/ensemble_definition.json")
```

Members may arrive in different sample or class-column orders; the constructor
reorders them by identifiers and rejects missing/extra objects or classes.
`object_ids` are *acquisition units*. Astronomical adapters must first combine
multiple alerts into one object-level prediction by a documented rule; this
module does not silently average alert rows.

`probability_mean` averages member probabilities. `logit_mean` takes the
softmax of weighted mean raw logits and requires all members to supply logits.
These aggregation rules are not interchangeable. Both support nonnegative,
normalized member weights. `disagreement()` is the weighted Jensen-Shannon
entropy difference **using the probability mean**, regardless of the
configured predictive aggregation. This is a practical deep-ensemble
*BALD-like* disagreement score, not a claim of Bayesian posterior sampling.

`save_definition` writes lightweight provenance metadata, **not** model
weights or prediction arrays. Reproduction additionally requires preserved
checkpoints, model preprocessing and input data identities. Model training,
checkpoint selection, ensemble membership policies, and integration into an
AL cycle are reserved for Package B/later integration and are not enabled by
this package.

## Validation scope

Unit tests use synthetic logits and object identifiers. No `/work/hdd` files,
NCSA SLURM jobs, or astronomical checkpoints are needed. Scientific validation
of calibration and ensemble acquisition on historical data will follow after
training is reliable.
