# Package C: model-independent advanced acquisition

These methods are **opt-in**. The package only adds
`al_pipeline.core.advanced_acquisition`; it does **not** modify `ActiveLearningLoop`,
existing acquisition strategies, astronomy training, `SimulatedALConfig` or the
historical replay runners. It does not read NPZ files or require PyTorch.

## Three methods

- **BADGE**: pseudo-label gradient embeddings for the final **linear softmax
  classification layer**, followed by seeded k-means++ batch selection.
  The first center maximizes the pseudo-label gradient norm (ties use original
  row order); subsequent centers are sampled in proportion to the squared
  distance to the nearest chosen center. Exact distances use factored class
  residuals and last-layer inputs, without materializing the dense
  `n_candidates × (n_classes × feature_dim)` gradient matrix. This assumes
  cross-entropy-style gradients and a single final affine classification
  layer. If a model uses a different loss or head, provide a separately
  justified gradient adapter. Bias coordinates are included by default.
- **Core-set**: greedy farthest-first k-center covering of candidates against
  **all existing labeled anchors** and newly selected points. It uses raw
  squared Euclidean distances; embeddings must be comparable and extracted
  using the same model and preprocessing configuration. It is not the
  pre-existing `DiversitySelector`, which diversifies within an uncertainty
  shortlist without conditioning on the labeled set.
- **Ensemble disagreement**: weighted Jensen–Shannon score already computed
  by `PredictionEnsemble.disagreement()` from Package A. It can be passed
  directly to `TopKSelector`. This is model disagreement, not automatically a
  calibrated Bayesian posterior mutual-information estimate.

## End-to-end minimal usage

```python
import numpy as np
from al_pipeline.core.advanced_acquisition import (
    BadgeInput, BadgeGradientNormScorer, BadgeSelector,
    CoreSetInput, CoreSetScorer, CoreSetSelector,
    EnsembleDisagreementScorer,
)
from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.pool import Pool
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.types import ClassificationPredictions

# One row per acquisition unit, with stable, explicit IDs.
ids = ("o1", "o2", "o3")
features = np.array([[0., 1.], [1., 0.], [-1., 0.]])
probabilities = ClassificationPredictions(
    np.array([[.5, .5], [.8, .2], [.6, .4]]), "probabilities"
)

badge = BadgeInput(probabilities, features, candidate_ids=ids)
badge_loop = ActiveLearningLoop(
    scorer=BadgeGradientNormScorer(),
    selector=BadgeSelector(badge, seed=135),
    pool=Pool.from_ids(ids),
)
selected = badge_loop.query(ids=list(ids), acquisition_data=badge, k=2)

core_set = CoreSetInput(
    candidate_embeddings=features,
    labeled_embeddings=np.array([[0., 0.]]),
    candidate_ids=ids,
    labeled_ids=("previous_labeled_object",),
)
core_loop = ActiveLearningLoop(
    scorer=CoreSetScorer(),
    selector=CoreSetSelector(core_set),
    pool=Pool.from_ids(ids),
)
selected_core = core_loop.query(ids=list(ids), acquisition_data=core_set, k=2)

# For an existing PredictionEnsemble ensemble from Package A:
# loop = ActiveLearningLoop(
#     scorer=EnsembleDisagreementScorer(),
#     selector=TopKSelector(), pool=Pool.from_ids(ensemble.object_ids),
# )
# selected = loop.query(
#     ids=list(ensemble.object_ids), acquisition_data=ensemble, k=2,
# )
```

`query()` proposes without modifying the pool; `acquire()` also marks IDs pending.
BADGE and core-set selectors validate that the supplied ordered candidate IDs
match the representation IDs, that scoring came from the corresponding scorer,
and that each candidate is currently unlabeled. Duplicate candidates, mismatched
arrays, invalid budgets, NaNs and infinities are rejected. Core-set output scores
are its **dynamic** squared covering distances at the moment of selection;
BADGE output scores are pseudo-label gradient squared norms, while the separate
`selection_d2` metadata records the stepwise k-means++ sampling distances.

## Reproducibility and computational notes

1. Construct a **new** `BadgeSelector(data, seed=...)` and matching scorer for
   each active-learning cycle; choose a cycle-specific deterministic seed and
   record it with the acquired IDs. The RNG is recreated per selection for
   reproducible retries. Do not repeatedly `acquire()` an unchanged pool.
2. The `BadgeInput` must receive the representation **entering the last linear
   classification layer**, not arbitrary ConvNeXt feature vectors. BADGE
   applies pseudo-label gradients (not true hidden candidate labels).
3. In astronomy, combine multiple alert predictions/features into **one row
   per object before acquisition**, using a documented, fixed aggregation
   protocol. Validate object and row order after aggregation; do not make
   alert-level acquisitions accidentally.
4. For core-set, use the **same feature extractor**, normalization, and
   trainable model checkpoint for the candidate and labeled features. Rebuild
   anchors after the representation model changes. Scale high-magnitude
   features explicitly to avoid squared-distance overflow.
5. Core-set's initial anchor distance calculation costs
   `O(n_candidates × n_labeled × feature_dim)`; selecting `k` candidates costs
   an additional `O(k × n_candidates × feature_dim)`. Benchmark/cache features
   before using large candidate pools.
6. Ensemble disagreement assumes the caller passes `ids` in
   `ensemble.object_ids` order; Package A aligns predictions from independent
   members and validates class order. Choose/check the ensemble aggregation
   and calibration rules in Package A separately.
7. Selection algorithms use no oracle responses, held-out test labels, or
   candidate target labels. Test metrics should remain report-only.

## Tests

```bash
python -m pytest tests/core/test_advanced_acquisition.py -q
python -m pytest tests/core -q
```

## Scope deferred until astronomical model validation

- No registration of new strategy names into `SimulatedALConfig` or modified
  historical replay trajectories yet.
- No AstroMiNN-specific feature, gradient, or object-pooling adapter yet.
- No claim that BADGE or core-set improves classification on the scientific
  datasets, or that model disagreement is calibrated epistemic uncertainty.
- Future options: approximate distances for larger pools, alternative
  representation normalization, modality-aware sampling, MC-dropout/SWAG
  uncertainty, compute-budget-controlled acquisition comparisons.

### References

- Ash et al., *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower
  Bounds* (BADGE, ICLR 2020): https://github.com/JordanAsh/badge
- Sener and Savarese, *Active Learning for Convolutional Neural Networks:
  A Core-Set Approach* (ICLR 2018): https://arxiv.org/abs/1708.00489
