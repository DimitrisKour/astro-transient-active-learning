# Package B — checkpoint selection and cross-cycle acceptance

Package B implements **two distinct opt-in mechanisms** for active learning, both
under `al_pipeline.core.training`. The package is independent of PyTorch,
AstroMiNN, the astronomical datasets, and SLURM. It adds no new dependencies.

Existing `TrainingRequest`, `TrainingResult`, `RunState`, `Pool`, training backends,
metrics, and experiment defaults are **unchanged**. Installing this overlay does
**not** activate checkpoint policies in any existing runner. Integration into a
specific training backend must be explicit, after that backend is validated.

## 1. Within-cycle metric-specific checkpoint retention

Module: `al_pipeline.core.training.checkpoint_selection`.

- `MetricCriterion(name, direction, evaluation_level)` describes a validation
  metric. Directions are `min` or `max`; equal values do *not* replace the
  previous best.
- Default criteria are `validation_loss` (min, **sample-level**) and
  `validation_macro_f1` (max, **object-level**). They may be independently
  changed to match the trainer's *actual* validation metrics.
- `CheckpointTrainingIdentity` freezes run/cycle/seed, labeled-set version,
  validation dataset/protocol fingerprint, evaluation level, and initialization
  checkpoint. Use `evaluation_level="mixed"` when retaining different granularities.
- `WithinCycleCheckpointKeeper.consider()` receives an epoch's **validation**
  metrics and a backend-specific checkpoint writer callback. The writer receives
  a temporary destination path. If either best metric improves, the keeper saves
  **exactly one** epoch artifact even if *both* improve. The independent loss/F1
  selections may reference the same artifact when their best epochs coincide.
- A JSON manifest records metric-wise best epoch, value, checkpoint path and
  training identity. Once the manifest is atomically published, superseded
  artifacts are removed when no longer referenced by any criterion. Removal
  failure does not invalidate published selections.
- Failed writes do not advance the recorded epoch or replace previous selected
  checkpoints. Use `resume=True` to validate and reopen a prior manifest.
  Crashes between artifact creation and manifest publication may leave orphan
  artifacts; the keeper fails closed rather than overwriting them silently.
- The model backend controls serialization and must be responsible for creating
  a *complete* checkpoint (weights and any necessary resume metadata). The
  keeper controls naming, temporary files, promotion, and selection metadata.
  Backend callback writes once to the supplied path, without renaming it.
- Single writer per fit only. Parallel SLURM tasks must use different fit dirs;
  no inter-process locking is provided by this component.

Illustrative adapter usage (not enabled automatically):

```python
import torch
from al_pipeline.core.training.checkpoint_selection import (
    CheckpointTrainingIdentity, WithinCycleCheckpointKeeper,
)

keeper = WithinCycleCheckpointKeeper(
    "/projects/.../fit_135/checkpoint_policy",
    identity=CheckpointTrainingIdentity(
        run_id="baseline_seed135", cycle=0, seed=135,
        labeled_set_version="L0:<manifest_sha256>",
        validation_identity="validation:<manifest_sha256>:<evaluation_protocol>",
        evaluation_level="mixed", initial_checkpoint=None,
    ),
)

# Called after every completed validation epoch, within the training loop.
keeper.consider(
    epoch=epoch,
    validation_metrics={
        "validation_loss": float(weighted_sample_validation_loss),
        "validation_macro_f1": float(object_level_validation_macro_f1),
    },
    save_checkpoint=lambda destination: torch.save(model.state_dict(), destination),
)

loss_best_path = keeper.selected("validation_loss").path
f1_best_path = keeper.selected("validation_macro_f1").path
```

The example demonstrates selection only. In production, checkpoint serialization
should preserve the full model spec, training and data identity, and relevant
reproducibility metadata. Training early-stopping selection is a **separate**
decision and must be explicitly specified by the backend.

## 2. Cross-cycle checkpoint acceptance

Module: `al_pipeline.core.training.checkpoint_acceptance`.

- `EvaluatedCheckpoint` records checkpoint path, source training cycle,
  labeled-set version, seed/run, initialization checkpoint, selection rule,
  ordered class names, fixed validation identity, evaluation level and metrics.
  `evaluation_split="test"` is forbidden.
- Comparisons require identical run/seed, class order, validation identity,
  evaluation level and explicit metric direction. The candidate must be from a
  later training cycle **with its own immutable checkpoint file path**.
- `CheckpointAcceptancePolicy` provides:

| Mode | Decision when a prior accepted model exists |
| --- | --- |
| `always_latest` | Accept the latest candidate, regardless of score (current default) |
| `keep_best` | Accept only a strict validation-metric improvement; ties retain the incumbent |
| `tolerate_regression` | Accept if metric deterioration is no greater than the nonnegative tolerance |
| `require_improvement` | Accept only if improvement is greater than the tolerance |

The metric can be validation loss (`min`) or macro-F1 (`max`), and the level
must be consistent (sample/object). A tolerance is measured in **native metric
units**, not relative percentage. This is *not* a statistical significance test.

- `decide_checkpoint_acceptance()` is a pure function, returning an explicit
  reason, accepted candidate or incumbent, and signed improvement (positive
  means better).
- `CheckpointAcceptanceLedger` persistently records **one candidate per cycle**
  and the accepted checkpoint, using atomic JSON replacement and idempotent
  exact retries. It requires sequential cycles and rejects policy changes on
  resume. Register the baseline first; later candidates are compared against
  the last **accepted** checkpoint, not automatically the previous cycle's
  rejected candidate.
- The ledger **does not change** the labeled dataset, AL pool, `RunState`,
  optimizer, or training artifacts. Rejecting a new model means only that a
  previous model is selected for future inference/acquisition; labels remain
  acquired and the next training strategy needs an explicit initialization rule.
- Do not overwrite old accepted weights, and verify checkpoint availability
  before deploying the returned artifact. The ledger does not read model files.

Example:

```python
from al_pipeline.core.training.checkpoint_selection import MetricCriterion
from al_pipeline.core.training.checkpoint_acceptance import (
    CheckpointAcceptanceLedger, CheckpointAcceptancePolicy,
    EvaluatedCheckpoint,
)

ledger = CheckpointAcceptanceLedger(
    "/projects/.../checkpoint_acceptance.json",
    policy=CheckpointAcceptancePolicy(
        mode="keep_best",
        criterion=MetricCriterion("validation_macro_f1", "max", "object"),
    ),
)

# Register fully evaluated L0 first. In cycle 1 and beyond, record new candidate.
# Each object is an EvaluatedCheckpoint with complete fixed-validation provenance.
ledger.record(l0_checkpoint)
result = ledger.record(cycle1_candidate)
checkpoint_for_next_acquisition = result.accepted.checkpoint_path
```

## Scientific and operational guardrails

1. **No test-driven choices.** Loss/F1 and acceptance criteria come from the
   validation protocol; test predictions remain report-only.
2. **Comparable populations.** Reuse the fixed validation identity, metric
   definition, class mapping and aggregation level across compared cycles.
   Beware repeated, adaptive optimization of many policies on the same
   validation split.
3. **No rollback of labels.** A rejected checkpoint does not change `L_t`.
   The accepted model may have been trained on an older labeled-set version.
4. **Separation of concerns.** Epoch selection, early stopping, cross-cycle
   acceptance, and training initialization are independent choices.
5. **No silent historical changes.** Existing `TrainingResult.checkpoint_path`
   and warm-start logic retain their original meaning until a concrete adapter
   is changed intentionally and separately, with regression tests.
6. **No simultaneous writers.** Use unique fit directories and a single writer
   for each manifest/ledger. Concurrent coordination/locking is future work.
7. **Restart checks.** Model files must be validated against their own metadata
   and hashes when loaded. These helpers persist decisions, not complete
   model-state integrity verification.
8. **Ensembles are future policies.** Package A can combine checkpoints for
   inference, but B does not introduce ensemble membership or automatic
   replacement rules.

## Tests

```bash
python -m pytest tests/core/training/test_checkpoint_selection.py \
  tests/core/training/test_checkpoint_acceptance.py -q
python -m pytest tests/core -q
```

The synthetic tests require no astronomical files or GPUs. They check the
separation of loss-best/F1-best checkpoints, crash-resistant save behavior,
resume validation, metric direction/ties, validation identity, label provenance,
immutable checkpoint paths, acceptance tolerances, rejected cycles, idempotency,
and persistent policy integrity.
