# Optional PyTorch SWA checkpoint generation (Package B.1)

This is **opt-in implementation machinery**, not an experimental recommendation
and not enabled in any current AstroMiNN or simulated-active-learning run.
Existing `TrainingRequest`, `TrainingResult`, within-cycle best-loss/F1 retention,
model acceptance, acquisition, data paths, and label budgets are **unchanged**.

## Placement

`al_pipeline.integrations.pytorch.swa` contains the PyTorch-specific adapter
**outside of `core`**. The repository architecture forbids `torch` imports
anywhere under `al_pipeline.core`, even behind lazy imports. This keeps PyTorch
optional for model-independent core users while making the adapter reusable
across astronomy and other domains.

- `SWAConfig(enabled=False, start_epoch=..., update_every=..., swa_lr=...)`:
  configuration, disabled by default.
- `SWATracker(model, config, optimizer=...)`: parameter averaging using PyTorch's
  `AveragedModel` and optional `SWALR` (the latter requires an optimizer).
- `tracker.end_epoch(epoch, base_scheduler=...)`: called once after a completed
  optimizer-training epoch; updates the average at eligible epochs and steps
  **exactly one** scheduler. Do not step the scheduler separately.
- `tracker.save_checkpoint(path, identity=...)`: atomically exports an immutable
  *candidate* with full `CheckpointTrainingIdentity` and SWA provenance.
- `load_swa_checkpoint(...)`: strictly loads its weights and verifies identity.
- `tracker.state_dict()` / `load_state_dict()`: optional continuation of the
  SWA accumulator. A genuine interrupted-training resume also requires the
  originating model, optimizer and base scheduler states from the same epoch.

## Minimal pattern

```python
from al_pipeline.integrations.pytorch.swa import SWAConfig, SWATracker

# After constructing the ordinary training model and optimizer:
tracker = SWATracker(
    model,
    config=SWAConfig(enabled=True, start_epoch=15, update_every=1,
                     swa_lr=1e-4, anneal_epochs=5),
    optimizer=optimizer,
)
for epoch in range(1, max_epochs + 1):
    train_one_epoch(model, optimizer, train_loader)   # normal optimization
    metrics = validate(model, val_loader)            # normal validation
    best_checkpoint_keeper.consider(...)             # independent Package B policy
    tracker.end_epoch(epoch, base_scheduler=base_scheduler)

# Only if averaging actually started (early stopping can precede it):
if tracker.ready:
    tracker.save_checkpoint(
        path_to_new_swa_checkpoint,
        identity=checkpoint_training_identity,
        bn_refresh=backend_specific_bn_refresh_if_needed,
    )
# Evaluate this SWA candidate on FIXED VALIDATION; only afterwards apply
# ordinary checkpoint selection / Package B cross-cycle acceptance, if enabled.
```

When `swa_lr=None`, the base scheduler continues during averaging. When
`swa_lr` is supplied, PyTorch `SWALR` replaces the base scheduler for epochs
at or after `start_epoch`. SWA does **not** extend the training run, change
patience/early stopping, optimizer steps, checkpoint selection or acquisition.
A prematurely stopped run may have zero averages; the backend should simply
omit its SWA candidate.

## BatchNorm is not optional bookkeeping

PyTorch SWA averages parameters, **not reliable BatchNorm running statistics**.
If the model contains BatchNorm, exporting a candidate requires refresh with
representative **training inputs only**, never validation/test or hidden U data.

`tracker.save_checkpoint(..., bn_loader=loader)` uses native PyTorch `update_bn`
for models accepting **one positional tensor**. For multimodal models requiring
multiple tensors (e.g., metadata and images), supply `bn_refresh=callback` that
recomputes BatchNorm buffers on training batches by calling the model with its
correct forward signature. Without a refresh callback/loader, export refuses
any model containing BatchNorm. The callback is called with the unwrapped
averaged `torch.nn.Module`.

No SWA weights are appended to a seed ensemble automatically. SWA is parameter
averaging over *one trajectory*; it is not averaging predictions across seeds.
Its exported checkpoint has to be evaluated before choosing it over best-loss
or best-F1, and its performance is not established merely by creating it.

## Future research (not implemented)

**TODO(SWAG):** Stochastic Weight Averaging-Gaussian needs posterior-moment or
covariance tracking, weight sampling and a separate uncertainty-scoring API.
It should be developed only after the basic SWA/ensemble and calibration
experiments identify whether a Bayesian approximation is worth the additional
compute and statistical assumptions. The matching TODO is in `swa.py`.

## Repository migration from B.1 v1

The previous overlay mistakenly placed `swa.py` under `core/training/`.
After installing the corrected adapter and test, remove
`src/al_pipeline/core/training/swa.py`; otherwise the architecture test
will continue to fail. Do not modify `tests/core/test_architecture.py`.
