# Task 9B v2 — SWA with BatchNorm statistics refresh

## Why this version exists

The six Task 9B v1 GPU jobs exited before the first epoch with `ValueError: ViT contains BatchNorm; SWA would require a separate BN refresh pass`. V1's guard correctly prevented an invalid SWA checkpoint. **Do not modify the frozen v1 source or delete its failure logs.** This v2 version has a separate experiment directory and versioned additive source files.

## Frozen scientific contract

- Six original Task 8B/Task 4 first rejection cases (same `L_t`, checkpoint hashes, and source training settings).
- Start from each **original rejected candidate** (not the incumbent). Ten extra epochs, freshly initialized AdamW, constant LR `2e-5`, no validation stopping or scheduler, Task4 `.85` manifest-based class weighting and `per_sample` weighting.
- SWA parameters are the equally weighted mean of epochs 6,7,8,9,10. `AveragedModel(..., use_buffers=False)` prevents averaging BatchNorm running statistics.
- Save **final nonaveraged epoch-10** checkpoint as control before changing ViT weights.
- Load SWA parameters into the ViT, reset its BatchNorm running statistics and update them with sample-count-weighted batch momentum, then pass **every sample in the original training manifest exactly once**, deterministic order, `augment=False`, with all non-BN layers in evaluation mode. Frozen expert towers stay in eval mode, no gradients or labels used. Single-sample last batches are prevented by deterministic batch-size adjustment. The saved SWA checkpoint includes refreshed BN buffers; `bn_refresh.json` records sample and batch counts.
- Evaluate validation only, comparing final nonaveraged, SWA, original incumbent and rejected. Do not use test. Repeated validation comparisons remain exploratory.
- `continuation_state.pth` is written atomically after each epoch, with optimizer/model/loader and RNG states for explicit resume after allocation failures.

## Install (additive, no overwritten Task 9B v1 scientific files)

```
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
conda activate al-pipeline
unzip -n /projects/bcrv/dkourtesis/task9b_swa_bn_refresh_v2.zip -d .
python -m pytest tests/jobs/test_task9b_swa_bn_v2.py -q
```

## Frozen preparation (no GPU submission)

```
python scripts/research/historical_replay/task9/task9b_swa_bn_v2.py plan
python scripts/research/historical_replay/task9/task9b_swa_bn_v2.py prepare
python scripts/research/historical_replay/task9/task9b_swa_bn_v2.py verify
python scripts/research/historical_replay/task9/task9b_swa_bn_v2.py status
```

The new root is `/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task9b_swa_continuations_bn_v2`. Never point v2 to the failed v1 root.

## Launch one case first

```
python scripts/research/historical_replay/task9/task9b_swa_bn_v2.py submit --indices 0 --walltime 00:20:00
squeue -u "$USER"
python scripts/research/historical_replay/task9/task9b_swa_bn_v2.py status
```

Check `sacct`, logs and `fits/entropy_balanced_explore_s135/training/bn_refresh.json` before submitting the five remaining cases. Case 0 is balanced-exploration seed 135, case 3 is entropy seed 135.

```
python scripts/research/historical_replay/task9/task9b_swa_bn_v2.py submit --indices 1,2,3,4,5 --walltime 00:20:00
python scripts/research/historical_replay/task9/task9b_swa_bn_v2.py status
python scripts/research/historical_replay/task9/task9b_swa_report_bn_v2.py
```

If case 0 fails, do not submit the others; inspect error logs first. Submission markers forbid duplicates. Explicit recovery must use the frozen worker and the correct case index, retaining existing per-epoch restart state.

## Scientific limitations

This is a post-hoc ten-epoch SWA continuation, **not** SWA integrated over the original Task4 training run. BN refresh is an additional train-manifest-only forward pass and its runtime/memory should be measured. The procedure freezes BN-affecting non-BN layers in eval mode during refresh; its statistics reflect unaugmented training data, which is explicitly recorded. Do not compare against v1 (no v1 SWA checkpoints exist).
