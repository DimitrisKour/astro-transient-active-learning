# AstroMiNN ViT-135 training reconstruction

This document records the provenance and behavior recovered for the frozen
five-class AstroMiNN ViT used by the adaptive-learning experiments.

## Provenance

- W&B run: `84r1x3gs`
- Started: 2026-01-22 19:34:33 UTC
- Command:
  `python guest_trainer.py -t vit_tower -ua True -a 10 -debug True`
- GPU: one NVIDIA A40 (48 GB)
- PyTorch: 2.5.1
- Python: 3.10.17
- Model seed recorded by the experiment: 135
- Loader/split seed: 125
- Runtime: 510 seconds
- Completed epochs: 29, including the early-stopping epoch
- Best validation-loss epoch: 14
- Saved classifier: `vit_tower_135.pth`

The supplied source matches the file names and traceback line numbers in the
W&B output. The frozen inference artifacts also include the three expert
tower checkpoints and the exact JSON configuration.

## Recovered optimization procedure

Only `vit_tower` was optimized. The light-curve, spatial, and coordinate
expert towers were loaded from seed-135 checkpoints and parameter-frozen.

- Training samples: 29,161
- Validation samples: 14,148
- Test samples: 14,179
- Maximum alert index: 10
- Training batch size: 512
- Validation/test batch size: 2,048
- Optimizer: AdamW
- Learning rate: `1e-4`
- Betas: `(0.99, 0.99)`
- Epsilon: `5e-10`
- Weight decay: `0.05`
- Gradient clipping norm: `9`
- Scheduler: ReduceLROnPlateau
- Scheduler patience: 5
- Scheduler factor: 0.4
- Minimum learning rate: `5e-10`
- Maximum epochs: 1,000
- Early-stopping patience: 15
- Checkpoint criterion: minimum weighted validation loss

The loss was cross-entropy with one-hot float targets. Preserving float
targets matters because PyTorch's mean reduction normalizes weighted float
targets differently from integer class indices.

The class-weight reference counts were computed over every compatible sample
in train, validation, and test, not only the training split:

| Output class | Samples |
|---|---:|
| AGN | 5,678 |
| TDE | 239 |
| SN I | 40,228 |
| SN II | 8,968 |
| CV | 2,375 |

For count `n`, the weight was `5000 / n**0.85`.

## Historical compatibility details

Two behaviors are preserved only through
`AstroMiNNViTTrainingConfig.legacy_seed135()`:

1. The same random image augmentation was applied to training, validation,
   and test samples. The validation AUPRC and validation loss were calculated
   in separate passes, so they used different augmented views.
2. Calling `model.train()` left the parameter-frozen expert towers in
   training mode. Their dropout remained active and their BatchNorm buffers
   could change, even though their parameters had no gradients.

The rotation augmentation also applied a 90-degree rotation immediately
followed by a 270-degree rotation, producing a full 360-degree rotation. It
is retained in compatibility mode rather than silently reinterpreted.

Finally, `get_dataset(..., seed=125)` reset Python, NumPy, and PyTorch after
the nominal model seed 135 was set. The random ViT was therefore effectively
initialized from the seed-125 PyTorch state. The compatibility policy records
this explicitly.

## Adaptive fine-tuning behavior

Normal adaptive fine-tuning should start from `vit_tower_135.pth`, not a
random ViT. It should retain the original preprocessing and label order while
using the cleaner defaults:

- augment training samples only;
- keep frozen expert towers in evaluation mode;
- cache deterministic preprocessing in memory rather than reopening every
  `.npy` file on every epoch;
- reset optimizer and scheduler for each cycle;
- save the best ViT checkpoint, full last-state checkpoint, epoch history,
  manifest hashes, runtime, and peak CUDA memory.

The learning rate, number of epochs, replay policy, and any later unfreezing
policy remain experiment configuration choices. They should not be hidden in
the generic active-learning coordinator.

## Verification

Run
`notebooks/boom/historical_2025/14_astrominn_vit135_training_reproduction_probe.ipynb`
for a three-epoch compatibility probe. The historical first three epochs were:

| Epoch | Training loss | Validation loss | Validation macro AUPRC |
|---:|---:|---:|---:|
| 1 | 1.8936 | 1.269 | 0.583846 |
| 2 | 1.2371 | 1.047 | 0.617959 |
| 3 | 1.0743 | 0.969 | 0.624022 |

Exact equality is not expected across PyTorch versions, worker scheduling,
and filesystem ordering. The loss/AUPRC scale and direction should agree
before the adaptive fine-tuning coordinator is enabled.
