# Fusion-image historical workflows

This directory groups the **current reference research pipeline** for the
standalone five-class AstroMiNN BTSbot/ConvNeXt image+metadata model.
It is distinct from the old frozen-expert ViT historical replay in
`scripts/research/historical_replay/`, and it is not yet a zero-configuration
public library quickstart.

## Scientific progression

1. `cache/` — full historical TRAIN and fixed VALIDATION stage cache,
   buffer/mmap validation, and CPU-only HDD processing. The cache contains
   preprocessed inputs but no class labels.
2. `recipe/` — full-historical resident-memory `cosine_aug` recipe,
   independently tested alternatives and calibration/extension fits. This is
   the later reference training path; earlier sharded-cache v1 and fastcache v2
   launchers are archived in `scripts/archive/fusion_image/`.
3. **Task 0** (`task0/`) —
   compare GalaxyZoo-pretrained and randomized ConvNeXt initialization with
   the same architecture, identical full TRAIN/VALIDATION splits.
4. **Task 1** (`task1/`) — historical active
   learning, distinct L0/restart/warm-start configurations and object-level
   pool acquisition; hold TEST separate from training and policy selection.

The recipe extensions are **scientific controls**, not interchangeable
replacements for the adopted baseline. Their weights/learning schedules have
different definitions and are retained.

## What is and is not publicly supported

All scripts here assume explicitly configured historical manifests, a pinned
external `fusion_model_basics` implementation, model/pretraining assets,
and (for actual training) CUDA + Slurm. Many original launchers still have
Delta-specific path defaults, which will be moved into public configuration
and examples separately.

The source modules are in `src/al_pipeline/astronomy/fusion_image/`. The
general active learning APIs remain in `src/al_pipeline/core/`; neither is
modified by this script grouping.

For portable tests, run `python -m pip install -e ".[dev]"` then
`python -m pytest -q`. Research-specific cache and recipe synthetic tests
are selected explicitly; the new GitHub Actions workflow checks those without
private source files or SLURM jobs.

## Safety

All original frozen experiment suites preserve their source SHA-256 files and
the old `scripts/adaptive/` paths. A relocated script has a **different
path and potentially a different fingerprint**. Never copy grouped scripts
into a frozen run or reuse an old plan/suite ID; choose a new frozen source
identity for any new research reproduction. Validation of paths and tests
does not constitute a successful end-to-end Delta experiment.

Historic launchers remain in Git history, and earlier experiments and
failed cache constructions are described in
`scripts/archive/fusion_image/README.md`. Do not run Slurm as a cleanup test.

## Live-checkout compatibility

Task 0 and Task 1 have canonical copies under `task0/` and `task1/`,
with separate new suite IDs. The 14 original `scripts/adaptive/` paths are
unchanged for previously queued jobs and frozen source identities. See
[Task 0/Task 1 compatibility](task0_task1_README.md) before using either
version. Do not mix original frozen experiments with new grouped scripts.
