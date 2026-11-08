# Archived pre–Task 0 training attempts and diagnostics

This directory separates **earlier operational implementations** from the
later adopted full-historical resident-cache workflow.

## Historical stages

| Directory | Status and meaning |
| --- | --- |
| `recipe_v1/` | Original full-historical training-recipe comparison using a sharded cache; retained as initial training baseline, not recommended for new runs |
| `fastcache_v2/` | Subsequent fast memory-mapped cache training variant; retained to explain the performance and storage experiments |
| `early_diagnostics/` | Early cache/first-batch diagnostic jobs, including failed, partial and bounded performance experiments |

The later `fusion_pre_task0_resident.py`, corresponding resident recipe,
`fusion_pre_task0_extensions.py`, full-historical cache source, `task0_full_history.py`
and `task1_closed_loop.py` remain in `scripts/adaptive/` as the reference
research workflow for further organization. Do not treat the original v1/v2
scripts as alternative supported user interfaces.

These archives are **not obsolete science**: original full-data training
recipe options remain distinct, and all their training implementations remain
in `src/al_pipeline/astronomy/fusion_image/` so the numerical tests and
research comparisons can be reproduced. The archived launchers only document
old operational/cache arrangements.

The early L0-only cache and initial L0 matrix live separately under
`scripts/archive/fusion_image/early_l0/`, with their own source contracts.

## Reproducing old work

Original completed or failed Delta experiment snapshots have immutable source
hashes, manifests and paths. **Do not copy archived scripts into an existing
frozen run or resubmit a historical suite in place.** For archival
reproduction, use a distinct suite identity and verify the old pinned data,
original cache format, upstream model sources and cluster-specific paths.

These archived launchers retain Delta-specific assumptions and are **not**
part of the portable package's default tests or recommended examples.

The original numerical validation is retained in
`tests/astronomy/fusion_image/test_pre_task0_recipe.py` and
`test_pre_task0_fastcache_v2.py`. The default library tests remain available
with `python -m pytest -q`; see `docs/testing.md`.
