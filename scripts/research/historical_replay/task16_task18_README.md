# Task16–18: acquisition budget, alternate L0 and AnchorAL

These are **frozen historical research protocols**, not user-facing training
entry points.

| Folder | Scientific comparison |
| --- | --- |
| `task16/` | Acquisition batch fraction (5%, 10% reference, 20%) and pool coverage |
| `task17/` | Random rather than stratified initial labeled set, new L0 bootstrap, and three-/nine-cycle continuation |
| `task18/` | AnchorAL acquisition with frozen Task11 source |

The Task17 `task17_full9.py` controller **must continue loading the original
frozen Task17 source**, `source/scripts/adaptive/task17.py`, and use its
original `source/scripts/adaptive/task17_worker.sbatch` for cycles 1–3. The
new checkout's grouped directory is used only for the separate continuation
controller/worker. These historical references are deliberate, not stale paths.

The Task18 `task18_repair.py` is a provenance-preserving *one-time* repair,
not an alternative acquisition strategy. The old frozen Task18 source and its
pre-repair manifest must not be overwritten by the reorganized checkout.

For a no-Delta validation, use the CI job in
`.github/workflows/task16-task18-packaging.yml`, which tests the scheduler,
scientific metadata, and accounting without private frozen artifacts.
The full source-anchored integration tests remain explicitly opt-in via
`APPLECIDER_TASK11_SOURCE`, as described in `docs/testing.md`.

Previous Delta snapshots retain `source/scripts/adaptive/task16*`,
`task17*`, and `task18*`. New grouped script paths must never be mixed
with previously frozen manifests, active held chains or existing suite IDs.
