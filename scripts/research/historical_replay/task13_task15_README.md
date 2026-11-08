# Task13–15: advanced acquisition experiments

These studies are frozen-source historical reproductions, **not public package
entry points**. Task13 tested TAILOR against uniform meta-acquisition; Task14
tested SIMILAR variants; Task15 tested DIRECT/VReduce.

- `task13/task13.py` manages the initial four-cycle comparison.
- `task13/task13_extension.py` extends its **already frozen** historical
  trajectories; its reference to the original
  `source/scripts/adaptive/task13.py` is deliberately retained.
- `task14/task14.py` and `task15/task15.py` start from verified, immutable
  Task11 sources. Their source-overlay directories are newly grouped, but the
  **original Task11 frozen path** remains `scripts/adaptive/` within old
  frozen snapshots. Do not rewrite that parent source.
- Each manager owns its science-specific patch and Slurm worker.

The standard, data-free contract checks are in
`.github/workflows/task13-task15-packaging.yml`. A few full integration tests
use the explicit `task11_frozen_source` fixture and remain in the opt-in
historical suite. See `docs/testing.md` for configuring that fixture.

Previously prepared frozen Delta sources, manifests and job submissions retain
their old locations. Do not copy relocated managers into an existing frozen
suite or mix new source hashes into an active experiment. For new reproduction
runs, choose a fresh suite identity after reviewing the original scientific
protocol and resource requirements.
