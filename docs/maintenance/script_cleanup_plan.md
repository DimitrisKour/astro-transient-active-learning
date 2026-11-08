# Research-script audit and staged public-interface cleanup

Status: **inventory completed; no script moved or deleted**. Audited against
`main` on 2026-11-08. Each of the 221 tracked script files has a row in
[script_inventory.csv](script_inventory.csv). The CSV classifications are
**provisional** and should not be treated as proof that scripts are obsolete.

## What was established

- 221 tracked items under `scripts/`: 17 top-level launchers, 201 under
  `scripts/adaptive/`, and three in an existing archived directory.
- Almost everything under `scripts/adaptive/` is a **research experiment
  implementation**, rather than a general AL command-line API.
- At least two plausible public demonstration entry points presently exist:
  `scripts/run_astrominn_simulated_al.py` and
  `scripts/run_astrominn_historical_replay.py`. Neither is yet certified as
  a zero-private-data quickstart; the upcoming example notebooks should select
  or replace the public workflow.
- Many controllers construct **frozen source copies** and verify filenames,
  hashes, and relative paths. Their stage workers often call
  `source/scripts/adaptive/<name>` explicitly. A simple rename breaks those
  contracts and may invalidate reproducibility.
- The frozen-source patch modules are used to create experiment variations;
  they are not all one-off mistakes. For example, `task11.py` imports
  `task11_patch.py`, and Tasks 13-18 have analogous setup dependencies.
- `task17_full9.py` extends rather than replaces the first three Task 17
  cycles. `task13_extension.py` likewise continues an existing trajectory.
- `task11_repair_import.py` and `task18_repair.py` appear to be scoped,
  auditable one-time recovery utilities. Preserve them until reproduction
  manifests and any operating jobs are accounted for.
- Fusion `fusion_pre_task0.py`, `fusion_pre_task0_fast.py`,
  `fusion_pre_task0_resident.py`, `fusion_pre_task0_extensions.py` select
  distinct training modules and therefore cannot be collapsed on name alone.
  The full-historical and L0 caches have different dataset scopes.
- Existing tests hardcode script locations in places, including
  `tests/jobs/test_task4_package.py`,
  `tests/jobs/test_task5_restart.py`,
  `tests/jobs/test_task8c2_schedule.py`,
  `tests/jobs/test_balanced357_strict_retry.py` and
  `tests/astronomy/fusion_image/test_pre_task0_recipe.py`.
  These must be changed and run alongside any move.

## Proposed public/research split (subject to notebook planning)

Keep the public-root interface small:

```text
scripts/
  examples/                 # small, documented synthetic/portable launchers
  experiments/
    astrominn/               # source-level historical and simulated loops
    acquisition_policies/    # supported AL comparison workflows
  research/                  # non-default experiments, documented by study
    historical_replay/
    checkpoint_retention/
    acquisition/
    representation/
    fusion_image/
  hpc/                       # HPC usage guidance and reusable Slurm templates
  archive/                   # one-time repair/prototype provenance
```

**Important:** this is conceptual, not a move order. For research code, keep
workers and setup overlays **beside their owning experiment family**, rather
than gathering all workers into `hpc/`. The top-level `hpc/` should only
hold reusable generic tools and documentation.

A future second *private* repository is justified only if the public-history
review identifies sensitive or collaborator-specific material that must remain
unpublished, or if keeping the research controllers in `al-pipeline` makes
the documented public API unacceptably confusing. A new directory layout alone
is not a compelling reason to split repositories.

## Migration order and gates

### Gate 0: do not affect active Delta runs

Before any path-changing commit, record all open/pending Slurm jobs and frozen
suite source hashes, including balanced357 row-order and seed-retry V2,
Task 13 extensions, Task 17, and any fusion cache/training suites. A Git rename
can be safe for jobs already executing from frozen copies, but only if they do
not read live checkout paths later. **Check, never assume.**

### Gate 1: identify a supported public entry point

With the training-school notebooks, choose one fully local synthetic example
with explicit input/output paths and one astronomy-backed example. Add a
zero-cluster smoke test to CI. This must not depend on BOOM credentials, hidden
labels, historical model weights or Delta paths.

### Gate 2: group a single experiment family at a time

For each family:
1. Find every caller in scripts, docs, tests, Slurm workers and frozen builders.
2. State scientific purpose, active/superseded status, and required inputs.
3. Keep the controller, workers, patches, reports, and manifests together.
4. Rewrite relative-path lookups and overlay manifests with explicit tests.
5. Run portable tests, family tests, `plan`/`verify` tests (without
   submission), and a frozen snapshot compatibility check.
6. Only then migrate the next family.

Recommended pilot family: **Task 4/Task 5 historical replay**. It includes
paired warm-start/baseline-restart workflows and good existing package tests.
Do not choose currently running balanced357 V2 as the first migration.

### Gate 3: archive only with provenance

Archive one-time repairs and superseded launchers in **versioned, study-named
folders** with a short README identifying their original experiment,
upstream base/commit, scientific status, and replacement (if any). If a script
is required by a frozen builder, retain its location or adapt the builder
under test. Avoid unexplained `old.py`/`new.py` renames.

### Gate 4: public hygiene and publication decision

- Replace Delta-owned defaults in installable Python modules and examples.
- Perform a credential, collaborator-data and absolute-path scan **including
  Git history**; merely deleting a tracked file does not remove its content
  from public history.
- Review license, external model/data dependencies and attribution.
- Remove stale worktree/branch instructions from public-facing README files.
- Validate a fresh clone on supported Python versions and keep historical
  experiments opt-in.

## Next action

**Do not reorganize 221 files at once.** First finalize the script
classification with the active-run inventory, then pilot the path migration
of Task 4/Task 5 (or introduce a portable examples entry point first if the
training-school notebooks need it immediately). The test workflow introduced
in `docs/testing.md` is a release gate throughout.

No scripts, research data, or active frozen snapshots were modified during
this inventory step.
