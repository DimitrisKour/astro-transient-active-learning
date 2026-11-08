# Teaching data — historical AstroMiNN validation histories

Notebook [02](../02_applecider_astronomy_active_learning.ipynb) is a **read-only, CPU-only investigation** of completed active-learning experiments. It does not train AstroMiNN or run a new acquisition loop. The interactive mechanics of pool-based active learning are taught in notebook 01 (handwritten digits).

## Data included

- `astro_recent_validation.csv` — **1,020 observed validation metric rows**, representing 51 policy/seed trajectories across eight experiment suites. Every trajectory has cycles 0–9, and every cycle has macro-F1 and balanced accuracy. No unmeasured cycles are fabricated.
- `astro_recent_provenance.json` — schema, original suite identifiers, SHA-256 of the CSV, experimental scope and interpretation limitations.

The aggregate-metric export was generated from **completed archived `evaluation_history.json` files**. It contains **only** aggregate validation metrics, policy, seed, cycle, labeled count and cumulative acquisition count. It contains no raw astronomical images, candidate/object identifiers, label oracle responses, checkpoints, test metrics, or credentials. No external download or local HPC configuration is required.

## Setup and run

From the repository root:

```bash
python -m pip install -e ".[teaching]"
python scripts/examples/execute_astro_notebook.py --output astro_school_local.ipynb
```

Or open the notebook interactively in a Jupyter environment. It uses NumPy, pandas, Matplotlib and the optional plotting theme from the repository. The notebook can be restarted and executed completely offline on a CPU.

## How to interpret the comparisons

- The introductory figure compares BADGE, entropy and entropy + diversity. The two entropy variants share a suite; BADGE belongs to a separate suite, so the combined view is **illustrative rather than a controlled policy comparison**.
- The BALD, TAILOR and uniform-mixture figure is a **descriptive cross-suite comparison**: same recorded cycle-zero metrics *by seed* and same label budgets, but the bundle does **not** certify identical initialization, training objective, or checkpoint selection across the source runs.
- An alternate-$L_0$ sensitivity example (Task 17) is deliberately plotted separately because the initial labeled objects differ.
- Mean curves have no error bands to reduce clutter; individual-seed variability is visualized in its own figure. Three seeds are not sufficient for a formal significance claim.
- Per-class and acquired-class conclusions cannot be drawn from the bundled aggregate CSV. Recorded per-class validation scores can be exported separately from saved result files without loading images or checkpoints.


## Optional instructor-only class-metric export

On Delta, the already-computed AstroMiNN object-level validation metrics can be gathered **without rerunning inference**:

```bash
python scripts/examples/export_recent_astro_per_class.py \
  --validation-csv examples/data/astro_recent_validation.csv \
  --artifact-root /work/hdd/bcrv/dkourtesis/applecider_adaptive \
  --output ~/astro_recent_per_class_pilot.zip
```

The exporter checks that all 51 trajectories have class metrics for cycles 0–9, and that per-class F1 averages reproduce the bundled validation macro-F1. It writes only aggregated class-level precision, recall, F1 and support, with a provenance JSON. The output is for review first; inclusion in the student notebook requires a redistribution check and a separate notebook update. It does not include acquired-object labels or identities.

## Included class-level and acquisition-overlap tables

- `astro_recent_per_class.csv` (2,550 rows): precomputed validation precision, recall, F1 and support, with suite, policy, seed and cycle. Metrics were calculated in the original experiments, not by rerunning models in the notebook.
- `astro_cycle2_overlap_by_seed.csv` (45 rows) and `astro_cycle2_overlap_mean.csv` (15 rows): intersection, union and Jaccard overlap for five policies at acquisition cycle 2. The notebook uses the **cumulative union of selected objects from cycles 1 and 2** (946 selected per policy), excluding the initial labeled set; the mean is taken over three corresponding seeds.
- `astro_recent_per_class_provenance.json` and `astro_cycle2_overlap_provenance.json` document the export scope and exclusions.

The blue symmetric Jaccard matrix in notebook 02 is a small, read-only summary, with the diagonal omitted and each off-diagonal cell labeled as a percentage. These tables are suitable for CPU-only student analysis. Cross-suite protocol equivalence is **not established** by matching cycle-zero scores or label budgets alone.

## Source and publication review

The user-provided ZIP `astro_recent_school_pilot.zip` supplied the validation-only CSV and an internal analysis audit. The resulting data file is reproduced in the repository for consistent, offline teaching, while the public-facing provenance JSON omits machine-specific and candidate-level information.

**Current status:** teaching review, not a full cross-suite protocol comparability or publication-rights certification. Before distributing these archived research measurements outside an authorized teaching audience, check rights, attribution and scientific interpretation with the project collaborators. Do not publish the raw historical samples, checkpoints or private manifests by copying these teaching files.

Legacy preparation scripts for the earlier optional image-feature demonstration remain in `scripts/examples/` for research reuse but are **not needed to run notebook 02**.