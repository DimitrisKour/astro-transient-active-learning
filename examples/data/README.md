# Teaching data — historical AstroMiNN validation histories

Notebook [02](../02_applecider_astronomy_active_learning.ipynb) is a **read-only, CPU-only investigation** of completed active-learning experiments. It does not train AstroMiNN or run a new acquisition loop. The interactive mechanics of pool-based active learning are taught in notebook 01 (handwritten digits).

## Data included

- `astro_recent_validation.csv` — **1,020 observed validation metric rows**, representing 51 policy/seed trajectories across eight experiment suites. Every trajectory has cycles 0–9, and every cycle has macro-F1 and balanced accuracy. No unmeasured cycles are fabricated.
- `astro_recent_provenance.json` — schema, original suite identifiers, SHA-256 of the CSV, experimental scope and interpretation limitations.

The bundled export was generated from **completed archived `evaluation_history.json` files**. It contains **only** aggregate validation metrics, policy, seed, cycle, labeled count and cumulative acquisition count. It contains no raw astronomical images, candidate/object identifiers, label oracle responses, checkpoints, test metrics, or credentials. No external download or local HPC configuration is required.

## Setup and run

From the repository root:

```bash
python -m pip install -e ".[teaching]"
python scripts/examples/execute_astro_notebook.py --output astro_school_local.ipynb
```

Or open the notebook interactively in a Jupyter environment. It uses NumPy, pandas, Matplotlib and the optional plotting theme from the repository. The notebook can be restarted and executed completely offline on a CPU.

## How to interpret the comparisons

- A simple reference comparison (random, entropy, entropy + diversity) comes from **one experiment suite**.
- The BALD, TAILOR and uniform-mixture figure is a **descriptive cross-suite comparison**: same recorded cycle-zero metrics *by seed* and same label budgets, but the bundle does **not** certify identical initialization, training objective, or checkpoint selection across the source runs.
- An alternate-$L_0$ sensitivity example (Task 17) is deliberately plotted separately because the initial labeled objects differ.
- Mean curves have no error bands to reduce clutter; individual-seed variability is visualized in its own figure. Three seeds are not sufficient for a formal significance claim.
- No per-class or acquisition-class conclusions can be drawn from this limited export. We will need a separate, audited and authorized export of those records for any future rare-class teaching extension.

## Source and publication review

The user-provided ZIP `astro_recent_school_pilot.zip` supplied the validation-only CSV and an internal analysis audit. The resulting data file is reproduced in the repository for consistent, offline teaching, while the public-facing provenance JSON omits machine-specific and candidate-level information.

**Current status:** teaching review, not a full cross-suite protocol comparability or publication-rights certification. Before distributing these archived research measurements outside an authorized teaching audience, check rights, attribution and scientific interpretation with the project collaborators. Do not publish the raw historical samples, checkpoints or private manifests by copying these teaching files.

Legacy preparation scripts for the earlier optional image-feature demonstration remain in `scripts/examples/` for research reuse but are **not needed to run notebook 02**.