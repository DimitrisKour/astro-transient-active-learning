# Active Learning for Astronomical Transients

**A Python toolkit for selecting informative transient candidates, managing label acquisition, and evaluating active-learning strategies.** It combines a reusable, model-independent active-learning core with astronomical integrations developed for AppleCiDEr and historical AstroMiNN experiments.

The central question is practical: **which objects should we label next, and how much does each acquisition improve classification?** The code supports both exploratory learning experiments and archived, multi-seed studies of early transient classification.

## What is included?

| Capability | Where to look |
| --- | --- |
| **Pool-based active learning** — uncertainty scoring, selection, budgets, and labeled/unlabeled/pending states | [`al_pipeline.core`](src/al_pipeline/core/) |
| **Label evidence and oracles** — provider-independent evidence collection and resolution contracts | [`core.evidence`, `core.oracle`, `core.workflow`](src/al_pipeline/core/) |
| **Astronomical integrations** — ZTF alert handling, BOOM catalogue access, historical survey manifests, AppleCiDEr taxonomy rules | [`al_pipeline.astronomy`](src/al_pipeline/astronomy/) |
| **Historical AstroMiNN studies** — acquisition policies including random, entropy, diversity, BADGE, BALD, TAILOR, SIMILAR, DIRECT, and AnchorAL | [`research/historical_replay`](scripts/research/historical_replay/README.md) |
| **Evaluation and diagnostics** — validation curves, per-class scores, seed variation, acquisition overlap, and run histories | [`core.evaluation`](src/al_pipeline/core/evaluation/), [teaching results](examples/data/README.md) |
| **Two CPU teaching notebooks** — run an acquisition loop, then analyze real archived astronomy experiments | [Examples](examples/README.md) |

The **core is independent of astronomy and PyTorch**. Historical AstroMiNN studies, by contrast, can require archived datasets, model checkpoints, external code, and HPC resources. The teaching notebooks do not.

## Quick start: notebooks on your own computer

Requires **Python 3.11 or newer**. Clone the repository and install the teaching dependencies:

```bash
git clone https://github.com/DimitrisKour/astro-transient-active-learning.git
cd astro-transient-active-learning
python -m venv .venv
source .venv/bin/activate                 # macOS/Linux
python -m pip install -e ".[teaching]"
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead. The repository is publicly accessible on GitHub.

Both student notebooks **already contain executed figures** and can be opened immediately in Jupyter or VS Code. To run them from scratch on CPU and save separate, executed copies:

```bash
python scripts/examples/execute_digits_notebook.py --mode core --output digits_core_local.ipynb
python scripts/examples/execute_astro_notebook.py --output astro_school_local.ipynb
```

| Notebook | What you do | Data and hardware |
| --- | --- | --- |
| [01 · Pool-based active learning with digits](examples/01_active_learning_digits.ipynb) | Run random, entropy, and margin acquisition; inspect the pool and solve five guided exercises | Bundled scikit-learn digits; CPU |
| [02 · Active learning for astronomical transients](examples/02_applecider_astronomy_active_learning.ipynb) | Analyze completed AstroMiNN learning curves, TDE/CV class metrics, seed variation, and a cumulative acquisition-overlap matrix | Bundled, precomputed research tables; CPU and offline |

The second notebook analyzes **recorded experiments**, not a newly trained astronomical model. Its [data guide](examples/data/README.md) explains the scope of the archived measurements and limitations of cross-suite comparisons.

## Use the core in Python

For example, select the most uncertain object from three candidates, using predictive entropy:

```python
import numpy as np
from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.pool import Pool
from al_pipeline.core.scoring.entropy import EntropyScorer
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.types import ClassificationPredictions

ids = ["candidate-a", "candidate-b", "candidate-c"]
probabilities = ClassificationPredictions(
    values=np.array([[0.95, 0.05], [0.51, 0.49], [0.80, 0.20]]),
    output_type="probabilities",
)
pool = Pool(unlabeled_ids=ids)
loop = ActiveLearningLoop(pool=pool, scorer=EntropyScorer(), selector=TopKSelector())
selection = loop.acquire(ids=ids, acquisition_data=probabilities, k=1)
print(selection.ids)  # ['candidate-b']
```

`acquire()` marks selected candidates as **pending**; labels are handled separately by the oracle/resolution workflow. The core does not need an astronomical dataset or pretrained model to run.

## Installation options

Install from the **repository root** using editable mode:

```bash
python -m pip install -e .                # Minimal core (NumPy)
python -m pip install -e ".[analysis]"     # Pandas, Matplotlib, scikit-learn
python -m pip install -e ".[teaching]"     # Both offline notebooks
python -m pip install -e ".[astronomy]"    # PyTorch/torchvision, Astropy, analysis tools
python -m pip install -e ".[dev,teaching]" # Tests, linting, and notebooks
```

The **GitHub repository name is separate from the Python package name**: the installed distribution remains `al-pipeline`, and imports remain `al_pipeline`. Astronomy extras install dependencies, but **do not download historical datasets, trained checkpoints, private registry files, or upstream model implementations**. For the frozen Delta environment, use the [HPC installation guide](docs/delta_installation.md) instead of casually upgrading its scientific dependencies.

## Repository guide

- [`src/al_pipeline/core/`](src/al_pipeline/core/) — portable acquisition, selection, pool, label, training and evaluation interfaces.
- [`src/al_pipeline/astronomy/`](src/al_pipeline/astronomy/) — broker, ZTF, resolver, AstroMiNN, data and experiment integrations.
- [`examples/`](examples/README.md) — student notebooks and portable CPU runners.
- [`scripts/`](scripts/README.md) — teaching tools, research launchers and archived experiments; not every script is a public quickstart.
- [`docs/`](docs/) — experiment protocols, migration notes and environment guides.

## Testing and reproducibility

```bash
python -m pip install -e ".[dev,teaching]"
python -m pytest -q                                      # Portable default suite
python -m pytest -q tests/examples                       # Notebook and export checks
```

The default suite runs without private astronomy data or GPUs. Historical experiment tests may need external artifacts, optional packages, and specific environments; consult the [testing guide](docs/testing.md) and [research script navigation](scripts/README.md). For implementation provenance, see the [migration notes](docs/migration.md).

**Research-data note:** the included teaching tables contain summarized validation results, not raw survey observations or model weights. Differences across independently executed experiment suites are descriptive unless their full protocols are established as comparable. Review the [data and redistribution notes](examples/data/README.md) before sharing research-derived measurements outside an authorized audience.

## License, authorship, attribution and citation

**Copyright (c) 2026 Dimitris Kourtesis.** The original software and its accompanying documentation are released under the [GNU General Public License v3.0 (GPL-3.0-only)](LICENSE), to the extent that their copyright is held by the stated author. The project is authored and maintained independently; the integrations do not imply ownership of upstream AppleCiDEr, AstroMiNN, FLARE, Hyrax, survey infrastructure, model architectures, or datasets. See [third-party and research-output scope](THIRD_PARTY.md) for distinctions between this repository's original code and separately supplied scientific systems and data.

For research attribution, see [CITATION.cff](CITATION.cff). For publication and Git-history security checks, see the [pre-publication review](docs/publication_review.md). The experimental notebooks and research scripts are provided for reproducibility; they are not claims of equivalence with independently maintained upstream model implementations.
