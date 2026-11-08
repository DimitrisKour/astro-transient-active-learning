# Training-school notebooks

## 01 — Active learning with handwritten digits

**[Open the complete notebook](01_active_learning_digits.ipynb)**

This is a guided, laptop-friendly, **CPU-only** introduction to pool-based
active learning using scikit-learn's built-in `load_digits()` dataset.
It requires **no network downloads, astronomy data, pretrained weights,
GPUs, or broker access**.

### Learning objectives

- Distinguish the small labeled set `L0`, eligible unlabeled pool `U0`,
  pending labels, a fixed validation set and a sealed test set.
- Inspect class-probability vectors and predictive entropy, then compare
  predictive entropy with top-two margin uncertainty.
- Use the **actual** `al_pipeline.core` `Pool`, `ActiveLearningLoop`,
  `TopKSelector`, `ExistingLabelsOracle`, `ClassificationPredictions`,
  `EvaluationHistory`, and metrics/plotting utilities.
- Run random, entropy and margin acquisition with identical initial labels
  and total budgets; compare macro-F1 and balanced accuracy.
- Explore optional exercises in batch sizing, multiple seeds and diversity.

The main notebook is **fully executable with unsolved exercises**:
`RUN_EXERCISES=False` and `RUN_OPTIONAL_EXPERIMENTS=False` by default.
Students can enable them after filling in the TODO functions or for
additional research challenges.

### Portable installation

From the repository root (Python 3.11+):

```bash
python -m pip install -e ".[teaching]"
python -m ipykernel install --user --name al-pipeline-teaching --display-name "AL Pipeline (CPU)"
jupyter notebook examples/01_active_learning_digits.ipynb
```

If `jupyter notebook` is not available, launch the existing JupyterLab or
Jupyter Notebook frontend supplied by your environment and select the
`AL Pipeline (CPU)` kernel. `.[teaching]` installs the kernel/backend,
not necessarily a full standalone notebook frontend.

### CPU verification — on Delta or locally

Use these steps **without submitting any Slurm jobs**, from the checked-out
repo. Use an interactive CPU allocation if required by your site's policies.
For Delta with conda already installed:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
git fetch origin
git switch feat/cpu-active-learning-digits-notebook
git pull --ff-only origin feat/cpu-active-learning-digits-notebook
conda activate al-pipeline
python -m pip install -e ".[teaching]"
python -m pip install pytest
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/examples/test_digits_notebook.py
```

This test executes **every cell** with `nbclient` on a CPU and fails if
any cell raises an exception. The same test runs automatically on clean
GitHub-hosted Python 3.11 and 3.12 CPUs.

To explicitly **save an executed copy with figures and tables**, use the provided CPU runner:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/examples/execute_digits_notebook.py \
  --output ~/digits_active_learning_executed.ipynb
```

The runner refuses to overwrite existing files and keeps the source notebook unchanged. It does not submit Slurm jobs.

Alternatively, run directly using the standard `nbclient` API:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
source = Path("examples/01_active_learning_digits.ipynb")
out = Path("examples/01_active_learning_digits_executed.ipynb")
notebook = nbformat.read(source, as_version=4)
executed = NotebookClient(
    notebook, timeout=300, kernel_name="python3",
    resources={"metadata": {"path": str(Path.cwd())}},
).execute()
nbformat.write(executed, out)
print("Saved:", out)
PY
```

The executed file is a *local experiment output*; do not commit generated
notebook results or plots without reviewing them.

### Experimental constraints

The notebook has exactly the same initial labeled IDs and fixed
train/validation/test partition for all default policies. All three policies
acquire 160 labels in eight rounds of 20 each. Logistic regression is trained
from scratch at each cycle using **only observed labels**.

The existing labels of otherwise unlabeled candidates are held inside an
`ExistingLabelsOracle` and revealed only for the selected batch. Hidden
labels are never used to choose candidates. The test split is evaluated
**only once at the end**, after selecting a policy using validation macro-F1.

An optional seed exercise deliberately keeps the split and `L0` fixed;
it only probes acquisition RNG variability (not full experiment variance).
A follow-up notebook can vary the initial set, label noise, or a more
expensive oracle.

### Relationship to the astronomical project

The classification model here is logistic regression, **not** the
AstroMiNN/ConvNeXt image model. The relevant generalizable abstractions
are scoring, pool state, selection, oracle, budgets and evaluation.

We will design a *separate*, optional astronomy example after confirming
the classroom experience and CPU runtime of this first notebook.
