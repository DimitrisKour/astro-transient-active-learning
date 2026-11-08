# Example notebooks

Hands-on examples of the reusable `al_pipeline` active-learning interfaces.
The notebooks are designed for participants on their own computers, in Jupyter
environments, or on shared computing systems. No particular institution or HPC
platform is required for the introductory example.

## 01 — Pool-based active learning with handwritten digits

**[Open the student notebook](01_active_learning_digits.ipynb)**

A CPU-only introduction using scikit-learn's bundled handwritten-digits
dataset. It requires no astronomical data, external downloads, GPU, API token,
survey broker, or pretrained model checkpoint.

The notebook **already includes an executed core demonstration** (text,
tables, plots, and learning curves). Every TODO remains an exercise for the
student. In particular, the provided code for TODO 3–5 is scaffolding, *not*
a worked solution.

### Learning objectives

- Distinguish labeled, unlabeled, and pending pool states; keep validation and
  sealed test data separate.
- Calculate predictive entropy and margin uncertainty.
- Use `al_pipeline.core` scoring, selection, pool, acquisition loop, oracle,
  evaluation history, and plotting functionality.
- Compare random, entropy, and margin acquisition under the same labeled-set
  initialization and total acquisition budget.
- Investigate batch-size choices, paired repeated runs, and uncertain-but-diverse
  selection through optional exercises.

### Installation and running

Use Python 3.11 or newer. From a clone of this repository:

```bash
python -m pip install -e ".[teaching]"
```

Open `examples/01_active_learning_digits.ipynb` in JupyterLab, Jupyter
Notebook, VS Code, or another notebook frontend with the Python environment
where you installed `al-pipeline`. A graphical notebook frontend may need
to be installed or enabled separately.

If your editor requires a separately registered kernel:

```bash
python -m ipykernel install --user --name al-pipeline-teaching --display-name "AL Pipeline (CPU)"
```

The instructions above also apply on remote computing systems, subject to their
local environment and resource-use policies.

### Worked core and student exercises

The supplied notebook preserves the core's executed outputs for reading and
comparison. To reproduce them, choose **Restart Kernel and Run All** in your
notebook editor. The following defaults ensure that the core executes
successfully even though the exercises are not yet solved:

```python
RUN_EXERCISES = False
RUN_OPTIONAL_EXPERIMENTS = False
```

The five exercises are intentionally unfinished:

1. Implement the predictive entropy formula.
2. Analyze entropy versus margin and implement a margin scorer.
3. Compare different acquisition batch sizes for a fixed labeling budget.
4. Compare repeated, **paired** initial labeled sets and uncertainty results.
5. Compare entropy with an uncertainty-plus-diversity selector.

After completing TODO 1–2, enable `RUN_EXERCISES=True` to run their checks.
For TODO 3–5, complete their guarded cells first, then enable
`RUN_OPTIONAL_EXPERIMENTS=True`. These later exercises perform additional model
fits; they are not required to reproduce the core notebook. The repository does
not distribute an instructor solutions notebook.

### Automated verification

From the repository root, the CPU runner saves a newly executed copy without
overwriting the published notebook:

```bash
python scripts/examples/execute_digits_notebook.py \
  --mode core \
  --output digits_core_local.ipynb
```

To test the unsolved student notebook's structure and core execution:

```bash
python -m pip install pytest
python -m pytest -q tests/examples/test_digits_notebook.py
```

An optional runner mode is available *after* solutions have been completed,
and the instructor-only mode requires a locally supplied, unpublished
solutions notebook. Neither mode is necessary for participants.

### Experimental design and limitations

The main experiment uses 50 initial labels (five per digit) and three
policies—random, predictive entropy, and top-two margin—each acquiring 160
additional labels over eight cycles. The classifier is multinomial logistic
regression; the results are a teaching demonstration, not a claim that one
acquisition policy is universally superior.

The simulated oracle stores the candidate ground truth and reveals only labels
requested by a selection. Candidate labels are not given to the acquisition
scorer. Policies are chosen using validation macro-F1; the test split is used
for one final evaluation. TODO 4 varies the balanced initial set *across*
replicates but holds it fixed *between* policies within a replicate.


## 02 — Historical active learning for astronomical transients

**[Open the executed student notebook](02_applecider_astronomy_active_learning.ipynb)**

A standalone, **CPU-only, read-only research-analysis lab** based on real, completed AstroMiNN active-learning experiments. Unlike the digits notebook, this notebook analyzes archived learning histories rather than retraining models. It requires **no GPU, astronomical images, model checkpoints, synthetic fallback, or connection to Delta**.

The small [included validation-history CSV](data/astro_recent_validation.csv) contains 1,020 metric records from 51 complete nine-cycle trajectories, with three seeds across eight suites. It includes **validation macro-F1, balanced accuracy, and label budgets only**—no candidate IDs, class-level acquisition outcomes, or test metrics. Its [provenance and limitations](data/README.md) are documented alongside the data.

### Guided learning sequence

- Verify the experimental label budget and validation metric contract.
- Compare random, entropy, and diversity sampling within a shared suite.
- Explore BALD, TAILOR, and uniform-mixture trajectories in a **separate, explicitly provisional cross-suite comparison**.
- Analyze early performance, final performance and seed variability.
- Compare original and alternative initial labeled sets in separate panels.
- Complete four open-ended, unsolved exercises involving budget thresholds, early efficiency, stability and methodological comparability.

Figures follow the presentation-inspired muted palette. Policy comparisons use at most three lines per panel; seed variation is shown separately without crowded error bars.

### Running it offline

From the repository root:

```bash
python -m pip install -e ".[teaching]"
python scripts/examples/execute_astro_notebook.py --output astro_school_local.ipynb
```

You can also open `examples/02_applecider_astronomy_active_learning.ipynb` in Jupyter or VS Code and **Restart Kernel and Run All**. Its four core figures are already embedded for GitHub preview.

To verify the historical data and notebook:

```bash
python -m pytest -q tests/examples/test_astro_notebook.py tests/examples/test_astro_replay_data.py
```

**Scientific limits:** equal cycle-zero metrics and identical budgets do not themselves prove that separate experiment suites are causally comparable. The alternative-`L0` experiment is treated separately; per-class/TDE/CV conclusions require an additional audited export. No results are inferred for unrecorded acquisitions.
