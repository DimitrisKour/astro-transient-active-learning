# AL Pipeline

Reusable active-learning infrastructure with explicit astronomy integrations.
Extracted from `DimitrisKour/AppleCiDer-AL`, commit
`b0aa85c11644b50db05e4dc0e9d94e656017fb17`.

## Start on Delta

See [Delta installation](docs/delta_installation.md). The first extraction
preserves existing dataset, legacy AstroMiNN code, model configuration, and
checkpoint paths. It does not move data or resume existing production runs.

## Find the code

| Package | Responsibility |
| --- | --- |
| `al_pipeline.core.scoring` and `selection` | Convert classifier outputs or embeddings to acquisition priorities; select eligible IDs |
| `al_pipeline.core.loop` and `pool` | Propose/commit acquisition and manage logical states |
| `al_pipeline.core.evidence` and `oracle` | Provider-independent evidence and label-resolution contracts |
| `al_pipeline.core.workflow` | Acquire, gather evidence, resolve labels, apply lifecycle policy |
| `al_pipeline.core.state` | Object history and durable run progress |
| `al_pipeline.core.training`, `evaluation`, `experiments` | Backend contracts, metrics/history, and run policies |
| `al_pipeline.astronomy.boom` | BOOM client, pool materialization, and catalogue evidence |
| `al_pipeline.astronomy.ztf` | ZTF packet interpretation, photometry, eligibility, and preprocessing |
| `al_pipeline.astronomy.labels` | AppleCiDEr-specific catalogue-to-taxonomy rules |
| `al_pipeline.astronomy.data` | Survey manifest schema, dataset registry, cumulative labelled CSVs |
| `al_pipeline.astronomy.astrominn` | Legacy model loading, inference, training, evaluation, and candidate inspection |
| `al_pipeline.astronomy.experiments` | Retrospective A/B coordinators and their dataset-specific setup |
| `al_pipeline.astronomy.evaluation` | Multi-seed reporting of the existing astronomy run artifacts |

`core` never imports `astronomy`. Its basic loop and resolution workflow need
only NumPy and the Python standard library. Analysis utilities add optional
pandas, Matplotlib, and scikit-learn dependencies. Astronomy integrations add
PyTorch, torchvision, and Astropy. No AppleCiDEr, Hyrax, or FLARE import is
required to use this extracted package.

The retrospective coordinator remains in `astronomy/experiments/simulated.py`
because it assumes survey/alert manifests and the current artifact layout.
Reusable orchestration lives in `core/loop.py` and `core/workflow.py`; we do not
pretend the retrospective manifest schema is domain-independent.

## Experiments and notebooks

- [Experiment A](docs/simulated_active_learning_runs.md): historical data plus
  cumulative acquired BOOM/TNS labels, with the existing initialization policies.
- [Experiment B](docs/astrominn_historical_replay_experiment_b.md): historical-only
  replay, fixed object split, seed-specific bootstrap, random ViT retraining.
- [Architecture and migration](docs/migration.md): import mapping and preserved behavior.

Launchers retain their argument names. Change the registry argument to
`configs/delta.toml`. Omitting `--run` retains each launcher's existing preflight
behavior; A can materialize its holdout and B its deterministic split, so use
fresh suite IDs for new checks. The guides use the `al-pipeline` environment
and a per-process Delta C++ runtime helper. Keep one-epoch checks separate
from full-budget experiments; retain the same settings when resuming a run.

Delta validation on 2026-10-08 passed all 316 tests, matched A's holdout rows
and B's split rows/source hashes against the original runs, and completed
one entropy cycle in both GPU smoke suites (including B's bootstrap).
Full-budget numerical reproduction remains unverified. See
[migration validation](docs/migration.md#validation-status-on-delta).

Notebooks are not copied yet. Adapt them using the module and symbol maps in
`docs/`; old `applecider.adaptive` imports are deliberately not installed as
aliases. Upstream AppleCiDEr and FLARE remain separate sibling repositories.
The original code has no standalone FLARE adapter to extract; future notebook
migration can use the sibling FLARE package or add an adapter here when needed.

## Testing

The default test suite is **portable**: no Delta filesystem, private
datasets, broker credentials, GPU, or historical model checkpoints are needed.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

This runs the core and synthetic integration tests. Optional PyTorch-dependent
checks are skipped when PyTorch is unavailable. GitHub Actions exercises the
same commands on clean Python 3.11/3.12 environments.

The historical research/Slurm tests are retained but explicitly opt-in.
They can require optional dependencies and frozen experiment artifacts:

```bash
python -m pip install -e ".[dev,astronomy]"
python -m pytest tests -q
```

See [the testing guide](docs/testing.md) for required environment variables,
test selection, and the distinction between portable and historical checks.

## Local development

```bash
python -m ruff check .
python -m ruff format --check .
```

Use the Delta instructions to reuse existing dependencies without upgrading them.
