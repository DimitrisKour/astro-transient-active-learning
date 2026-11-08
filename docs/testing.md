# Testing guide

The package has **two testing workflows**, with different requirements. Neither
workflow needs actual broker credentials for its synthetic tests.

## 1. Portable tests (default, public contributors and CI)

On Python 3.11+ in a fresh virtual environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

This default command collects only `tests/core/` and `tests/integration/`
(via `pyproject.toml`). It exercises scoring, selection, pool transitions,
oracle/resolver contracts, validation metrics, checkpoint policies, state
persistence, and a synthetic end-to-end astronomy-label workflow.

These tests do **not** download datasets, call BOOM/TNS, read Delta scratch
directories, access frozen historical training outputs, or require GPUs.
Most use synthetic arrays or temporary test directories.

One optional last-layer Fisher autograd comparison and optional SWA tests
use PyTorch **if installed**, but skip when it is absent. All remaining tests
must pass without PyTorch. Analysis dependencies are installed in the `dev`
extra. A skipped optional test is **not** proof that its optional integration
works; run a separate environment with PyTorch to cover it.

A narrower smoke check for minimal installations is:

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest tests/core/data tests/core/eligibility tests/core/evidence tests/core/oracle tests/core/test_loop.py tests/core/test_pool.py -q
```

The default portable suite assumes `.[dev]` and its optional analysis
dependencies, not just the NumPy-only runtime installation.

## 2. Research and historical reproduction (explicit opt-in)

The complete repository test collection is **not** the public default:

```bash
python -m pip install -e ".[dev,astronomy]"
python -m pytest tests -q
```

This explicitly includes `tests/astronomy/`, `tests/jobs/`, and historical
experiment tests at the root of `tests/`. These require optional astronomy
libraries, pinned legacy model implementations and/or historical experiment
artifacts in some cases. Installing `.[astronomy]` alone does not supply
external `fusion_model_basics`, legacy AstroMiNN source, pretrained weights,
or Delta-specific datasets.

The Task 11 frozen-source fixture is intentionally opt-in, with **no** implicit
cluster-specific fallback. For relevant tests, set:

```bash
export APPLECIDER_TASK11_SOURCE=/path/to/verified/task11/source
python -m pytest tests -q
```

PowerShell equivalent:

```powershell
$env:APPLECIDER_TASK11_SOURCE = "C:\path\to\verified\task11\source"
python -m pytest tests -q
```

`APPLECIDER_TASK11_SOURCE` can point to a directory containing
`src/al_pipeline/` or its parent containing `source/`. The fixture fails
clearly if it is absent or invalid; it never quietly substitutes arbitrary
artifacts. Depending on the experiment, additional paths, model code,
checkpoints, and environment configuration must be provided. Consult the
relevant experiment's reproduction guide.

For synthetic scheduler/experiment contract tests without the full historical
data, you can select an individual test module explicitly, e.g.:

```bash
python -m pytest tests/jobs/test_balanced357_strict_retry.py -q
```

The full historical suite is a reproducibility interface, **not** currently
claimed to pass on a clean external machine. It is not part of public CI until
each research-only dependency is documented and made available or mocked.

## Test-organization policy

- Preserve invariant, synthetic, and scientific regression tests. Do not remove
  them merely because an experiment has been superseded.
- Do not add Delta paths or private data as fallback defaults in shared
  test fixtures.
- Add portable tests under `tests/core` or `tests/integration` only if
  they can run without network calls, GPUs, historical directories or private
  artifacts.
- Place frozen experiment and cluster job tests under `tests/astronomy`,
  `tests/jobs` or separately opt-in historical test modules.
- Run `python -m pytest -q` after public-library changes, then targeted
  research tests after changing experiment-specific code.
