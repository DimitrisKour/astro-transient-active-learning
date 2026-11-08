# Task 10–12 historical research experiments

These are **research reproducibility controllers**, not general entry points for
the active-learning package. The scientific changes were implemented in
isolated frozen sources, and cannot be conflated with the public `src/` API.

| Family | Purpose |
| --- | --- |
| `task10/` | Twenty fixed-seed retries at an original Task4 entropy regression, with validation-only analysis |
| `task11/` | BADGE, core-set and approximate Fisher A-optimal active learning |
| `task12/` | Monte Carlo dropout BALD acquisition, matched to historical Task4 |

Task10 depends on the **historically frozen Task8B report** and keeps its
original archived `scripts/adaptive/task8b_report.py` source provenance.
Its relocated readout can additionally resolve the current grouped reader
when invoked from the checkout. Do not substitute an unverified newer
report implementation into an original frozen experiment.

Tasks 11 and 12 apply source-only patches to validated **Task4 frozen**
checkpoints and training/inference implementations. The patching procedure
refuses changed anchors and never edits production `src/` files.

Task11's `task11_repair_import.py` is intentionally retained as a one-time
audited import repair, not advertised as a new acquisition method.
Task12's BALD pilot and later full chain have explicit dependency gates.

## Synthetic validation

```bash
python -m pip install -e ".[dev]"
python -m pytest -q \
  tests/jobs/test_task10_entropy20.py \
  tests/jobs/test_task11_fisher.py \
  tests/jobs/test_task11_protocol.py \
  tests/jobs/test_task11_repair_import.py \
  tests/jobs/test_task12_protocol.py
```

BALD's PyTorch tests require the optional `torch` installation:

```bash
python -m pytest tests/jobs/test_task12_bald.py -q
```

These tests do not train astronomy models, contact brokers, or submit Slurm
jobs. Source snapshots previously created on Delta preserve their
`scripts/adaptive/task10*`, `task11*`, and `task12*` paths. Never use
the reorganized checkout to overwrite or resume those frozen sources.
For a fresh reproduction, choose a **new suite ID** and verify the
historical artifact/data availability before submitting.
