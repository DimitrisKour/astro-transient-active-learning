# Astronomy notebook — included teaching data

[Notebook 02](../02_applecider_astronomy_active_learning.ipynb) uses **small, precomputed results from completed AstroMiNN active-learning experiments**. Students can analyze the learning curves, class-level outcomes and acquisition overlaps without downloading observations, loading trained models or using a GPU.

## Files

| File | Contents |
| --- | --- |
| `astro_recent_validation.csv` | 1,020 recorded validation measurements (macro-F1 and balanced accuracy) from 51 trajectories, each evaluated at cycles 0–9 |
| `astro_recent_per_class.csv` | 2,550 object-level validation precision, recall, F1 and support records across five classes (AGN, TDE, SN I, SN II, CV) |
| `astro_cycle2_overlap_by_seed.csv` | 45 seed-level policy-pair overlap records at cycle 2 |
| `astro_cycle2_overlap_mean.csv` | 15 policy-pair summaries averaged over three matched seeds |
| `astro_recent_provenance.json`, `astro_recent_per_class_provenance.json`, `astro_cycle2_overlap_provenance.json` | Data origin, schema and export scope |

**Cumulative Jaccard overlap** uses the sets of acquired objects from **cycles 1 and 2 combined** (946 objects per policy), excluding the shared initial labeled set. The matrix compares entropy, entropy + diversity, BADGE, BALD and TAILOR. Values are means of same-seed Jaccard coefficients; no selected-object IDs are distributed.

## Run the student notebook

From a local clone of this repository:

```bash
python -m pip install -e ".[teaching]"
python scripts/examples/execute_astro_notebook.py --output astro_school_local.ipynb
```

Or simply open the [executed notebook](../02_applecider_astronomy_active_learning.ipynb) in Jupyter or VS Code. Everything required for its plots is included here; no Delta configuration, model training or internet access is needed.

## Interpretation

- All curves and per-class measurements come from **saved validation evaluations**, not from newly trained student models.
- The first learning-curve figure combines two entropy policies from one suite with BADGE from another. The BALD/TAILOR comparisons similarly span recorded suites; matching label budgets and baseline scores alone **do not establish full protocol equivalence**.
- The alternative initial labeled set is shown separately. Three seeds support descriptive variability analysis, not formal significance testing.
- The class-level metrics describe **validation performance**; the overlap tables describe **which candidate objects were selected**. Neither includes test metrics, raw observations, checkpoints, revealed candidate identities or hidden-label tables.

## Recreating the exports (instructor/research use)

The following scripts read existing recorded artifacts and do **not** train or evaluate models from scratch:

- [`export_recent_astro_replay.py`](../../scripts/examples/export_recent_astro_replay.py) — validation histories from persistent run state.
- [`export_recent_astro_per_class.py`](../../scripts/examples/export_recent_astro_per_class.py) — recorded per-class evaluation files.
- [`export_cycle2_selection_overlap.py`](../../scripts/examples/export_cycle2_selection_overlap.py) — recorded selections; exports aggregate overlap, not IDs.

Running those exporters requires access to the historical research files; **students do not need to run them**. Older optional feature-export scripts in `scripts/examples/` are retained for research reuse but are not part of notebook 02.

**Redistribution:** this is a research-derived teaching export. Review permission to share these numerical results outside an authorized audience; do not copy private raw data, credentials or checkpoints into the repository.