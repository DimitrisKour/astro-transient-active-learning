# Astronomical teaching data — small, reviewed exports only

Notebook [02](../02_applecider_astronomy_active_learning.ipynb) has two modes:

- **No data files:** a conspicuously marked, deterministic **synthetic** demonstration of a CPU-only active-learning loop, plus a few actual *presentation-level* AstroMiNN statistics. This is fully runnable offline but **is not an astronomical experiment**.
- **Real export installed:** the same notebook uses the compact object-level astronomical feature file `astro_real_v1.npz` for its interactive loop. If vetted historical CSVs are present, it also displays recorded AstroMiNN learning curves and class acquisition summaries.

We do **not** bundle raw historical `.npy` samples, trained AstroMiNN/PyTorch model checkpoints, Felipe's model sources, or private broker credentials. The supported notebook therefore needs only `pip install -e ".[teaching]"` and CPU time.

## A · One-time real features export

Run **on a machine where the historic manifest CSVs and original sample `.npy` files are available**. Work from the `al-pipeline` checkout; replace the placeholders with the actual manifest paths and columns from the dataset registry. This is a *preparation* step for the instructor, not something participants need to do.

```bash
python scripts/examples/export_astro_school_features.py \
  --train-manifest /PATH/TO/train.csv \
  --validation-manifest /PATH/TO/validation.csv \
  --test-manifest /PATH/TO/test.csv \
  --data-root /PATH/TO/HISTORICAL/SAMPLES \
  --object-id-column object_id \
  --file-column filepath \
  --label-column label \
  --train-objects 700 --validation-objects 200 --test-objects 200 \
  --output examples/data/astro_real_v1.npz
```

The actual column names may be different. Check `configs/delta.toml` (locally, not distributed) or the relevant `DatasetSpec` fields for `object_id_column`, `sample_file_column`, `label_column` and the three source manifests. The script also requires an integer `alert` column and trusted samples containing `obj_id`, `metadata` (34 values), and `image` (3×63×63).

Features are *deterministic CPU image summaries*: 8×8 averages from median-subtracted, standard-deviation-scaled science/reference/difference cutouts plus 34 clipped metadata features. They are **not** AstroMiNN ViT embeddings. The latest available alert per object is retained, and the number of available alert rows is stored separately. The exported benchmark is capped, preferentially keeps up to 12 objects/class in training and 4 objects/class in each evaluation split, and deliberately ensures rare classes have support; **its class proportions are not population estimates**.

A companion `astro_real_v1.provenance.json` lists the original manifest SHA-256 hashes, class counts and extraction method. Do not edit this to claim a model checkpoint was used.

## B · One-time historical-report export

Use an existing, complete multi-seed report created by `al_pipeline.astronomy.evaluation.reporting.write_report_tables` (the source report must be independently validated first):

```bash
python scripts/examples/export_astro_school_replay.py \
  --report-directory /PATH/TO/VERIFIED/REPORT/CSVS \
  --output-directory examples/data
```

This copies a **small, validated column subset**, rather than the source images or checkpoint files. It never invents missing cycles, seeds, or uncertainty scores.

- `astro_historical_evaluation.csv`: per-seed validation macro-F1 and balanced accuracy; enables complete historical learning curves.
- `astro_historical_acquisition.csv` (optional): revealed acquisition class counts by cycle.
- `astro_historical_overlap.csv` (optional): object-selection Jaccard statistics.
- `astro_historical_per_class.csv` (optional): object-level validation recall and F1.
- `astro_historical_provenance.json`: original report hashes and row counts.

The notebook reads these files only when they exist. It must not imply that changing a saved policy in a CSV can retrain AstroMiNN.

## Publication / teaching-school acceptance checklist

Before committing generated results, confirm all of the following:

1. **Rights and attribution:** the archived sample/label sources and derived numerical values are authorized for redistribution in the school repository; if uncertain, distribute the notebook privately instead.
2. **Data integrity:** cross-split object IDs are disjoint; `schema_version=1`; every target class is represented; sample, manifest and oracle labels agree; no unexpected NPY pickle sources are loaded.
3. **No leakage:** the feature construction never uses validation/test labels or any full-history supervised checkpoint. For a future AstroMiNN embedding export, only a checkpoint trained on the allowed `L0` (or independently permitted external pretraining) can be used for the hidden training candidates and validation/test.
4. **Small package:** inspect `.npz` and CSV byte counts and avoid unnecessary full-history arrays; consider Git LFS or external distribution if too large (ordinary GitHub files must remain below its file-size ceiling).
5. **Scientific quality:** run notebook 02 with the real export on a CPU and check early learning, per-class support, robustness across seeds and sensible class distributions. Poor performance is legitimate but must be explained rather than silently substituted.
6. **Provenance:** record which archived report suite, policies, model initialization, data split, seed set, metric level and validation-only selection protocol were exported.
7. **Presentation:** execute and commit the final **student** notebook with all its figures, but do not package instructor solutions or private storage paths in outputs.

The two notebook sections are deliberately separate: a **student-trained CPU surrogate** is not a reproduction of the saved AstroMiNN research trajectories.