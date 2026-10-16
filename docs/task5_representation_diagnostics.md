# Task 5 — Representation diagnostics and adaptation roadmap

Task 5A is a separate experiment. It does **not** use Task 4's changing acquisitions,
write into Experiment B, train the AstroMiNN ViT, or reveal hidden `U0` labels.

## Scientific questions and scope

1. Does a **linear probe** of frozen expert-tower outputs classify historical objects on the fixed evaluation splits?
2. How does that compare to the **ViT CLS embedding** presented to `vit_tower.fc_out` after training the bootstrap on `L0`?
3. Which astrophysical classes, especially low-support TDE and CV, show weak separability or low nearest-training-object label agreement?

The probe fits **only to original Experiment B `L0`**. Validation and test are
original published historical splits; they are never used to train or tune the
probe. The probe uses a predeclared standard scaler + class-balanced logistic
regression with C=1.0; no grid search or test-driven selection. All requested
representations are reported, not just a selected winner.

The original AstroMiNN model's **expert towers are previously trained**.
Their historical training provenance must be checked before interpreting
probe results as independent generalization; we cannot automatically prove
those tower weights never saw objects in the historical replay pool. This is
a diagnostic, not a claim of leakage-free pretraining or causal improvement.

## Representation definition

- `vit_cls`: normalized CLS vector passed to `model.vit_tower.fc_out`, captured
  with a forward pre-hook, identically to current AL embedding inference.
- `lc_expert`, `spatial_expert`, `coord_expert`: tensor returned by each
  expert tower in the model's standard forward pass.
- `expert_concat`: concatenation of the three expert tower outputs in fixed
  order (lightcurve, spatial, coordinates). Unlike `vit_cls`, this hook
  boundary depends on the external historical `astroMiNN_V2` implementation.

All representations are evaluated per **object**, keeping only the single
highest-numbered alert in the manifest. Duplicate alert numbers for the same
object are rejected. This mirrors the production `latest` aggregation method
and reduces CPU work. It is not mean-alert pooling and is not an independent
scientific choice made after seeing metrics.

There is no augmentation during feature extraction. The canonical existing
AstroMiNN manifest dataset handles image and metadata preprocessing.
The fitted extractor uses the existing random-ViT bootstrap for seed 135
from Experiment B and the same frozen expert checkpoint files as Task 4.

## Project paths on Delta

Repository: `/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning`

Registry: `/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_v1/registry.toml`

Original L0: `/projects/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_historical_replay_v1/shared/historical_replay_split/initial_labeled_manifest.csv`

Checkpoint is resolved from the registry's `storage.data_root` as:
`runs/astrominn_historical_replay_v1/shared/seed_135/bootstrap_training/best_vit_tower.pth`.

All Task 5A outputs go under `/projects/bcrv/dkourtesis/applecider_adaptive/representation/task5a_v1/`.

## 1. Install and test

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/task5_representation_overlay_v1.zip -d .
python -m pytest tests/astronomy/representation/test_task5a.py -q
python -m pytest tests/ -q
```

The patch adds new files only. It does not touch the existing Task 4 source
snapshot or SLURM queue.

## 2. Prepare CPU smoke run (no scheduling or model inference yet)

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export T5_ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/representation/task5a_v1
export T5_REGISTRY=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_v1/registry.toml
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/adaptive/task5a.py --run-dir "$T5_ROOT/smoke" prepare \
  --registry "$T5_REGISTRY" --mode smoke --shards 1 --seed 135 \
  --representations vit_cls expert_concat
```

The command validates source input hashes, all original object-level splits,
class labels, duplicate alerts, and disjointness. It writes a frozen plan.
There is no job submitted and no GPU use.

Run the CPU smoke **on a compute node**, not on the shared login node:

```bash
mkdir -p "$T5_ROOT/logs"
sbatch --job-name=T5A_smoke --output="$T5_ROOT/logs/smoke_%j.out" \
  scripts/adaptive/task5a_cpu.sbatch smoke "$T5_ROOT/smoke"
```

The CPU smoke samples three objects per class per split (up to 45 total),
extracts their features, runs linear probes and saves a report. It is a
technical check only, not a scientific result. It also checks whether hooks
on the external published expert towers are compatible. If `expert_concat`
fails, **do not mask or silently substitute features**: inspect the shape and
call-site error first.

Smoke outputs:

```
smoke/plan.json
smoke/features/{train,validation,test}/shard_000_of_001.npz
smoke/features/{train,validation,test}/shard_000_of_001.json
smoke/report.json
smoke/metrics.csv
smoke/per_class_metrics.csv
smoke/summary.md
smoke/pca_projections.csv
```

`plan.json` contains source/checkpoint/model and code SHA-256 fingerprints,
object-level sample selections, split counts, and the fixed probe protocol.
Each feature archive has a matching SHA-256 metadata file. A rerun with an
unchanged completed shard validates it and skips inference. An uncommitted NPZ without metadata is safely re-extracted on retry;
a mismatched or corrupt committed shard is refused rather than silently
mixed into results.

## 3. Plan a complete CPU experiment (after smoke passes)

```bash
python scripts/adaptive/task5a.py --run-dir "$T5_ROOT/full" prepare \
  --registry "$T5_REGISTRY" --mode full --shards 12 --seed 135 \
  --representations vit_cls expert_concat
```

This plans **all** objects in original `L0`, validation, and test, one latest
alert per object. Shards are assigned by stable object-ID hash, so workload
can be spread over short CPU jobs and retries are isolated. Nothing is
submitted automatically.

**Example** submit one shard for timing, then evaluate its runtime and memory:

```bash
sbatch --job-name=T5A_train00 --output="$T5_ROOT/logs/train00_%j.out" \
  scripts/adaptive/task5a_cpu.sbatch extract "$T5_ROOT/full" train 0
```

After inspecting one shard's actual runtime, the optional **CPU-only job
array launcher** can submit all three splits × 12 shards with a concurrency
limit, plus an `afterok` report job. It defaults to dry-run:

```bash
bash scripts/adaptive/task5a_submit_cpu.sh --run-dir "$T5_ROOT/full" --dry-run
```

If the CPU runtime and memory look reasonable, explicitly submit:

```bash
bash scripts/adaptive/task5a_submit_cpu.sh --run-dir "$T5_ROOT/full" \
  --submit --parallel 4 --walltime 00:30:00
```

This makes one 36-task SLURM CPU array (at most four running at a time),
plus a report job that waits for **all** shards to succeed. Jobs are held
during submission and released only once the dependency is established.
The launcher writes `full/submitted_cpu.tsv` and refuses duplicate
submission. It does not create GPU jobs. The per-task walltime of 30
minutes is a starting scheduling request, not a measured runtime guarantee.
If any shard fails, the report does not run; inspect and repair that shard
rather than duplicating the entire array.

To report manually after all shards exist, run:

```bash
sbatch --job-name=T5A_report --output="$T5_ROOT/logs/report_%j.out" \
  scripts/adaptive/task5a_cpu.sbatch report "$T5_ROOT/full"
```

The reporter **refuses** to run with missing, duplicate, changed, mixed or
corrupted shards. It reports classification accuracy, balanced accuracy,
macro-F1, confusion, per-class precision/recall/F1, log loss and nearest-L0
neighbor label agreement and L0-fitted 2D PCA coordinates for all requested representations, across train,
validation and test. Validation can inform Task 5B design; test is descriptive
and not used to select an adaptation method.

## 4. Task 5B and 5C (NOT queued)

`representation/design.py` declares type interfaces for feature extractors,
fixed-manifest adaptation designs, and downstream closed-loop comparisons.
They do not execute GPU training. After 5A, decide whether representation
adaptation should target the ViT, experts, or a lightweight projection, and
pre-register trainable layers, fixed label budget, losses, and seeds.

**5B** should compare adaptations on identical saved acquisition manifests
before making any active-learning claim. **5C** should run a closed-loop
comparison only after Task 4 has completed and its results have been
examined. No Task 5 GPU submission is implemented in this overlay.

### TODOs

- Validate that each expert's forward-hook output has semantics appropriate
  to a *representation* rather than merely a classifier score.
- Check historical expert pretraining provenance and overlap with replay data.
- If TDE support is tiny, report uncertainty across random seeds before
  claiming robust separability.
- Reuse the 5A object split and provenance contract if implementing 5B/5C.
- For 5B comparisons across acquisition cycles, use identical cumulative
  manifests and hold validation/test fixed, avoiding confounding acquisition.
