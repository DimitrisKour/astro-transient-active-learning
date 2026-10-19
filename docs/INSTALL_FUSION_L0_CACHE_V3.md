# New-model L0 + validation cache and single-epoch benchmark

This is a **self-contained overlay**, built for the existing `al-pipeline` checkout on NCSA Delta. It adds two new modules `cache_helpers.py` and `cache_v2.py` and a bounded SLURM orchestration. It does **not** overwrite the existing `backends.py`, L0 source plan, original manifests, historical data, or any AL training code. The current backend already supports cache version 2 whenever `FUSION_IMAGE_CACHE_ROOT` is set.

## Scope and scientific guardrails

- Cache only the **original historical L0 train manifest** and the **unchanged fixed validation** manifest. No U0, full training pool, or test split is included.
- Keep original labels and row order (including repeated object-alert rows) in the source CSVs. Cache only deterministic, float32, preprocessed image/55-feature inputs.
- For each object, decompress its source NPZ once per cache build shard, then select its verified historical alert stages.
- The build verifies model-source and index identities, hashes input manifests, checks each NPZ alert index, compares cached values against the original per-alert loader, and publishes `cache_manifest.json` **only after all shards complete**.
- The GPU benchmark measures **one full pretrained BTSbot seed-135 L0 epoch** and **complete validation**, retaining weights and metrics but never examining test labels.
- No active learning, model-selection comparisons, hyperparameter tuning, or six-fit matrix is submitted by this overlay.
- Use a new immutable root `new_image_l0_cache_v3` so previously attempted caches or jobs cannot be overwritten.

## Transfer and extract

Windows PowerShell:

```powershell
scp "$HOME\Downloads\fusion_l0_cache_ready_v3.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

Delta (from the repo root):

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/fusion_l0_cache_ready_v3.zip -d .
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/astronomy/fusion_image/test_sharded_cache_v2.py -q
python -m pytest tests/astronomy/fusion_image -q
bash scripts/adaptive/submit_fusion_cache_v3.sh --prepare
```

Check the `--prepare` printed object and stage counts and the two source manifest paths. Then run:

```bash
bash scripts/adaptive/submit_fusion_cache_v3.sh --submit
```

This submits up to six 1-CPU preprocessing shards concurrently (24 shards total), a dependent CPU finalization step, and **one** dependent A40 GPU one-epoch benchmark. The batch script runs only after the cache completes successfully. Do not submit the original L0 six-fit matrix.

## Monitoring

```bash
PIPE=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_cache_pipeline_v3
CACHE=/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_l0_cache_v3
cat "$PIPE/submitted_jobs.tsv"
squeue -u "$USER"
find "$CACHE" -maxdepth 2 -name done.json | wc -l
tail -40 "$PIPE"/logs/*.err
```

After the benchmark finishes:

```bash
BENCH=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_cached_benchmark_v3
cat "$BENCH/training_history.json"
cat "$BENCH/training_summary.json"
tail -80 "$PIPE"/logs/bench_*.out
```

The most useful metrics are `training_seconds`, `validation_seconds`, `epoch_seconds`, `train_loss`, `validation_loss`, and `validation_object_macro_f1`. Compare against the former uncached epoch only when the model/data/seed configuration matches.

## Retries and safety

If a shard fails, inspect the corresponding log and residual `.shard_XXX_*` staging directory first. Once the old jobs have finished, use:

```bash
bash scripts/adaptive/submit_fusion_cache_v3.sh --retry
```

Completed shards are reused. Never delete `shard_*/done.json` blindly. Do not reuse this L0+validation cache for pool inference or test evaluation: missing stages will raise an error. If cache finalization succeeded but the benchmark failed, investigate the benchmark logs before submitting the standalone `fusion_cache_v2_benchmark.sbatch` job manually.

## Interpretation

The previously observed 7–27x difference compared two **different CPU processing paths**. No training speedup is established until this exact cached DataLoader finishes an epoch and full validation on the GPU node. The cache's peak resident RAM includes staged input tensors, so GPU jobs use a 20-GB host memory request.

The `core` architecture is unchanged. This is a purely optional astronomy adapter optimization.
