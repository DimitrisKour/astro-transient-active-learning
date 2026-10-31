# Task 14 — SIMILAR facility-location active learning

**Status:** isolated implementation, mathematical CPU tests and source-patch dry run complete; **Delta GPU pilots and closed-loop training have NOT been run**.

Source: Kothawade, Beck, Killamsetty, Iyer (NeurIPS 2021), *SIMILAR: Submodular Information Measures Based Active Learning In Realistic Scenarios*. https://proceedings.neurips.cc/paper/2021/hash/9af08cda54faea9adf40a201794183cf-Abstract.html

This is a deliberately scoped SIMILAR-style benchmark, not a claim of reproducing every SIMILAR objective or the authors' exact experimental hyperparameters. Three explicit variants (the rare-query variant is specifically **FL2MI**, not the saturating FL1MI objective):

| Policy | Objective and anchor sets | Purpose |
|---|---|---|
| `similar_fl` | Facility-location coverage of current U | Standard representative/diverse subset |
| `similar_flcg` | FL conditional gain `f(A|L_t)` | New coverage beyond confirmed labeled set |
| `similar_flqmi_rare` | **FL2MI** (facility-location variant mutual information): `sum_(q in Q) max_(a in A) sim(q,a) + sum_(a in A) max_(q in Q) sim(a,q)`, with Q built from confirmed objects of two rarest L0 classes | Rare class targeting, no candidate truth |

Kernels are clipped-to-nonnegative, cosine-normalized, **factored last-linear-softmax pseudo-label gradient** similarities. This guarantees nonnegative monotone FL marginals, and avoids computing full dense gradient embeddings. Every class with a nonzero L0 count is eligible as a rare-class query; two rarest L0 classes are chosen by confirmed **object-level** counts and frozen for all cycles and seeds. Query representatives grow from *revealed* labels within each policy's L_t; no hidden labels are read for selection, and validation/test examples are never used as a query. Conditioning includes the whole already-confirmed training L_t. Some choices depart from the SIMILAR paper's particular experiments: they are documented, not silently claimed equivalent.

## Model / data controls

- Same initial L0 and U0, baseline model checkpoint, warm-start, training budget, validation-only evaluation as Task 4/Task 11.
- Independent seed 135/246/357 and policy trajectories, nine cycles (81 fits), 10% acquisition (~473 objects) per cycle, last cycle completes pool.
- Verified Task11 frozen source as parent, with an explicit patch only in a *copy*; other runs remain untouched.
- Held Slurm submission, no external Task11/12/13 dependencies, `afterok` only between adjacent cycles within each trajectory.
- 16 MiB filesystem probe before each pilot/training job because of recent Delta I/O issues.

## Deploy on Delta (after copying ZIP from Windows)

In Windows PowerShell:

```powershell
scp "$HOME\Downloads\task14_similar_overlay_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

In Delta's login terminal:

```bash
cd /projects/bcrv/dkourtesis
mkdir -p task14_similar_overlay_v1
unzip -n task14_similar_overlay_v1.zip -d task14_similar_overlay_v1
cd task14_similar_overlay_v1
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
python scripts/adaptive/task14.py plan
python scripts/adaptive/task14.py prepare
python scripts/adaptive/task14.py verify
python scripts/adaptive/task14.py preflight
```

## Acquisition-only GPU pilots; do not submit 81 fits until inspected

```bash
JOB=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task14_similar_facility_v1
for policy in similar_fl similar_flcg similar_flqmi_rare; do
  sbatch --parsable --time=00:45:00 \
    --job-name="T14_pilot_${policy}" \
    --output="$JOB/logs/%x_%j.out" --error="$JOB/logs/%x_%j.err" \
    "$JOB/source/scripts/adaptive/task14_worker.sbatch" "$JOB" pilot "$policy" 135
done
python scripts/adaptive/task14.py pilot-status
```

**Scientific pilot review:** compare selected IDs against Task4 entropy, check intra-/inter-cycle similarity, class enrichment only *after* label resolution, selector runtime, filesystem and CUDA errors. Test that the first complete training cycle writes its selection, labels, checkpoint, validation metrics and state before releasing later held cycles.

## Full training: separate explicit actions after pilots

```bash
python scripts/adaptive/task14.py submit --walltime 00:35:00
# 81 Slurm jobs are submitted HELD, without releasing or running any of them.
python scripts/adaptive/task14.py release --policy similar_fl --seed 135 --cycle 1
# After verifying first-cycle output, release others individually or:
python scripts/adaptive/task14.py release --all
python scripts/adaptive/task14.py status
```

Time limit intentionally starts conservatively at 35 min; after timing initial work, decide whether to reduce it to 20 min. A delayed filesystem read or GPU backend can cause Slurm timeouts despite fast selectors. Be careful that Slurm may refuse TimeLimit updates after jobs start.

To collect:

```bash
python scripts/adaptive/task14.py collect --destination /u/dkourtesis/task14_similar_results.zip
```

Then Windows PowerShell:

```powershell
scp dkourtesis@dt-login.delta.ncsa.illinois.edu:/u/dkourtesis/task14_similar_results.zip "$HOME\Downloads\"
```

## Verification on local checkout

Copy `src/al_pipeline/core/advanced_acquisition/similar.py` into the corresponding local checkout package or put the assembled frozen Task14 source on `PYTHONPATH`, then run:

```bash
PYTHONPATH=/path/to/frozen/task14/source/src pytest -q tests/test_task14_similar.py
```

The suite-specific patch is strict and will fail rather than silently modify unknown or changed source anchors. If Task11 frozen source has changed, do not bypass hashes; reconstruct from the verified frozen parent.

### Engineering assumptions / scientific limits

- With a linear softmax classifier the gradient factors match the final layer's pseudolabel cross-entropy gradient. For nonlinear heads or other models an adapter must supply suitable gradient features.
- `similar_flqmi_rare` references *training L_t*; unlike some paper setups there is no held-out rare-class **validation** query. No protected OOD set is available for this benchmark, so no OOD-conditioned SCMI variant is implemented.
- The algorithm treats each *object* as an acquisition unit, taking its selected latest alert's classifier-input embedding. Early-time performance and late-time drift are separate research questions.
- Performance comparisons are exploratory; rare-class enrichment and macro-F1 trade-offs must be measured on obtained labels and untouched validation, not inferred from the acquisition function.
