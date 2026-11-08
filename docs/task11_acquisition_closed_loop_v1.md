# Task 11 — advanced acquisition policies, Task 4-matched closed loop (v1)

## Frozen scientific design

New suite: `astrominn_task11_advanced_acquisition_v1`.

Three independent acquisition policies `badge`, `core_set`, `fisher_batch` × three training seeds 135, 246, 357 × nine acquisition/training/evaluation cycles = **81 independent GPU training fits**, chained within (policy, seed) only. All cycles reuse existing Task 4 L0 bootstraps from `astrominn_historical_replay_v1`. All underlying training hyperparameters and cumulative labeling budgets match Task 4: warm-start from loss-best ViT checkpoint, frozen experts, epoch=50 cap, patience=10, AdamW learning-rate=1e-4, batch=512, validation batch=2048, per-sample weighting and class weights computed from current confirmed L_t. The initial stratified L0 uses split seed 135, initial fraction 10%, at least 5/class, U_t is acquired in the same 10%-of-historical-training-size steps and the final cycle exhausts the pool. A held-out external labeled set is excluded as in Task 4.

**No SWA, ensemble, retries, temperature calibration, checkpoint acceptance, or new architecture.** As a deliberate evaluation hygiene improvement, the new benchmark evaluates fixed validation objects only, never test samples, during repeated cycles; Task 4 comparisons use its validation metrics only. No hidden candidate labels are used for acquisition: they are revealed by the existing simulated-label oracle **after** selection.

## Implementation audit

- `src/al_pipeline/core/advanced_acquisition/badge.py` already implements valid factored pseudo-label gradients and weighted k-means++ selection (first center maximum gradient norm). Task11 wires its real final linear input (`vit_tower.fc_out` forward pre-hook) into the AstroMiNN cycle driver, preserving object-level latest alert selection.
- `core_set.py` implements true k-center selection conditioned on **previously labeled anchors** plus new centers. Task11 extracts representations of confirmed L_(t-1) under the **same** checkpoint, performs unit-row normalization for both sets, and accelerates anchor initialization with chunked SciPy distance computation. Core-set is not the original unlabeled entropy shortlist diversity policy.
- Current `src/al_pipeline/core/scoring/fisher.py` implements a **pointwise expected Fisher trace**, not batch Fisher. Task11 leaves it untouched and introduces an **approximate BAIT-inspired** projected expected last-linear-layer Fisher *pool-risk A-optimal* greedy selection. With a single deterministic Gaussian projection of class gradient embeddings to rank 32, the objective is to reduce `trace(F_pool_mean @ inv(I + F_L + F_selected))` with a regularization ridge 1.0. Each selected candidate changes the inverse and marginal information gain via Woodbury. This is a *forward greedy* method, **not exact BAIT**, with no backward swaps and no full ViT Jacobians. The projection is constant across runs (seed 271828), and coordinates of the full input to the final classifier are included (global RMS scale; bias appended). The expected covariance is computed from predictive class probabilities, not pseudo-labels or oracle labels.

## Safeguards

`prepare` verifies and copies the **original Task4 frozen source tree and runner** to a new frozen location, adds any missing independent advanced-selector modules from the checkout, and patches only the copy; the user checkout, Task4 runner and old outputs are not overwritten. Source files and scientific settings receive SHA-256 verification. Patching uses exact upstream anchors and fails if the checkout differs. The frozen storage registry has only storage paths changed, not dataset/split definitions. Training chains are static SLURM `afterok` dependencies; a failed fit blocks descendants. Submission manifests prevent repeats. Initially require three **GPU acquisition-only pilot** jobs (one per policy, seed135) to verify actual representations, uniqueness, runtime and same-cycle overlap with Task4 entropy. If Fisher's first-cycle overlap is **95% or higher**, full submission refuses until investigated: the purpose is to avoid repeating the original entropy-like trace experiment by accident. This is a *pilot diagnostic* and not a way to tune projection hyperparameters to validation performance.

Note: A different mathematical objective does not guarantee different selections; the 95% pilot audit measures overlap empirically, without a forced diversity constraint.

### Install

Copy zip into `/projects/bcrv/dkourtesis/` on Delta. From the checkout:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
conda activate al-pipeline
unzip -n /projects/bcrv/dkourtesis/task11_advanced_acquisition_v1.zip -d .
python -m pytest tests/jobs/test_task11_fisher.py tests/jobs/test_task11_protocol.py -q
python scripts/research/historical_replay/task11/task11.py plan
python scripts/research/historical_replay/task11/task11.py prepare
python scripts/research/historical_replay/task11/task11.py verify
python scripts/research/historical_replay/task11/task11.py preflight
```

`preflight` executes Task4-compatible no-training manifest/bootstrap checks. **Inspect this output before submitting any GPU work.**

### Three acquisition-only GPU pilots

```bash
JOB=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task11_advanced_acquisition_v1
for method in badge core_set fisher_batch; do
  sbatch --parsable --job-name="T11_pilot_${method}" \
    --output="$JOB/logs/pilot_${method}_%j.out" \
    --error="$JOB/logs/pilot_${method}_%j.err" \
    --time=00:20:00 \
    "$JOB/source/scripts/research/historical_replay/task11/task11_worker.sbatch" "$JOB" pilot "$method" 135
done
```

Check `sacct`, the logs, and:

```bash
python scripts/research/historical_replay/task11/task11.py pilot-status
```

A pilot computes no model-training epochs. It may score the entire U0 pool and confirmed L0; therefore it must execute on a GPU node. Compare selection times and overlap: it should run comfortably within the allotted time. It is not appropriate to launch all 81 fits if any pilot fails or is unacceptably slow.

### Submit (after all pilots pass)

```bash
python scripts/research/historical_replay/task11/task11.py submit --walltime 00:20:00
python scripts/research/historical_replay/task11/task11.py status
```

This queues **81 GPU jobs** as **9 independent seed/policy chains**, with one job per cycle and a 20-minute walltime. GPU training duration is estimated from Task 4 at 4–6 device-hours total, **plus additional acquisition inference**. Core-set and Fisher extra runtime must be determined from the pilots; increase `--walltime` if necessary. Pinned scientific code is not modified by a walltime increase.

When the nine trajectories finish:

```bash
python scripts/research/historical_replay/task11/task11.py status
python scripts/research/historical_replay/task11/task11_report.py
```

Reports are saved under `/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task11_advanced_acquisition_v1/report/` as `trajectories.csv`, `paired_task4.csv`, `classwise_validation.csv`, `acquisition_overlap.csv`, and `summary.json`. They contain validation-only macro-F1, BA, accuracy, AUCPR, NLL, Brier and ECE; for every class and cycle, **TP/FP/FN**, precision, recall, F1; and same-cycle acquisition overlap with Task4 entropy. Full image-level predictions stay in the experiment's private heavy storage and do not need to be copied for aggregate analysis.

### Limitations / go-no-go

- This package has synthetic local tests and **has not been executed against Delta's real checkpoints yet**. The first GPU pilots are mandatory to validate the live model integration, time, and RAM.
- The batch Fisher method is **BAIT-inspired** but is not the original paper's exact full algorithm. The projected 32D subspace is a principled compute compromise; do not equate a weak result with all Fisher-based batch methods.
- Source checkpoint compatibility and byte-identical L0/U0 are asserted in the pilot. Run all science jobs using the frozen source, never the working checkout.
- If the Fisher pilot is nearly identical to entropy, do **not** simply randomize its output to force novelty. Inspect the objective, feature geometry, and gradient sketch diagnostic before changing a scientific parameter; use a *new suite version* for changes.
- A cycle-9 endpoint consumes the full pool. Aggregate comparisons should focus on intermediate label budgets and area under curves.
