# Task 12 — MC-dropout BALD on AstroMiNN (strict, Task 4-matched)

## Scientific goal

Test Bayesian Active Learning by Disagreement (BALD; mutual information)
against the previously completed Task 4 baseline policies, without changing
training, initial labeled objects, selection budget, or checkpoint handling.
The only intended scientific difference is the acquisition policy.

**Experiment:** Three seeds (135, 246, 357) × nine cycles = 27 fits;
10% of the historical training objects acquired per cycle (473 objects, last
cycle 469), initial 10% stratified L0, original Task 4 warm-start settings.
Each trajectory uses its own sequence of checkpoint updates and acquired labels.
Validation evaluation only during policy development; test predictions are not
read for selection or repeated reporting.

## BALD definition and inference protocol

For each candidate object x and T=20 approximate posterior model samples:

    score(x) = H[mean_t p(y | x, w_t)] - mean_t H[p(y | x, w_t)]

Entropy uses natural logarithms (nats). This is estimated mutual information
between the class and the dropout-induced model-weight approximation. It is
**not** equal to predictive entropy or ensemble disagreement from independent
training seeds. A top-k selector chooses the 473 largest MI scores; ties are
resolved by deterministic original pool order.

- Original Task 4 ViT checkpoint is loaded in evaluation mode. Published
  expert towers and BatchNorm remain in evaluation mode.
- Only effective `nn.Dropout`, `nn.Dropout1d`, `nn.Dropout2d`, and `nn.Dropout3d`
  modules in the *ViT tower* are sampled. Each sample draws a mask for each
  dropout module, **keeps that mask fixed across all batches/objects**, and
  redraws all masks between posterior samples. Thus model index t has a
  coherent interpretation over the entire pool, as required for genuine
  joint-model-based acquisitions.
- In inference, only the latest historical-compatible alert for each candidate
  object is used, matching Task 4's object-level `latest` aggregation.
  There is no sample-level acquisition. Duplicate latest-alert ties are refused.
- No candidate labels are read to compute BALD scores. No validation or test
  labels are accessed during acquisition or dropout fitting; there is no
  dropout fitting phase.
- If the model has no nonzero supported dropout modules or predictions have
  effectively zero stochastic variation, acquisition **fails closed**. An
  alternate posterior model (e.g. last-layer Laplace) then requires a distinct
  explicitly named experiment; do not silently use entropy or injected noise.
- PyTorch functional dropout and non-standard stochastic-depth modules are
  **not** automatically intercepted. A pilot audits the supported layers and
  nonzero stochasticity; this must be considered when interpreting posterior
  uncertainty. The dropout posterior is a variational approximation, not
  exact Bayesian posterior inference.

Per-cycle MC predictive samples are cached in a compressed NPZ alongside a
checkpoint/object-order/seed/cycle provenance JSON. Cached candidate scores
are verified against the current ordered candidate set and acquisition policy.

## Why not include BatchBALD in the main experiment?

BatchBALD (Kirsch, van Amersfoort & Gal, NeurIPS 2019) greedily maximizes
joint mutual information, I(Y_batch; weights), rather than sums of individual
I(Y_x; weights). Exact joint entropy grows exponentially in selected batch
size; even published approximate approaches need a carefully validated joint
entropy Monte Carlo approximation. The current batch size is ~473. A naive
k-member ensemble's or BALD's top-k ranking is NOT BatchBALD.

For this protocol, defer full BatchBALD rather than build an unvalidated
approximation or change batch size. A later targeted feasibility study can
compare accurate joint-entropy estimates for small batches, then evaluate a
validated large-batch approximation only if BALD is promising.

## Installation

From repository root on Delta:

    unzip -n /projects/bcrv/dkourtesis/task12_bald_mc20_v1.zip -d .
    conda activate al-pipeline
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
    python -m pytest tests/jobs/test_task12_bald.py tests/jobs/test_task12_protocol.py -q
    python scripts/research/historical_replay/task12/task12.py plan
    python scripts/research/historical_replay/task12/task12.py prepare
    python scripts/research/historical_replay/task12/task12.py verify
    python scripts/research/historical_replay/task12/task12.py preflight

Task 12 freezes a patched copy of the verified ORIGINAL Task 4 source into a
new suite-specific directory. Repeated `prepare` verifies rather than edits.
It does not modify the Task 4, Task 9, Task 10 or Task 11 experiments.

## Real-model pilot: mandatory

Once prepared and preflight succeeds, queue **one GPU acquisition-only pilot**:

    JOB=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task12_bald_mc20_v1
    sbatch --parsable --job-name=T12_pilot_s135 \
      --output="$JOB/logs/pilot_%j.out" --error="$JOB/logs/pilot_%j.err" \
      --time=00:45:00 \
      "$JOB/source/scripts/research/historical_replay/task12/task12_worker.sbatch" "$JOB" pilot 135

This loads the seed-135 L0 checkpoint and performs T=20 coherent MC passes
on U0. It compares the 473-object selection to Task 4 entropy and records
runtime, number of active dropout modules, mean/median/max MI, fraction of
nonzero scores and matching L0/U0 provenance. There is **no training**.
If this fails due to zero dropout or excessive runtime, STOP; do not queue
27 fits or alter frozen code.

    python scripts/research/historical_replay/task12/task12.py pilot-status

## Submission after Task 11

After the pilot passes and the full Task 11 81-fit suite has been queued, run:

    python scripts/research/historical_replay/task12/task12.py submit --walltime 01:30:00

The launcher requires proof that Task 11 has completed all nine trajectories
OR reads its submitted-job manifest and binds `afterok` dependencies to all
nine **terminal Task 11 jobs**. Three separate nine-cycle BALD chains then
run in dependency order. If Task 11 has not yet been submitted, Task 12
explicitly refuses to queue; it does not pretend to schedule future work.
Submission records every job ID and refuses accidental double submission.
A failed Task 11 terminal job prevents BALD from proceeding.

Monitor with:

    python scripts/research/historical_replay/task12/task12.py status
    squeue -u "$USER"

**Walltime must be reevaluated after the pilot.** BALD runs a separate
stochastic inference stage in addition to the historical deterministic
inference, training and validation evaluation. The pilot's MC inference
runtime does NOT measure full-cycle runtime. Use a safe walltime based on
actual Task 11 cycle timings and BALD pilot. One worker occupies one A40.

## Reporting after 27 fits

    python scripts/research/historical_replay/task12/task12_report.py

The reporter produces validation trajectories, per-cycle paired comparisons
to Task 4 random/entropy/entropy-diversity/entropy-balanced-explore,
per-class precision/recall/F1/TP/FP/FN, acquisition overlap to same-seed
Task 4 entropy, and descriptive learning-curve summaries. It does not read
the test split. Primary endpoints are validation macro-F1 and balanced
accuracy at matched object-label budgets and learning-curve area, with TDE
false alerts and SN I/SN II tradeoffs reported separately.

## References

- Houlsby et al., *Bayesian Active Learning for Classification and Preference Learning*, 2011. https://arxiv.org/abs/1112.5745
- Gal and Ghahramani, *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, ICML 2016. https://arxiv.org/abs/1506.02142
- Kirsch, van Amersfoort, Gal, *BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning*, NeurIPS 2019. https://proceedings.neurips.cc/paper/2019/hash/95323660ed2124450caaac2c46b5ed90-Abstract.html
- BatchBALD Redux documentation, especially consistent MC dropout. https://blackhc.github.io/batchbald_redux/
