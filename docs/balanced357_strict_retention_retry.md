# Balanced exploration seed 357 — strict checkpoint retention + four retries

Two **independent** single-trajectory experiments branch from the first **negative macro-F1 change** in the original Task 4 balanced-exploration seed-357 history. V2 supersedes the prepared V1 suites without modifying their snapshots.

- **Row order:** `astrominn_balanced357_strict_row_order_retention_retry_v2`
- **Training seeds:** `astrominn_balanced357_strict_seed_retention_retry_v2`

Both use policy `entropy_balanced_explore`, acquisition seed **357**, acquisition size **473** per cycle, complete through cycle **9**, warm-start from the active incumbent, class-weight exponent **0.85**, frozen expert towers, 50 epochs and patience 10. These controls come from Task 8C.2 and are verified against Task 4. The original Task 4 artifacts are immutable. No test-set evaluation is used for checkpoint decisions.

## Scientific state machine

1. Read the **saved Task 4 object-level validation history** and discover the first cycle with **Δ macro-F1 < 0** relative to the previous historical checkpoint, regardless of balanced-accuracy change. Do not hardcode cycle 4 or 6.
2. CPU-replay Task 4 cycles 1 through the first rejected cycle, with source checkpoint and manifest provenance checks. Reuse the first rejected standard candidate as **attempt 0** without retraining.
3. In the diverged closed loop, a base candidate triggers retries **only if its macro-F1 decreases** relative to the incumbent. Run **up to four independent retries**, sequentially, all on the *same* cumulative manifest and the *same* incumbent checkpoint. Stop at the **first passing retry**; do not pick the maximum of all four.
4. A **retry** is accepted **only if** validation object-level macro-F1 **and** validation balanced accuracy both strictly exceed the incumbent values; ties fail. A standard base candidate is promoted under the same strict two-metric rule **after divergence**. If macro-F1 is unchanged or increases but BA fails to strictly improve, **retain the incumbent without trying alternatives**. Historical reference checkpoints preceding the first macro-F1 regression are replayed intact, even if BA previously declined. Training's original weighted-loss-best epoch selection is unchanged.
5. If all four retries fail, keep the incumbent checkpoint (for next inference **and** next warm start). Do **not** stop, roll back labels, or repeat acquisition. All acquired objects remain in cumulative `L_t`. Resume with the next acquisition cycle.
6. For every later cycle after divergence, first train one standard-order, seed-357 candidate. If it exhibits negative macro-F1 change, retry as above. If macro-F1 is nonnegative but BA does not strictly improve, retain without retries. Once a decision is finalized, subsequent acquisition uses the **retained or accepted** checkpoint.

The sole intervention difference:
- **row_order** changes only the order of training rows using independent deterministic permutation seeds `41004, 42004, 43004, 44004` plus `100*cycle`. Both process and loader seed stay **357**. The CSV row multiset (including duplicate samples) is verified unchanged.
- **seed** keeps the exact original training manifest order and changes both the process seed and the loader seed to `1001, 1002, 1003, 1004`, as in Task 8C.2. Acquisition seed remains **357**.

A retry fit is logged with its initial-checkpoint SHA256, cumulative manifest SHA256, actual training-manifest SHA256, retry mode, permutation/seed, checkpoint SHA256, evaluation SHA256 and selected decision. Scripts are designed to reject unexpected partial training outputs and identity mismatch rather than overwrite.

## Local repository checkout on Delta

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
git fetch origin
git switch feat/balanced357-retention-retry
git pull --ff-only
conda activate al-pipeline

python -m pytest tests/jobs/test_balanced357_strict_retry.py -q
```

These tests are CPU-only. They cover strict two-metric decisions, source first-failure discovery, row-permutation multiplicities, one-case stage counts, and the basic continuation contract. **The local/Delta tests and real-data preparation have not yet been run by the assistant.**

## Prepare: no Slurm submission

Run for each independent variant:

```bash
python scripts/adaptive/balanced357_row_order_retry.py plan
python scripts/adaptive/balanced357_row_order_retry.py prepare
python scripts/adaptive/balanced357_row_order_retry.py verify

python scripts/adaptive/balanced357_seed_retry.py plan
python scripts/adaptive/balanced357_seed_retry.py prepare
python scripts/adaptive/balanced357_seed_retry.py verify
```

`plan` validates Task 4 source and computes the first strict failure. `prepare` freezes an isolated Task 4 scientific-source copy plus the new adapters under `/work/hdd/bcrv/dkourtesis/applecider_adaptive/closed_loop/<suite>` with SHA256 identities. It routes only registry storage roots to HDD. `verify` checks frozen source/plan/input provenance. Original Task 4 and other experiments are not overwritten.

## CPU prefill: read-only replay, one short job per variant

Only after successful plan/prepare/verify and tests:

```bash
python scripts/adaptive/balanced357_row_order_retry.py submit-prefill --walltime 00:20:00
python scripts/adaptive/balanced357_seed_retry.py submit-prefill --walltime 00:20:00
```

Each submission creates a one-element CPU array. Wait for `sacct` to show `COMPLETED 0:0` and inspect both statuses:

```bash
python scripts/adaptive/balanced357_row_order_retry.py status
python scripts/adaptive/balanced357_seed_retry.py status
```

Require exactly the discovered first-rejection cycle completed, its predecessor decisions accepted and reference-reused, and its first rejected base candidate persisted **without** a final decision. Do not submit GPUs if the prefix diverges from saved Task 4.

## GPU continuation: separately controlled chains

**GPU submission is intentionally a separate explicit command.** Each variant submits its own fixed-length dependency chain; the first-rejection cycle has at most four retry stages (attempt 0 reused), and every following cycle has one base stage and four conditional retry slots. Already-finalized stages skip training; do not interpret job count as fit count.

```bash
python scripts/adaptive/balanced357_row_order_retry.py submit --walltime 00:20:00
python scripts/adaptive/balanced357_seed_retry.py submit --walltime 00:20:00
```

Do not issue `submit` twice for an already registered case. Job submissions are held during registration and then released in dependency-safe reverse order. A nonzero exit stops downstream `afterok` scheduling. A scientifically rejected checkpoint is **not** a process error and must progress to the next retry or next cycle.

Inspect:

```bash
squeue -u "$USER" -o "%.18i %.32j %.12T %.10M %.20R"
python scripts/adaptive/balanced357_row_order_retry.py status
python scripts/adaptive/balanced357_seed_retry.py status
```

If the first historical strict failure is at cycle `f`, the maximum *conditional Slurm stage slots* per variant are `4 + 5*(9-f)`. Fewer than that number should actually train when a base candidate or early retry succeeds. Training logs and source snapshots remain separate.

## Interpretation

Report **base candidate**, **retry candidate** and **deployed incumbent** macro-F1/BA by cycle, accepted attempt number, class-level recall/correct counts, acquired-label overlap with Task 4, retry fits used and training duration. Compare the two variants at identical cumulative label budget and first divergence. Validation controls retention and is also used for within-cycle epoch selection; final generalization claims require independent evaluation.

**Important:** The row-order-only treatment changes the presentation order in the trainer's CSV, rather than reseeding the trainer. The seed-only treatment changes trainer/loader seeds while preserving the CSV order. Because successful retries cause later acquisition to diverge, later training manifests may differ *between the two full-loop trajectories*, as intended.

## V2 correction (November 2026)

The previous V1 implementation wrongly used strict improvement in **both** metrics as the trigger for the first rejected checkpoint, selecting balanced357 cycle 4. In the observed Task4 history, cycle 4 macro-F1 increased from 0.645389 to 0.653365 while balanced accuracy declined from 0.740967 to 0.711512. Under the corrected trigger, this is **not** the first macro-F1 regression. The V2 controllers discover the correct starting cycle from source history and use fresh suite names. **Do not schedule or reuse V1 prefills or V1 GPU jobs in V2.**
