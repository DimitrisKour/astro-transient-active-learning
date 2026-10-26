# AppleCiDEr Task 8C.1 — retention-only closed loop (v1)

## Research question

Does retaining the last accepted checkpoint after a substantial within-cycle validation macro-F1 regression improve **subsequent** active-learning trajectories, or does it impede representation improvement? This is a **nine-cycle, six-trajectory** experiment compared to original Task 4. No retry seeds; no temperature scaling; no ensemble-based inference. The existing Task 4 experiments are immutable.

## Model/acceptance state machine

At cycle `t`:

1. Infer over the remaining `U_t` using the *active incumbent* checkpoint `M*_{t-1}`.
2. Select `A_t` with unchanged original entropy or entropy_balanced_explore acquisition, reveal historical labels, append to cumulative `L_t = L_{t-1} ∪ A_t`.
3. Train the ViT candidate `M_t` from `M*_{t-1}` using **all** `L_t`: Task 4 weighted-validation-loss-best, exponent **0.85**, per-sample objective, 50 epochs/patience 10, frozen experts, original seed, fresh optimizer and scheduler.
4. On validation objects, compute `Δ macro-F1 = macro-F1(M_t) − macro-F1(M*_{t-1})`, aligning unique object IDs and verifying their targets.
5. **Accept** when `Δ macro-F1 >= −0.01`. Otherwise **reject** and retain `M*_{t-1}`. `L_t` is **never** rolled back. At `t+1`, both inference and warm-start use the accepted/retained `M*_t`.
6. Every cycle records the candidate, the active incumbent, and full per-class validation differences. It never promotes a rejected candidate as the next training initializer.

### Shadow rule: recorded, never executed

Independent warnings: (a) macro-F1 loss >1 pp; (b) balanced accuracy loss >3 pp; (c) net class-correct losses >= **AGN 35, TDE 3, SN I 100, SN II 30, CV 10**. *Any* violated guard would flag the candidate in the hypothetical stricter rule. Shadow decisions **do not** determine model retention, do not modify acquisition, and cannot predict the stricter rule's counterfactual later trajectory. The class thresholds are exploratory engineering heuristics, not statistically validated effects.

## Reuse of original Task 4 without leakage

The six first rejections from the frozen Task 8A 1-pp rule are:

| Policy | Seed 135 | Seed 246 | Seed 357 |
|---|---:|---:|---:|
| entropy | cycle 4 | cycle 4 | cycle 5 |
| entropy_balanced_explore | cycle 4 | cycle 3 | cycle 6 |

Cycles **1 through the first rejected cycle** are replayed from original Task 4 source artifacts, *not retrained*. The replay reconstructs the new experiment's own cumulative manifest, checks the identical manifest SHA against the original training summary, verifies checkpoint ancestry and evaluation SHA, copies original selection and inference artifacts into the new suite, writes independent evaluation history and checkpoint decisions, and refuses to train in the CPU phase. The six reference first-rejection cycles must agree with the actual recomputed 1-pp decisions.

Thus the new GPU portion requires **28** training cycles, instead of 54, across six dependent chains. Once a rejection occurs, no later original Task 4 checkpoint or acquisition is recycled. A source checkpoint may still be an *incumbent* after rejection, but the continuation is recomputed with newly selected batches.

## Storage, immutability, and restart

- Source Task 4 frozen code: `/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_v1`.
- Original reference runs: `/projects/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_closed_loop_warm_v1` and `/work/hdd/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_closed_loop_warm_v1`.
- Task 8A first rejections: `/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task8a_retrospective_acceptance_v1/first_rejections.csv`.
- New **frozen source, plan, SLURM logs**: `/work/hdd/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task8c1_retention_macro1pp_v1`.
- New independent AL run state, selections, and heavy training artifacts: `/work/hdd/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_task8c1_retention_macro1pp_v1`.

Preparing the experiment snapshots the **actual frozen Task 4 source**, overlays only 8C.1-specific modules/scripts, and routes only the *new* registry storage roots to HDD. Original training/evaluation modules remain unchanged. Hash mismatches fail closed. Interrupted completed cycles with a summary but no retention ledger record can reconstruct that decision from saved validation predictions. Incomplete model training directories are not overwritten or silently restarted: investigate any walltime failures before rerunning.

**Note:** source experiment `evaluation_summary.json` files can include both validation and test records, but the replay explicitly accesses **only** the validation split and validation NPZ. New evaluation also runs **validation only**. The test set is not used for either checkpoint selection or acceptance. Because the same validation set is repeatedly used to select within-cycle epochs and retention decisions, validation curves can be optimistic; this is an exploratory workflow, not a statistically independent estimate of deployment performance.

## Installation

From PowerShell, send the zip to `/projects/bcrv/dkourtesis/` using your normal Delta `scp`. On Delta:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
conda activate al-pipeline
unzip -n /projects/bcrv/dkourtesis/task8c1_retention_only_v1.zip -d .
python -m pytest tests/core/test_retention_gate_v1.py tests/jobs/test_task8c1_preparation.py tests/jobs/test_task8c1_retention_decisions.py -q
```

Expected **20 passing synthetic/local tests**; these are not a substitute for real Delta preflight/source validation.

## Review and prepare — no training, no submission

```bash
python scripts/adaptive/task8c1.py plan
python scripts/adaptive/task8c1.py prepare
python scripts/adaptive/task8c1.py verify
```

`prepare` freezes all sources under HDD. Rerunning it with identical inputs verifies the snapshot. It never submits or runs SLURM jobs.

## Phase I — CPU read-only replay

```bash
python scripts/adaptive/task8c1.py submit-prefill --walltime 00:20:00
squeue -u "$USER"
python scripts/adaptive/task8c1.py status
```

Exactly **six independent CPU-array tasks** replay the first-rejection prefix. The CPU code aborts rather than training if the observed first-rejection cycle disagrees with Task 8A. After they complete, `status` should show first rejection **4/4/5** for entropy and **4/3/6** for balanced exploration and matching `decisions` counts. Inspect `closed_loop/.../logs/T8C1_pf_*.out` and `*.err` if any case failed. **Do not submit GPU jobs until all six prefixes verify.**

## Phase II — GPU continuation

```bash
python scripts/adaptive/task8c1.py submit --walltime 00:20:00
squeue -u "$USER" -o "%.18i %.35j %.2t %.10M %R"
python scripts/adaptive/task8c1.py status
```

`submit` checks all six prefill states and queues **28 GPU jobs** in six per-trajectory `afterok` dependency chains, held until all submissions succeed. If submission fails, it attempts to cancel the queued jobs and preserves failure markers requiring manual inspection. Every job performs exactly one newly divergent cycle. The default walltime is 20 minutes, but you can adjust the flag if needed before submission. No Task 4 training is duplicated.

## Read-only report and upload

```bash
OUT=/work/hdd/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task8c1_retention_macro1pp_v1/report
python scripts/adaptive/task8c1_report.py --out "$OUT"
cat "$OUT/summary.json"
tar -czf /work/hdd/bcrv/dkourtesis/applecider_adaptive/task8c1_review_v1.tar.gz \
  -C "$OUT" summary.json decisions.csv class_recall.csv acquisition_overlap.csv trajectories.csv
```

Retrieve that tarball to your workstation and upload it for scientific comparison with original Task 4, preserving the distinction between **candidate validation** and **deployed incumbent validation** after rejection.

### Diagnostic focus

- Number and location of rejections, including consecutive rejections that strand an old incumbent.
- Effects on future acquisition overlap and class composition.
- Candidate versus active incumbent performance (different after rejection).
- Macro-F1, balanced accuracy, and net correctly classified objects per class, particularly SN I and SN II.
- Shadow BA/recall warnings and shadow/operational disagreements.
- Compute expense, premature saturation, and downstream learning after rejection.

### Limitations

This pilot uses a provisional validation-informed rule and only three original seeds. The effects on full trajectories need comparison against Task 4 at identical labeled-object budgets; it does **not** claim guaranteed gains. Task 8C.2 can introduce seed retries under the **same** acceptance rule, isolating the intervention, while stricter shadow rules should receive their own follow-on policy-controlled comparison instead of changing both rule and retries simultaneously.
