# Task 8C.2 — four-seed bounded-retry retention closed loop

## Frozen scientific contract

Reference: original Task 4 `astrominn_closed_loop_warm_v1`. New isolated suite:
`astrominn_task8c2_retry4_macro1pp_v1`, on HDD `/work/hdd/bcrv/dkourtesis/applecider_adaptive`.

Six trajectories: strategies `entropy`, `entropy_balanced_explore` × seeds `135`, `246`, `357`. Nine cycles. Acquire 473 objects in cycles 1–8 and the 469 remaining in cycle 9. Original Task 4 `.85` weighted-loss-best selection, per-sample weighting, 50 epochs, patience 10, frozen expert towers, warm start from the **last accepted checkpoint**, same original L0, pool U0, validation split, holdout and hidden-label oracle. The training loop and the class order (`AGN`, `TDE`, `SN I`, `SN II`, `CV`) are unmodified.

At each acquisition cycle, first train one standard candidate on the entire expanded cumulative `L_t`. Compare its *object-level validation macro-F1* against the active incumbent. Accept if `candidate_macro_f1 >= incumbent_macro_f1 - 0.01`. If rejected, train new models **from the same incumbent** and **on exactly the same `L_t`**, with `(process_seed, loader_seed)` set successively to `1001`, `1002`, `1003`, `1004`. Accept the **first** passing retry and skip the remaining retries. If all four fail, retain the incumbent. In all cases, keep all acquired objects/labels and advance to `L_{t+1}`. NO best-of-four cherry-picking; NO acquisition between retries.

3 pp BA loss and per-class net correctly classified object losses (AGN 35, TDE 3, SN I 100, SN II 30, CV 10) are recorded **only in shadow mode**. They never control the active decision. Neither test predictions nor test metrics enter acceptance or training decisions; evaluation during the experiment is validation-only. Validation was also used for within-cycle epoch selection, thus these are exploratory results.

## Design and provenance

- The original Task 4 repository source snapshot and `registry.toml` are SHA256 verified. Only the registry `[storage]` roots are routed to HDD; all dataset and profile TOML fields are preserved.
- A source/plan snapshot is created at `HDD/closed_loop/<suite>/source` and verified before every scheduled stage. Installation modifies the worktree, never the Task 4 reference or Task 8C.1 outputs.
- The **CPU prefill phase** replays the 26 historical Task 4 cycles *through and including* each trajectory's first rejected candidate. It does NOT finalize those six first rejected candidates: their retry decisions are pending. This stage must never train.
- The **GPU continuation phase** therefore schedules 28 new base-candidate slots, plus at most 136 conditional retry slots. All 164 Slurm jobs can be prequeued with `afterok` dependencies, but retry jobs after an acceptance perform provenance checks and exit without training. There is still scheduler overhead and brief GPU allocations for these no-op stages. This is deliberately a resource-safety/completion trade-off; `--case-index` allows rollout one trajectory at a time.
- Each base cycle, retry attempt and final decision is persisted separately. Only a finalized incumbent can perform the next acquisition. State can resume after an interrupted evaluation using completed training records; **partial training directories are never silently deleted or overwritten**. A timed-out fit with partial training artifacts requires inspection, and dependent jobs will not proceed automatically.
- This implementation deliberately **does not automatically import Task 8B fit checkpoints**. Although those are potentially reusable at the same first-rejection manifests, they have different run identifiers/provenance contracts, and automatic mixing would require a separate explicit equivalence audit. The six *base rejected checkpoints* are reused read-only from Task 4.
- The original 1 pp acceptance rule is unchanged; the only experimental intervention relative to 8C.1 is the retry policy.

## Installation

From Windows PowerShell (after downloading archive):

```powershell
scp "$HOME\Downloads\task8c2_bounded_retries_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

On Delta:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
conda activate al-pipeline
unzip -n /projects/bcrv/dkourtesis/task8c2_bounded_retries_v1.zip -d .
python -m pytest tests/jobs/test_task8c2_schedule.py tests/jobs/test_task8c2_retry_gate.py -q
python scripts/research/historical_replay/task8c2/task8c2.py plan
python scripts/research/historical_replay/task8c2/task8c2.py prepare
python scripts/research/historical_replay/task8c2/task8c2.py verify
```

No jobs are submitted by the commands above. A preparation failure does not imply GPUs have been scheduled.

## Step 1: CPU prefill

```bash
python scripts/research/historical_replay/task8c2/task8c2.py submit-prefill --walltime 00:20:00
squeue -u "$USER"
python scripts/research/historical_replay/task8c2/task8c2.py status
```

Expect `completed` equal to `4,4,5,4,3,6`, but `decisions` equal to `3,3,4,3,2,5` respectively, with each first-rejected base candidate pending retries. Check Slurm `sacct` first: the six CPU jobs must be `COMPLETED` with exit code `0:0`. The CPU worker includes the Conda `libstdc++` compatibility environment fix that 8C.1's first CPU submission lacked.

## Step 2: GPU continuation

Only after CPU prefill is verified. Recommended: queue **one** case first (index 0 = entropy seed 135), inspect successful first retry, then queue remaining cases. Jobs are held until all stages in each case are registered, then released in reverse dependency order.

```bash
python scripts/research/historical_replay/task8c2/task8c2.py submit --case-index 0 --walltime 00:20:00
squeue -u "$USER" -o '%.18i %.32j %.2t %.10M %R'
python scripts/research/historical_replay/task8c2/task8c2.py status
```

After confirming the first trajectory's staged jobs behave correctly, submit other **unsubmitted** cases individually:

```bash
for i in 1 2 3 4 5; do
  python scripts/research/historical_replay/task8c2/task8c2.py submit --case-index "$i" --walltime 00:20:00 || break
done
```

Alternatively, before any GPU case has been queued, `python scripts/research/historical_replay/task8c2/task8c2.py submit --walltime 00:20:00` queues all six chains at once. **Do not repeat** `submit` for already-submitted cases; the submission markers are designed to refuse duplicates. Job names have the form `T8C2_<case-index>_c<cycle>_b` or `_r1.._r4`. On a failed stage, `afterok` stops dependent stages; inspect stdout/stderr and persisted `training_summary.json`/retry markers before attempting recovery.

Status and complete report:

```bash
python scripts/research/historical_replay/task8c2/task8c2.py status
python scripts/research/historical_replay/task8c2/task8c2_report.py
```

If all 6 trajectories completed, archive the entire report rather than cherry-picking metrics:

```bash
D=/work/hdd/bcrv/dkourtesis/applecider_adaptive
R="$D/closed_loop/astrominn_task8c2_retry4_macro1pp_v1/report"
tar -czf "$D/task8c2_review_v1.tar.gz" -C "$R" .
```

The report contains `summary.json`, `decisions.csv`, `retry_attempts.csv`, `trajectories.csv`, `class_recall.csv`, `acquisition_overlap.csv`, `paired_task4.csv`. Compare both **base** and **active** checkpoint metrics against Task 4 and 8C.1 at equal budgets; distinguish performance improvement from selection/retention uplift. Compare actual retry fit counts and compute time rather than nominal job submissions. Review SN I, SN II, TDE and CV correct-object counts alongside macro-F1 and BA.
