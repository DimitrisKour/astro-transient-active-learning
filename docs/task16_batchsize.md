# Task 16: full-pool acquisition cadence (5%, 10%, 20%)

## Scientific question

How does **the amount acquired per retraining cycle** affect validation learning
curves, class recalls, and final performance? Every trajectory starts from
exactly the same Task 4 L0 and seed-specific bootstrap and acquires **all U0**.
The 10% trajectories are reused **without retraining** from Task 4.

| Schedule | Regular batch | Cycles | Last batch | New jobs (2 policies × 3 seeds) |
|---|---:|---:|---:|---:|
| 5% of 4,729 historical objects | 237 | 18 | 224 | 108 |
| 10% existing Task 4 | 473 | 9 | 469 | 0 |
| 20% of 4,729 historical objects | 946 | 5 | 469 | 30 |

**Total new training jobs: 138.** Policies: `random`, `entropy`; seeds 135,
246, 357. For each trajectory, cycle `t+1` depends on successful completion
of cycle `t`, with no cross-policy, cross-seed, cross-fraction or Task 11–15
Slurm scheduling dependencies. Acquisition percentages refer to the entire
historical train population including L0, not the smaller U0.

## Why these two policies?

Random is a control for the number of retraining steps independent of an
uncertainty score. Entropy tests the interaction between retraining cadence
and model-driven pool querying. Existing random and entropy Task 4 runs
serve as the 10% reference.

**Causal caution:** This comparison changes the number of warm-start fits,
optimizer passes, and opportunities for model updates, not only the size of
individual acquisition batches. It must not be described as compute-matched.
Compare object-level validation metrics against cumulative *acquired object*
counts / fraction, **not** cycle numbers. Report final-cycle macro-F1 and
balanced accuracy, AULC to common acquisition fractions, classwise recall,
training wall-clock/epoch counts, and seed variability. No test metrics used
for choosing the best schedule.

## Deployment to Delta

From **local Windows PowerShell** (after downloading the ZIP):

```powershell
scp "$HOME\Downloads\task16_batchsize_overlay_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

From **Delta terminal**:

```bash
cd /projects/bcrv/dkourtesis
mkdir -p task16_batchsize_overlay_v1
unzip -n task16_batchsize_overlay_v1.zip -d task16_batchsize_overlay_v1
cd task16_batchsize_overlay_v1
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
python scripts/adaptive/task16.py plan
python scripts/adaptive/task16.py prepare
python scripts/adaptive/task16.py verify
TASK16_JOB=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task16_batchsize_sensitivity_v1
PYTHONPATH="$TASK16_JOB/source/src" python -m pytest -q tests/test_task16_batchsize.py
python scripts/adaptive/task16.py preflight
```

Preflight validates compatibility for all seeds in both suites, confirms L0
and ordered U0 match Task 4, and validates all three bootstrap checkpoints
without GPU inference or training.

### Pilot: one GPU inference, four acquisition-only checks

```bash
sbatch --parsable \
  --time=00:20:00 \
  --job-name=T16_pilot \
  --output="$TASK16_JOB/logs/%x_%j.out" \
  --error="$TASK16_JOB/logs/%x_%j.err" \
  "$TASK16_JOB/source/scripts/adaptive/task16_worker.sbatch" \
  "$TASK16_JOB" pilot _ _ 135
```

Once finished:

```bash
python scripts/adaptive/task16.py pilot-status
sacct -u "$USER" --starttime 2026-10-31 -X \
  --format=JobID,JobName%35,State,ExitCode,Elapsed | grep T16_
```

The pilot performs an actual seed-135 AstroMiNN U0 inference using the frozen
L0 checkpoint, then checks random and entropy acquisition at **both** new
budgets. It does not reveal hidden labels or train. A successful result gates
all 138 training jobs.

### Submit held jobs (NOT before successful pilot)

```bash
python scripts/adaptive/task16.py submit --walltime 00:35:00
python scripts/adaptive/task16.py status
```

**This creates held jobs only**, with no training released. Since there are
138 new jobs, you may instead submit complete independent chains in stages:

```bash
# 30 jobs (20%: two policies x three seeds x five cycles)
python scripts/adaptive/task16.py submit --walltime 00:35:00 --submit-fraction 20

# Example: 18 jobs (5%: one policy and one seed)
python scripts/adaptive/task16.py submit --walltime 00:35:00 \
  --submit-fraction 05 --submit-policy entropy --submit-seed 135
```

Further staged calls skip already submitted chains. The submission ledger is
persisted **after each complete chain**; failures attempt to cancel only the
incomplete chain, leaving earlier submissions recorded. This supports Delta
pending-job limits without modifying experiment results. Running the plain
`submit` command later fills in any remaining chains. Confirm the Slurm
pending-job limit before attempting a large batch.

Release the first `entropy` seed-135 5% job (or another first-cycle case):

```bash
python scripts/adaptive/task16.py release \
  --fraction 05 --policy entropy --seed 135 --cycle 1
```

After verifying checkpoint/evaluation/revealed-label state, and confirming
queue capacity, explicitly release the remaining jobs:

```bash
python scripts/adaptive/task16.py release --all
```

If Slurm rejects a submission, inspect the ledger and `squeue` before retrying.
The script uses recorded `(fraction, policy, seed)` cases to avoid duplicate chains.

### Result collection

```bash
python scripts/adaptive/task16.py status
python scripts/adaptive/task16.py collect --destination /u/dkourtesis/task16_results.zip
```

The collector includes both new schedules and the original Task 4 reference
metadata, without copying model weights or large image files.

From Windows PowerShell:

```powershell
scp dkourtesis@dt-login.delta.ncsa.illinois.edu:/u/dkourtesis/task16_results.zip "$HOME\Downloads\task16_results.zip"
```

## Frozen source and reproducibility

The overlay copies a verified, frozen Task 11 source tree into an independent
Task 16 snapshot. **The sole change to existing source is the runner policy
guard**, restricting this experiment to `random` and `entropy` (both already
implemented and tested in the frozen source). All other source and experiment
logic, including the Task 4 `--complete-pool` budget planner, remains frozen.
Only new Task 16 management/patch/worker files are added; all are hash-verified.

Suites are separate:

- `astrominn_task16_batch05_v1`
- `astrominn_task16_batch20_v1`

Task 4, 11, 12, 13, 14, and 15 artifacts are not modified.
