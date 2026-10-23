# Task 6A — independent-chain SLURM scheduler correction

**Reason:** The first package's `task6a.py submit` submits six-task arrays with `afterok:<previous_array_id>`. This is a **whole-array barrier**, not an independent per-seed/policy chain. It delays a finished trajectory until all other trajectories complete that cycle.

The new `task6a_submit_chains.py` instead submits 54 **one-task arrays**, each with its own SLURM job ID and only one `afterok` dependency: the previous-cycle job of that same seed and acquisition policy. Cycle 1 has no dependency. The frozen scientific worker is **unchanged**, so preparation already completed on Delta remains valid.

Without unrelated cross-chain dependencies or an external site resource cap, all six trajectories can run concurrently (up to six A40 GPUs). Jobs request one A40, 8 CPUs, 32 GiB each. There is **no global three-GPU cap**. If the user/account has fewer GPUs allocated or available, SLURM queues remaining tasks.

The script defaults to `00:20:00`, matching Task 4's late-cycle requests. `00:24:00` is allowed for safety (under 25 minutes); both are requested upper limits, not predicted compute time. Task 4 requested 15 minutes for early cycles, 20 minutes for later ones. Task 5B fits completed in roughly 1.3–8.4 minutes of training, but closed-loop fits additionally perform model inference and evaluation, so training-only figures are not sufficient proof of 20-minute safety.

## Installation, after Task 6A has already been prepared

Download `task6a_independent_chains_patch_v1.zip`; in Windows PowerShell:

```powershell
scp "$HOME\Downloads\task6a_independent_chains_patch_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

Delta:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/task6a_independent_chains_patch_v1.zip -d .
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/jobs/test_task6a_independent_chains.py -q
python scripts/adaptive/task6a.py verify
python scripts/adaptive/task6a_submit_chains.py plan --walltime 00:20:00
```

**Do not run `task6a.py prepare` again**, and **do not run `task6a.py submit`**; that is the old whole-array-barrier scheduler. Do not alter the frozen Task 6A source under `/projects/.../closed_loop/astrominn_closed_loop_warm_weight05_v1/source`.

After checking the plan, queue with:

```bash
python scripts/adaptive/task6a_submit_chains.py submit --walltime 00:20:00
```

To give every task a little extra walltime, use `00:24:00` instead for both `plan` and `submit`.

If you already submitted the original one-fit Task 6A probe, this scheduler requires it to be finished successfully and skips that completed cycle-1 fit (53 new jobs, not 54). It will never launch another cycle-1 fit over an unfinished probe.

The completed submission is recorded in `independent_submission.json`; it refuses duplicate submission. If `submitted.json`, `submission.json`, or `submission_pending.json` from the prior mechanism exists, stop and inspect rather than re-submit. The scheduler uses `--hold` and releases jobs after each per-trajectory chain is constructed, cancelling newly created jobs on setup failure.
