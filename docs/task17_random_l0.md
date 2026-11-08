# Task 17 — Uniform-random L0 sensitivity, frozen AstroMiNN replay

This is an **isolated new historical replay**, not a resumption or replacement of Tasks 4 or 11–16. Source and registry are frozen under:

`/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task17_random_l0_v1`

There are **three new bootstrap fits**, followed by **18 AL fits** (random and entropy × seeds 135, 246, 357 × three cycles). All Slurm jobs are submitted **held**; releasing them is a separate explicit action. The six AL series have independent first cycles; later cycles use only the preceding cycle's `afterok` dependency.

## Scientific protocol

- Reference: completed Task 4 stratified L0 / full-pool random and entropy trajectories.
- Alternative: exactly **476 historical training objects chosen uniformly at random without replacement** from all 4,729 distinct training objects. The draw is **not class stratified** and does **not** reject draws missing TDE or another class.
- All model-training seeds use the **same new L0 and U0**, chosen with fixed split seed `17017` (and a second deterministic RNG namespace value `17`). This isolates training-seed variation from split variation.
- The new **4,253-object U0** contains every historical training object not in the new L0. It is not necessarily the same list as Task 4 U0, although L0+U0 is exactly the same original historical population.
- 10% of all historical training objects per acquisition cycle: **473** selected per cycle for **3 cycles**, acquiring 1,419 of 4,253 U0 objects (1,895 total labeled including L0). This follows the originally agreed compact Task17 sensitivity-study budget; **Task16**, not Task17, is the full-pool cadence experiment.
- Initial model: **fresh random ViT bootstrap trained on the newly sampled L0**, separately for model seeds 135, 246, and 357. Do not reuse Task 4 checkpoints: those learned from a different L0.
- Subsequently train using **warm-start from the previous cycle**, same epoch cap/patience/optimizer and frozen expert towers as Task 4. Validation set unchanged, test never used for selection.
- The legacy trainer normally requires nonzero counts in every class. The **Task17-only** frozen trainer patch handles zero-count classes with `max(count, 1)` **only when computing class weights**. Actual observed training counts remain unchanged and missing classes are explicitly reported; no class is manufactured, backfilled, or conditionally sampled.
- This comparison varies L0 composition and therefore the bootstrap initialization, available U0 members and subsequent trajectories. It does **not** isolate random row order, and the result of one random L0 should not be generalized as an ensemble of random splits.

## 1. Transfer from Windows

Download `task17_random_l0_overlay_v1.zip` to your local Windows `Downloads` folder. In **Windows PowerShell**, not the Delta terminal:

```powershell
scp "$HOME\Downloads\task17_random_l0_overlay_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

## 2. Extract and activate on Delta

```bash
cd /projects/bcrv/dkourtesis
mkdir -p task17_random_l0_overlay_v1
unzip -n task17_random_l0_overlay_v1.zip -d task17_random_l0_overlay_v1
cd task17_random_l0_overlay_v1

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
```

## 3. Prepare, verify, CPU tests, preflight (no training)

```bash
python scripts/research/historical_replay/task17/task17.py plan
python scripts/research/historical_replay/task17/task17.py prepare
python scripts/research/historical_replay/task17/task17.py verify
T17=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task17_random_l0_v1
PYTHONPATH="$T17/source/src" python -m pytest -q tests/test_task17_random_l0.py
python scripts/research/historical_replay/task17/task17.py preflight
cat "$T17/preflight_audit.json"
```

**Expected:** the split audit confirms 476/4253 objects, the historical population matches Task 4, but the L0 differs. Inspect the **actual L0 class counts**, especially TDE. Do not resample because the class balance is unappealing: the unstratified draw is the scientific intervention.

On Delta, the test cases that require `/mnt/data/direct_parent` (a local development fixture) skip intentionally. The real source patch is checked by `prepare` and `verify`. CPU tests don't prove full GPU training works.

## 4. Create new L0 bootstraps (held initially)

```bash
python scripts/research/historical_replay/task17/task17.py submit-bootstrap --walltime 01:00:00
```

This registers **three independent HELD Slurm jobs** and does not run them. Inspect:

```bash
cat "$T17/bootstrap_jobs.json"
squeue -u "$USER" -o '%.18i %.30j %.2t %.10l %R'
```

Release **only seed 135** first:

```bash
python scripts/research/historical_replay/task17/task17.py release-bootstrap --seed 135
```

Once that job completes successfully, inspect its Slurm log and checkpoint. Additional bootstraps can be released independently:

```bash
python scripts/research/historical_replay/task17/task17.py release-bootstrap --seed 246
python scripts/research/historical_replay/task17/task17.py release-bootstrap --seed 357
```

Do not manually rerun existing completed bootstrap jobs; the runner's checkpoint validation checks for compatible training artifacts.

## 5. One acquisition-only GPU pilot after seed 135 bootstrap

Once the seed 135 bootstrap is complete, submit an **acquisition-only** pilot. This does **not** reveal labels or train a model:

```bash
sbatch --parsable \
  --time=00:20:00 \
  --job-name=T17_pilot_135 \
  --output="$T17/logs/%x_%j.out" \
  --error="$T17/logs/%x_%j.err" \
  "$T17/source/scripts/research/historical_replay/task17/task17_worker.sbatch" \
  "$T17" pilot 135
```

Check:

```bash
python scripts/research/historical_replay/task17/task17.py pilot-status
sacct -u "$USER" --starttime 2026-10-30 -X \
  --format=JobID,JobName%32,State,ExitCode,Elapsed | grep 'T17_'
```

The pilot validates real inference for the random-L0 checkpoint on the new U0 and first-cycle uniqueness for both policies.

## 6. Submit full AL chains ONLY after successful bootstrap verification and pilot

Each policy/seed trajectory requires its **own seed's completed bootstrap**. The submission command checks that. To submit one chain held:

```bash
python scripts/research/historical_replay/task17/task17.py submit-cycles --walltime 00:35:00 --policy entropy --seed 135
```

To submit the remaining chains (after other bootstraps finish):

```bash
python scripts/research/historical_replay/task17/task17.py submit-cycles --walltime 00:35:00
```

The second call skips already submitted chains. **No AL job is automatically released.** Inspect the ledger:

```bash
cat "$T17/submitted.json"
```

Release a single first-cycle integration test:

```bash
python scripts/research/historical_replay/task17/task17.py release-cycles --policy entropy --seed 135 --cycle 1
```

Once that cycle has a validated checkpoint and `run_state.json`/`cycle_summary.json`, optionally release all remaining held jobs (they remain ordered by `afterok`):

```bash
python scripts/research/historical_replay/task17/task17.py release-cycles --all
```

The manager doesn't create dependencies on Tasks 11–16. Reduce walltime only after measuring actual Task17 timings; Delta has recently had sporadic filesystem slowdowns.

## 7. Results and recovery

```bash
python scripts/research/historical_replay/task17/task17.py status
python scripts/research/historical_replay/task17/task17.py collect --destination /u/dkourtesis/task17_random_l0_results.zip
```

PowerShell:

```powershell
scp dkourtesis@dt-login.delta.ncsa.illinois.edu:/u/dkourtesis/task17_random_l0_results.zip "$HOME\Downloads\"
```

The `collect` output includes run histories, selections and labels, L0 initial manifest, split provenance, and available pilot/Slurm metadata (not heavy checkpoints). Compare validation macro-F1 and balanced accuracy against Task 4 at the **same number of acquired objects** and inspect rare-class recall at cycle zero and later cycles. Acquired labels may be revealed only after selection in the simulated experiment.
