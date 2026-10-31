# AppleCiDEr — remaining Delta experiment queue reference

**Prepared 2026-10-31.** This is a command reference, not a record that jobs have been submitted. All example Slurm jobs request **at most 00:20:00**. Run commands on the **Delta login node** in the `al-pipeline` conda environment after completing each task's `prepare`, `verify`, and `preflight` steps. Do not run all submission blocks at once; first validate pilots, and consider Delta's pending-job limits.

```bash
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning-tasks14-17
P=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop
```

## Global status

```bash
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.10l %R"
sacct -u "$USER" --starttime 2026-10-28 -X --format=JobID,JobName%34,State,Elapsed,Timelimit,ExitCode
python scripts/adaptive/task13.py status
python scripts/adaptive/task14.py status
python scripts/adaptive/task15.py status
python scripts/adaptive/task16.py status
python scripts/adaptive/task17.py status
```

## Task 13 — TAILOR vs uniform (24 jobs = 2 methods × 3 seeds × 4 cycles)

Both acquisition-only pilots completed 2026-10-30. Confirm before submitting:

```bash
python scripts/adaptive/task13.py verify
python scripts/adaptive/task13.py pilot-status
python scripts/adaptive/task13.py submit --walltime 00:20:00
# The 24 jobs are HELD. Verify the ledger before release.
cat "$P/astrominn_task13_tailor_pilot_v1/submitted.json"
```

**Optional first-cycle-only release (recommended integration check):**

```bash
python - <<'PY'
import json, subprocess
from pathlib import Path
p=Path('/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task13_tailor_pilot_v1/submitted.json')
jobs=json.loads(p.read_text())['jobs']
for policy in ('tailor_diversity','uniform_mixture'):
    matches=[x for x in jobs if x['policy']==policy and x['seed']==135 and x['cycle']==1]
    assert len(matches)==1
    jid=str(matches[0]['job_id'])
    print('Releasing',policy,jid)
    subprocess.run(['scontrol','release',jid],check=True)
PY
```

After verifying both cycle-1 trajectories' checkpoints/evaluations, release remaining held jobs (within-chain `afterok` dependencies remain):

```bash
python scripts/adaptive/task13.py release
python scripts/adaptive/task13.py status
python scripts/adaptive/task13.py collect --destination /u/dkourtesis/task13_results_final.zip
```

`collect` refuses to overwrite an existing Task 13 archive: use a new filename on repeat.

## Task 14 — SIMILAR (81 jobs = 3 methods × 3 seeds × 9 cycles)

**Preconditions:** `prepare`, `verify`, `preflight` passed; all three acquisition-only GPU pilots passed. If pilots are not submitted:

```bash
python scripts/adaptive/task14.py verify
python scripts/adaptive/task14.py preflight
T14="$P/astrominn_task14_similar_facility_v1"
for policy in similar_fl similar_flcg similar_flqmi_rare; do
  sbatch --parsable --time=00:20:00 --job-name="T14_pilot_${policy}" \
    --output="$T14/logs/%x_%j.out" --error="$T14/logs/%x_%j.err" \
    "$T14/source/scripts/adaptive/task14_worker.sbatch" "$T14" pilot "$policy" 135
done
python scripts/adaptive/task14.py pilot-status
```

After successful pilots:

```bash
python scripts/adaptive/task14.py submit --walltime 00:20:00  # 81 HELD jobs
# Initial integration release:
python scripts/adaptive/task14.py release --policy similar_fl --seed 135 --cycle 1
# After verification of first cycle, release remaining chains:
python scripts/adaptive/task14.py release --all
python scripts/adaptive/task14.py status
python scripts/adaptive/task14.py collect --destination /u/dkourtesis/task14_results_final.zip
```

Task 14 submission creates all 81 held jobs in one call; check your current pending-job quota before issuing it.

## Task 15 — DIRECT (27 jobs = 3 seeds × 9 cycles)

**Preconditions:** Source tests, `verify` and preflight already passed; real DIRECT intra-cycle oracle pilot still needs validation.

```bash
python scripts/adaptive/task15.py verify
T15="$P/astrominn_task15_direct_vreduce_v1"
sbatch --parsable --time=00:20:00 --job-name=T15_pilot_direct \
  --output="$T15/logs/%x_%j.out" --error="$T15/logs/%x_%j.err" \
  "$T15/source/scripts/adaptive/task15_worker.sbatch" "$T15" pilot direct 135
python scripts/adaptive/task15.py pilot-status
```

After a successful pilot (inspect intra-cycle journal and acquisition budget):

```bash
python scripts/adaptive/task15.py submit --walltime 00:20:00 # 27 HELD jobs
python scripts/adaptive/task15.py release --policy direct --seed 135 --cycle 1
# After first end-to-end cycle succeeds:
python scripts/adaptive/task15.py release --all
python scripts/adaptive/task15.py status
python scripts/adaptive/task15.py collect --destination /u/dkourtesis/task15_results_final.zip
```

## Task 16 — full 5% and 20% acquisition schedules (138 new jobs)

The fraction keys are `05` and `20`, **not** `5`, `0.05`, `p05`, or `20%`. Six trajectories have 18 cycles (5%) and six have 5 cycles (20%). Existing 10% Task 4 results serve as reference.

**Preconditions:** `verify`, `preflight`, and the shared 5%/20% GPU acquisition pilot pass.

```bash
python scripts/adaptive/task16.py verify
T16="$P/astrominn_task16_batchsize_sensitivity_v1"
sbatch --parsable --time=00:20:00 --job-name=T16_pilot \
  --output="$T16/logs/%x_%j.out" --error="$T16/logs/%x_%j.err" \
  "$T16/source/scripts/adaptive/task16_worker.sbatch" "$T16" pilot _ _ 135
python scripts/adaptive/task16.py pilot-status
```

Submit in **stages** to avoid hitting the pending-job cap. Example for one full 5%/entropy/seed-135 chain:

```bash
python scripts/adaptive/task16.py submit --walltime 00:20:00 \
  --submit-fraction 05 --submit-policy entropy --submit-seed 135
python scripts/adaptive/task16.py release \
  --fraction 05 --policy entropy --seed 135 --cycle 1
# Once that first cycle succeeds, either add further chains using --submit-* filters,
# or submit every remaining chain, all held:
# python scripts/adaptive/task16.py submit --walltime 00:20:00
# Release all remaining submitted jobs explicitly once ready:
# python scripts/adaptive/task16.py release --all
python scripts/adaptive/task16.py status
python scripts/adaptive/task16.py collect --destination /u/dkourtesis/task16_results_final.zip
```

Note: there is no change to Task 4/10% runs. The 5%/20% comparison includes different numbers of warm-start training updates, so report results vs cumulative acquired objects and training cost.

## Task 17 — random unstratified L0 (3 new L0 bootstraps + 18 cycles)

**Preconditions:** Already prepared, tested and preflighted, but no bootstrap submission at last update. Bootstrap jobs must finish before pilot or acquisition-training jobs. New L0 intentionally has no class quotas.

```bash
python scripts/adaptive/task17.py verify
python scripts/adaptive/task17.py status
# Three HELD bootstrap jobs; user requested 20 min maximum:
python scripts/adaptive/task17.py submit-bootstrap --walltime 00:20:00
# Optional one-seed validation first:
python scripts/adaptive/task17.py release-bootstrap --seed 135
# After inspecting seed-135 bootstrap, release the other two:
python scripts/adaptive/task17.py release-bootstrap
```

Inspect actual class counts and bootstrap completion first. Then the acquisition-only pilot:

```bash
T17="$P/astrominn_task17_random_l0_v1"
sbatch --parsable --time=00:20:00 --job-name=T17_pilot \
  --output="$T17/logs/%x_%j.out" --error="$T17/logs/%x_%j.err" \
  "$T17/source/scripts/adaptive/task17_worker.sbatch" "$T17" pilot 135
python scripts/adaptive/task17.py pilot-status
```

After successful pilot and all needed bootstraps:

```bash
python scripts/adaptive/task17.py submit-cycles --walltime 00:20:00
# 18 HELD jobs in 6 independent three-cycle chains
python scripts/adaptive/task17.py release-cycles --policy entropy --seed 135 --cycle 1
# Verify checkpoint, validation metrics and run state; then:
python scripts/adaptive/task17.py release-cycles --all
python scripts/adaptive/task17.py status
python scripts/adaptive/task17.py collect --destination /u/dkourtesis/task17_results_final.zip
```

## Operational warnings

- **`submit` is not `release`** in these suites. Creating held jobs does not start work. Do not use `--all` until ready.
- Never remove `afterok` dependencies **within** a policy/seed trajectory; later cycles need preceding checkpoints.
- The 20-minute cap is a scheduling choice, **not a measured safe upper bound**. Task 11 had a successful 31-minute outlier, and Tasks 12/14–17 have different inference and training demands. Monitor `TIMEOUT`, `FAILED`, and slow I/O, especially new L0 bootstrap fits. Resubmission after a failure must account for later `afterok` dependencies and ledger state.
- Slurm accounting: `sacct -u "$USER" --starttime 2026-10-28 -X --format=JobID,JobName%34,State,Elapsed,Timelimit,ExitCode`.
- If a job or pilot already exists, **inspect before submitting another**. Some experiment managers reject duplicate submission ledgers.
- For transfers of completed archives, run `scp` **from Windows PowerShell**. Example:

```powershell
scp "dkourtesis@dt-login.delta.ncsa.illinois.edu:/u/dkourtesis/task17_results_final.zip" "$HOME\Downloads\"
```
