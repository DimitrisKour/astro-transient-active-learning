# Task 6A v2 — complete 0.5-weighted warm-start closed loop

## Why v2

The preceding prepared suite, `astrominn_closed_loop_warm_weight05_v1`, was designed to **reuse** the original 0.85-exponent L0. This is a **scientifically different experiment**: freshly train one 0.5-exponent L0 for each of seeds 135, 246, 357, then use that L0 both for cycle-1 inference and as the cycle-1 warm-start checkpoint. The 0.5 weighting continues through cycles 1–9, without checkpoint selection or object-weighting changes.

**Do not run `repair6a_bootstrap_policy.py`, `task6a.py submit`, or `task6a_submit_chains.py` for this experiment.** The old frozen source and results stay untouched. This package uses a **new suite ID** `astrominn_closed_loop_warm_weight05_newl0_v1` and scripts named `task6a_l0_*`.

## Scientific contract

- Historical split seed: 135, 10% L0 and 4,253 U0 candidates, corresponding to the original Experiment B split.
- Seeds: 135, 246, 357 (one independent L0 fit per seed, three in total).
- Policies: **entropy** and **entropy_balanced_explore** only. Entropy diversity is excluded from this pilot.
- L0 trained from a fresh randomly initialized ViT with **class-weight exponent 0.5**; expert towers pretrained and frozen, as in the original Experiment B. All cycle models use **0.5**.
- AL cycles 1–9 acquire 473 per cycle except 469 in cycle 9, as in Experiment B.
- Warm-start full ViT fine-tuning (previous cycle loss-best checkpoint for cycle t>1); per-sample cross-entropy, class weights from each training manifest, best weighted validation-loss checkpoint, 50 max epochs, patience 10, batch size 512, learning rate 1e-4. No added checkpoint-acceptance logic.
- Both policies within the same seed share the **same new L0** checkpoint, and their subsequent acquisitions and trained checkpoints are independent.
- Baseline comparator: completed Task 4 warm-start with exponent 0.85 from L0 through cycle 9. This is an **end-to-end** comparison: the learned L0 and future acquired object sets both change; it does not isolate the effect of cycle weighting alone.

## 57-job dependency graph

There are 3 GPU bootstrap jobs and 6 independent policy trajectories of 9 GPU jobs each: **3 + 3×2×9 = 57**. Each job requests one NVIDIA A40 GPU, 8 CPUs, 32 GB memory; default walltime is 20 minutes (configurable separately for L0 and cycles, maximum under 25 minutes).

```text
seed135 L0(.5) ┬─ entropy c1 → c2 → ... → c9
              └─ balanced-explore c1 → c2 → ... → c9
seed246 L0(.5) ┬─ entropy c1 → c2 → ... → c9
              └─ balanced-explore c1 → c2 → ... → c9
seed357 L0(.5) ┬─ entropy c1 → c2 → ... → c9
              └─ balanced-explore c1 → c2 → ... → c9
```

Every `afterok` points to exactly **one parent job**: c1 to the correct seed L0, or cN to the immediately preceding cycle of the **same seed and policy**. There is no global cycle barrier; at most six GPU jobs may be running simultaneously if Delta allows it (no global three-GPU cap is asserted).

## Installation, prepare, dry run

From Windows PowerShell, after downloading this ZIP:

```powershell
scp "$HOME\Downloads\task6a_l0_weight05_v2.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

On Delta, run **without sourcing the common shell script directly** (it enables `set -euo pipefail` and can terminate an interactive session on error):

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/task6a_l0_weight05_v2.zip -d .

python -m pytest tests/jobs/test_task6a_l0_v2.py -q
python scripts/research/historical_replay/task6a/task6a_l0.py prepare
python scripts/research/historical_replay/task6a/task6a_l0.py verify
python scripts/research/historical_replay/task6a/task6a_l0.py plan --through 9 --walltime 00:20:00 --l0-walltime 00:20:00
```

`prepare` freezes its **own source**, materializes the immutable candidate split with no training, verifies matching original split content, and verifies each seed's no-run preflight. `plan` is read-only. In an environment where import of `sqlite3` fails with `CXXABI_1.3.15`, run the commands via the existing Conda `al-pipeline` environment with its proper C++ library configuration, or a subprocess of `task6a_l0_activate`, not by sourcing that function into the interactive session. The worker itself performs the full activation and library configuration.

## Submit only after prepare and plan both succeed

```bash
python scripts/research/historical_replay/task6a/task6a_l0.py submit \
  --through 9 \
  --walltime 00:20:00 \
  --l0-walltime 00:20:00 \
  --submit
```

The command submits all **57 held jobs**, installs each precise dependency, persists the scheduler IDs under `closed_loop/astrominn_closed_loop_warm_weight05_newl0_v1/submitted.json`, then releases held jobs in reverse order. No previously prepared Task 6A/Task 4 directories are modified. If a submission action fails midway, an `scancel` is attempted for newly queued jobs, the attempt is logged, and manual inspection is required. If a bootstrap fails, both policy chains for its seed remain blocked by `afterok` and do not run against the wrong L0.

Check progress:

```bash
squeue -u "$USER" -o '%.18i %.38j %.2t %.10M %.10l %R'
J=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_weight05_newl0_v1
ls -ltr "$J/logs" | tail
```

Read-only results reporting after runs finish:

```bash
python scripts/research/historical_replay/task6a/task6a_l0_report.py
```

The reporter verifies the newly trained per-seed L0 exponent, warm-start checkpoint ancestry, cycle objectives, distinct acquisitions, and paired validation metrics against Task 4, including per-class precision, recall, F1. The test split must remain report-only when selecting which objective to retain.

## Explicit restrictions

- No direct reruns of the older `.85` L0 checkpoint in this v2 suite.
- No `entropy_diversity` policy.
- Do not manually edit `source_snapshot.json` or alter frozen sources.
- If a job hits the time limit, do not blindly resubmit all 57 jobs. Inspect SLURM failures and incomplete run folders first; the scheduler uses `afterok` to prevent downstream runs from starting.
- This is a **new-L0 + new-cycle objective** comparison, unlike the previous Task6A proposal which shared the original L0.
