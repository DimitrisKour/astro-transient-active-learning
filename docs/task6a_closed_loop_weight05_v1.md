# Task 6A — warm-start closed-loop with gentler class weights

## Exact scientific contract

**Question:** With the best-supported warm-start architecture retained, does replacing the current 0.85 class-weight exponent by 0.5 improve live acquisition–training performance?

- **Reference:** completed Task 4, `astrominn_closed_loop_warm_v1`, with class exponent 0.85.
- **New suite:** `astrominn_closed_loop_warm_weight05_v1`, independently acquired U at every cycle with exponent 0.5.
- Two acquisition policies: `entropy` and `entropy_balanced_explore`.
- Three paired seeds: `135 246 357`; nine cycles with acquisition 473 objects per cycle (469 final) and full 4,253-object pool exhaustion.
- **54 new fits** (2 policies × 3 seeds × 9 cycles), one A40 GPU per fit, at most **three simultaneously**; per-job walltime default `00:24:00`.
- Reuse the original *same* seed-specific L0 checkpoints from `astrominn_historical_replay_v1`; these were trained at exponent 0.85. The change applies to cycles 1–9 only, not L0.
- Own in-loop model: cycle `c` acquires with the cycle `c-1` model's logits; trains on its OWN cumulative manifest using previous cycle's loss-best checkpoint as warm-start; acquisition trajectories may diverge from Task 4 after cycle 1.
- **All else remains as Task 4:** trainable ViT, frozen three experts, per-sample cross-entropy, class weights from cumulative training manifest, loss-best validation checkpoint selection, 50 max epochs, patience 10, learning rate 1e-4, batch size 512, no validation augmentation, same profiles and original data split, fixed test/validation, same batching, no F1 checkpoint selection, no equal-object weighting.

This isolates a single intervention. Do not modify Task 4, Task 4b, Task 5B, their checkpoints, or the original experiment runner. The generated runner refuses any other suite ID, bootstrap source, policy, or initialization. Source files, registry, and a scientifically meaningful JSON contract are **frozen before GPU scheduling**. A changed source cannot be silently resumed under the same suite name.

## Install — PowerShell

Download the ZIP from ChatGPT to Windows Downloads, then:

```powershell
scp "$HOME\Downloads\task6a_weight05_overlay_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

Delta:

```bash
source ~/al_delta_setup.sh
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/task6a_weight05_overlay_v1.zip -d .
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/jobs/test_task6a_weight05.py -q
python -m py_compile scripts/research/historical_replay/task6a/task6a_*.py
```

Short unit tests only; no GPU or Slurm submission.

## Prepare / preflight — no GPUs and no submission

```bash
python scripts/research/historical_replay/task6a/task6a.py prepare
python scripts/research/historical_replay/task6a/task6a.py verify
python scripts/research/historical_replay/task6a/task6a.py submit --walltime 00:24:00
```

Preparation creates a frozen source under:

`/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_weight05_v1/source`

It materializes a distinct, deterministic L0/U0 replay split *without model inference or training*, checks that it matches the original split and U0 labels, verifies original L0 checkpoints for all seeds, and saves preflight plans. It should report **54 fits planned; no jobs queued**. The last `submit` call without `--submit` is a dry run.

## First real-GPU integration probe

One **cycle 1, seed 135, entropy** fit is enough to check the patched runner, snapshot, data, GPU environment, and duration:

```bash
python scripts/research/historical_replay/task6a/task6a.py probe --walltime 00:24:00
# DRY RUN only. When ready:
python scripts/research/historical_replay/task6a/task6a.py probe --walltime 00:24:00 --submit
```

This queues **one single-A40 job**, not the whole suite. Inspect it:

```bash
squeue -u "$USER"
J=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_weight05_v1
cat "$J/probe_submission.json"
tail -n 60 "$J"/logs/probe_*.out
# use the actual ID printed above
sacct -j JOBID --format=JobID,State,Elapsed,Timelimit,ExitCode
```

Confirm `COMPLETED`/`0:0`; the log should show Task 6A exponent 0.5 and the expected L0 checkpoint. The full submission requires this probe to have completed if it was queued.

## Full 54-fit submission (explicit action only)

```bash
python scripts/research/historical_replay/task6a/task6a.py submit --walltime 00:24:00
# Inspect the dry-run plan first. When ready:
python scripts/research/historical_replay/task6a/task6a.py submit --walltime 00:24:00 --submit
```

This queues nine **SLURM arrays** with six GPU tasks per array; `%3` limits concurrent tasks to three, and cycle `c+1` has an `afterok` dependency on the entire cycle `c` array. This is intentionally conservative: a failure prevents future cycles from training on incomplete histories. The successful probe is reused by the framework, not retrained.

Expected *maximum requested* GPU time: `54 × 24 min = 21.6 GPU-hours` (actual should be lower). At most three GPUs are used simultaneously, and no CPU node allocation is needed. Walltime is per task; the pipeline is **not mid-epoch resumable**. If a fit times out, inspect before retrying. The scheduler saves job IDs in `submission.json`. Never delete the submission record simply to requeue.

## Status and scientific reports

```bash
squeue -u "$USER"
J=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_weight05_v1
cat "$J/submission.json"
python scripts/research/historical_replay/task6a/task6a_report.py
```

The reporter audits all available new trajectory checkpoints and exports matched **validation** metrics versus Task 4 into `$J/report/paired_validation.csv`, per-class metrics into `paired_per_class.csv`, statuses into `run_status.csv`, and summary stats into `summary.json`. It is safe on partial runs; only completed cycles appear.

For sharing with ChatGPT after runs:

```bash
P=/projects/bcrv/dkourtesis/applecider_adaptive
J="$P/closed_loop/astrominn_closed_loop_warm_weight05_v1"
cd "$P"
tar -czf "$P/task6a_review_bundle.tar.gz" \
  "closed_loop/astrominn_closed_loop_warm_weight05_v1/report" \
  "closed_loop/astrominn_closed_loop_warm_weight05_v1/contract.json" \
  "closed_loop/astrominn_closed_loop_warm_weight05_v1/source_snapshot.json"
```

PowerShell download:

```powershell
scp "dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/applecider_adaptive/task6a_review_bundle.tar.gz" "$HOME\Downloads\"
```

## Interpretation and next task

Primary: mean validation object-level macro-F1, early 1–5 learning curve, stability, and paired seed/policy differences at *equal labeled-object budgets*. Secondary: balanced accuracy, per-class F1/recall (TDE, SN II, CV), Brier/ECE, and GPU time. Test labels are not used for selecting training protocol. Because each trajectory acquires its own objects, this is an **end-to-end comparison**, not a fixed-labeled-set training ablation.

After assessing 0.85 vs 0.5, use the better-justified class exponent for an independent within-cycle **macro-F1-best versus validation-loss-best checkpoint selection** experiment. Keep *cross-cycle acceptance/rollback* separate. Equal-object weighting is another isolated follow-up (with the selected exponent/checkpoint policy fixed). Avoid an unidentifiable full factorial grid unless evidence demands it.
