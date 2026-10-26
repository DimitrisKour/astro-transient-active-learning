# Task 8B — First-rejection seed recovery, v1

**Purpose.** Test whether changing only training randomness recovers an AstroMiNN ViT checkpoint at each of the six Task 4 first rejections. Task 8A identifies the rejection cases under *exploratory* `macro_tolerance_1pp`; this is a provisional screening rule, not a scientifically established threshold or a deployed controller.

## Fixed experiment

- **Reference:** Task 4 `astrominn_closed_loop_warm_v1`, original 0.85 exponent, per-sample training objective, full ViT tower warm-start, weighted validation-loss-best checkpoint and frozen pretrained expert towers.
- **Six cases**: entropy seeds 135/246/357 at cycles 4/4/5; entropy_balanced_explore seeds 135/246/357 at cycles 4/3/6. The previous accepted cycle is `t-1` for each case.
- **Five retries per case**: training seeds `1001,1002,1003,1004,1005`. Both `TrainingRequest.seed` and DataLoader `loader_seed` change, to isolate *training stochasticity* as a package. All other Task 4 training configuration fields are copied from the exact source candidate's training summary. This is not compute-matched against the historical prior cycle or an isolated augmentation-only experiment.
- **Source L_t**: historical cumulative manifest already generated for the rejected cycle; no object, label or sample is added or removed. Its SHA256 must match Task4's original `training_summary.json`.
- **Starting checkpoint**: exactly the existing selected Task4 checkpoint from cycle `t-1` (not rejected checkpoint), verified against its saved validation evaluation and candidate's original training ancestry.
- **Evaluation**: validation split only; never test-split evaluation. Original rejected candidate and incumbent are read from their saved validation NPZ. Existing Task4 runs are read-only.
- **No acquisition, no new oracle labels, no modification of Task4 or Task8A.** Each retry writes into a unique HDD directory.
- **Ensembles**: simple mean of five retry probabilities and separately mean of incumbent-plus-five. No new training. We do **not** use a 'best-of-five' ensemble or pretend validation tuning is an independent test.

## Install (Delta)

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/task8b_first_rejection_recovery_v1.zip -d .
conda activate al-pipeline
python -m pytest tests/core/test_seed_recovery.py tests/astronomy/test_task8b_cases.py tests/jobs/test_task8b.py -q
```

### Before GPU submission (CPU/login, read-only)

```bash
python scripts/adaptive/task8b.py plan
python scripts/adaptive/task8b.py prepare
python scripts/adaptive/task8b.py verify
```

`prepare` copies **Task4's already frozen `src/`** and adds only the new Task8B modules and scripts under the isolated Task8B HDD job directory; writes a frozen `registry.toml`, complete source hash manifest and `plan.json`. It validates the selected six cases against the existing merged 8A `first_rejections.csv`, its hash, all 6 saved cumulative manifests, checkpoint/evaluation SHA256s, Task4 trainer policies, and the Task4 frozen source snapshot. Prepare and verify **submit no Slurm jobs and perform no GPU operations**. Re-running prepare verifies equality; source changes require a distinct suite.

### Submit (explicit user action only)

```bash
python scripts/adaptive/task8b.py submit --walltime 00:20:00 --max-concurrent 3
```

This submits a single SLURM GPU array (`0-29%3`); each array element is one independent training + validation-only evaluation. It uses the `gpuA40x4` partition and `bcrv-delta-gpu` account, at most 3 concurrent A40 GPU jobs. **The 20-minute walltime is a limit, not an expected runtime.** A GPU fit and its validation inference must both finish; if timeouts occur, do not blindly resubmit partial directories. Discuss safe recovery first. To reduce scheduling contention, use `--max-concurrent 1` or `2`; increase walltime only if actually necessary. SLURM logs: `$JOB_DIR/logs/T8B_%A_%a.{out,err}` where `$JOB_DIR=/work/hdd/bcrv/dkourtesis/applecider_adaptive/recovery/astrominn_task8b_first_rejection_recovery_v1`.

### Status and reporting

```bash
python scripts/adaptive/task8b.py status
python scripts/adaptive/task8b_report.py
```

The report requires all 30 completed fits unless `--allow-partial` is set. Produces `report/summary.json`, `report/case_summary.csv`, `report/model_comparisons.csv`, `report/per_class_comparisons.csv`, `report/prediction_flips.csv` (only if changed objects). Status verifies checkpoints; report additionally verifies provenance, predictions and class order. Run `task8b_report.py --allow-partial` for an initial inspection before all jobs finish, noting incomplete ensembles. 

For transfer, package just the report (not checkpoint blobs):

```bash
D=/work/hdd/bcrv/dkourtesis/applecider_adaptive/recovery/astrominn_task8b_first_rejection_recovery_v1
tar -czf /work/hdd/bcrv/dkourtesis/applecider_adaptive/task8b_review_v1.tar.gz -C "$D" report
```

From PowerShell:

```powershell
scp "dkourtesis@dt-login.delta.ncsa.illinois.edu:/work/hdd/bcrv/dkourtesis/applecider_adaptive/task8b_review_v1.tar.gz" "$HOME\Downloads\"
```

## Interpretation, safeguards and limits

We report a **distribution** across the five trials and class recall/f1 changes, and both original-rejected and incumbent baselines. Success against the provisional 1 pp macro rule is not equivalent to better science: inspect TDE and SN II and balanced accuracy as well. All training selects epochs using **the same historical validation set**, and candidate seed selection would reuse it again; best-of-five results suffer selection optimism. Bootstrap intervals are descriptive and do not correct this. All six cases are along the saved original Task4 trajectory up to first rejection; after that point no later Task4 checkpoint is interpreted as a hypothetical acceptance-controlled continuation. Task8C will need a new closed-loop counterfactual.

Check source provenance and role of L0 carefully before changing anything. Do not remove source runs, training manifests, or the frozen Task4 source snapshot during this experiment.
