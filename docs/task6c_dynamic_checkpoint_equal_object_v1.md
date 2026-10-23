# Task 6C — Equal-object weighting with selectable checkpoint strategy

**Choice is made at queue time, never within a trajectory.** There are two prepared, separately frozen suites:

| CLI choice | Reference | Reused seed-specific L₀ | Checkpoint selection during cycles |
| --- | --- | --- | --- |
| `loss` | Task 4 `astrominn_closed_loop_warm_v1` | Original Experiment B L₀ | Weighted validation loss-best |
| `macro_f1` | Task 6B `astrominn_closed_loop_warm_f1best_085_v1` | Task 6B L₀ (F1-best) | Object-level macro-F1-best |

Both retain class-weight exponent **0.85**, frozen pretrained expert towers, full ViT fine-tuning, warm-start per cycle, loss-driven LR scheduler and loss-driven early stopping. The *sole new intervention* is `new_object_weighting="equal_object"` during cycles 1–9. Historical L₀ rows retain weight one. Each newly acquired object's total contribution is set to the historical mean number of alert samples per object; its individual alerts share that weight. This is **not** equal weighting of every historical object and must be described precisely in reports.

**No new L₀ fits are required.** The selected reference L₀ is identical at cycle zero, so each chosen mode requires **54 new A40 GPU jobs**: three seeds × two policies × nine cycles. The unused mode remains unqueued. The seed-specific bootstrap is validated against its *original per-sample* training policy; only the subsequent training policy uses `equal_object`, preventing the Task 6A bootstrap-validation failure. Each job is capped at **20 minutes** by default (adjustable to 15 after a pilot). Dependencies are independent for each of six seed/policy chains.

## Installation and preparation (safe before 6B completes)

PowerShell:

```powershell
scp "$HOME\Downloads\task6c_equal_object_dynamic_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

Delta (don't source the strict common shell in your interactive session):

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/task6c_equal_object_dynamic_v1.zip -d .
python -m pytest tests/jobs/test_task6c_v1.py -q
python scripts/adaptive/task6c.py prepare --selection loss
python scripts/adaptive/task6c.py prepare --selection macro_f1
python scripts/adaptive/task6c.py verify --selection loss
python scripts/adaptive/task6c.py verify --selection macro_f1
```

`prepare` freezes each mode's scientific runner and creates the replay split **without training or requiring Task 6B to be finished**. `verify` checks source/registry/split and reports L₀ readiness. An unfinished Task 6B should produce `bootstrap ready=False` for the macro-F1 choice, not an error.

## After reviewing Task 6B

Choose **one** checkpoint strategy. Dry-run is harmless:

```bash
python scripts/adaptive/task6c.py plan --selection loss
python scripts/adaptive/task6c.py plan --selection macro_f1
python scripts/adaptive/task6c.py preflight --selection CHOSEN_MODE
python scripts/adaptive/task6c.py submit --selection CHOSEN_MODE --walltime 00:20:00
```

The `preflight` reads and validates the three existing L₀ checkpoints using the frozen runner and confirms no training will be started on the login node. Only then, and only after confirming no prior submission for the chosen mode:

```bash
python scripts/adaptive/task6c.py submit --selection CHOSEN_MODE --walltime 00:20:00 --submit
```

It submits **54 held jobs**, installs one `afterok` dependency per consecutive cycle of the **same seed/policy**, and releases them in reverse order. It blocks all other re-submission attempts if a submission marker or trajectory state exists. The six first-cycle jobs are independent and reuse the appropriate matching L₀ checkpoint.

For progress: `squeue -u "$USER" -o '%.18i %.35j %.2t %.10M %R'`.

## Reporting

```bash
python scripts/adaptive/task6c_report.py --selection CHOSEN_MODE
```

Report output is under `closed_loop/<suite>/report/` with `summary.json`, `paired_validation.csv`, `paired_per_class.csv`, and `run_status.csv`, compared exclusively against the matched checkpoint-selection reference. Test is report-only, not used to choose future protocols.

## Limits and safety

- The two modes have **different suite IDs**, frozen snapshots, and result directories. Never change a mode after submission or copy L₀ checkpoints between modes.
- Calibration and test data are not training inputs; all acquired labels use the historical simulator, with the fixed U₀ candidate pool and holdout validation.
- This study tests *equal weighting for newly acquired objects*, while preserving the historical sample-level loss. It is not a full-object-weighted historical retrain.
- The checkpoint selection for macro-F1 comes from Task 6B; if its L₀ checkpoint is missing/incompatible, submission fails closed.
- Do not edit the snapshot or hand-retry all 54 jobs after an interruption. Inspect logs and failed Slurm chains first.
