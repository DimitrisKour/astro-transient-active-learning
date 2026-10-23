# Task 6B — Object-macro-F1 checkpoint selection, end-to-end

## Scientific question

Using the same historical initial pool, class-weight exponent **0.85**,
per-sample optimization, full ViT fine-tuning, and warm-start as completed Task 4,
does choosing the **best validation object-level macro-F1 epoch** improve
closed-loop active learning relative to the original **best weighted validation
loss epoch**?

## Treatment, control, and policy

- **Control**: `astrominn_closed_loop_warm_v1` (Task 4), best validation loss
  selected; original separately trained seed-specific L0 checkpoints.
- **Treatment**: `astrominn_closed_loop_warm_f1best_085_v1`, newly trained
  seed-specific L0 and cycle models, macro-F1-best checkpoint selected.
- Both share the deterministic split and the hidden-label oracle; policies
  `entropy`, `entropy_balanced_explore`, seeds 135/246/357, nine acquisitions.
- Both use class-weight exponent 0.85, full ViT training, weighted cross
  entropy, identical optimizer, epoch cap 50, patience 10, fixed seed 135 split,
  frozen expert towers, and equal weight **per sample**.
- **Only epoch checkpoint selection is changed**. Learning-rate scheduling and
  *early stopping* are **still driven by weighted validation loss**. Thus
  macro-F1 selection is restricted to the epochs that the original loss-based
  stopping rule visits; this is a deliberate control rather than an F1 early
  stopping experiment.
- Object-level macro-F1 is computed on the validation manifest using exactly
  **one latest alert per object**; ties in macro-F1 are resolved by lower
  validation loss, then earlier epoch. All five classes contribute equally.
- The selected checkpoint `best_vit_tower.pth` is used for the next cycle's
  **inference AND warm-start**. The reference
  `best_loss_vit_tower.pth` is saved as a companion, never used to advance
  the Task 6B trajectory.
- Neither suite uses test results to select a checkpoint or experimental
  configuration. The fixed validation set drives epoch selection; its use for
  comparing the two approaches therefore has selection bias, which should be
  discussed when interpreting differences. The report includes no test
  metrics in its primary comparison CSV.

## Job DAG / resources

57 independent one-A40 Slurm jobs (not job arrays):

- 3 seed-specific L0 jobs, with no dependencies.
- 2 policy branches per seed, each with 9 jobs.
- For seed S, both cycle-1 jobs depend only on L0(S).
- For each branch (seed S, policy P), cycle n>1 depends only on (S,P,n−1).
- **No cross-seed, cross-policy, or whole-cycle barriers**.
- Each job requests one A40, 8 CPUs, 32 GB RAM. Default **20 minutes** per
  job, including L0 (override with `--walltime` / `--l0-walltime`).
- In the completed Task 4/6A training summaries, longest recorded *training*
  time was ~305 seconds, excluding pool inference, start-up, and I/O. Hence
  15 minutes has not been established as *completely safe*. Use 20 minutes.
- The script queues held jobs first and releases them in reverse order after
  all 57 have been submitted. Partial submission triggers cancellation and
  a durable `submission_failed.json` marker. It refuses blind resubmission.

## Installation (Delta)

Extract ZIP in the **existing active-learning repository**, after verifying
`conda activate al-pipeline` (or your Delta environment setup script):

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/task6b_f1best_closed_loop_v1.zip -d .
python -m pytest tests/jobs/test_task6b_v1.py -q
```

**Preparation runs no model training**; it only writes a new frozen scientific
snapshot and the split after calling the no-`--run` generated runner:

```bash
python scripts/adaptive/task6b.py prepare
python scripts/adaptive/task6b.py verify
python scripts/adaptive/task6b.py plan --walltime 00:20:00 --l0-walltime 00:20:00
```

Only submit after these commands succeed:

```bash
python scripts/adaptive/task6b.py submit --walltime 00:20:00 --l0-walltime 00:20:00 --submit
```

Observe jobs:

```bash
squeue -u "$USER" -o '%.18i %.32j %.2t %.10M %.10l %R'
```

Submission manifest:

```
/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_f1best_085_v1/submitted.json
```

Report while running, or when completed:

```bash
python scripts/adaptive/task6b_report.py
```

Results:

```
/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_f1best_085_v1/report/
```

CSV files `paired_validation.csv`, `paired_per_class.csv`, `run_status.csv`,
`checkpoint_decisions.csv` (loss-best vs F1-best epoch choices), and `summary.json` compare Task 6B with Task 4 on matched budgets and seeds.

## Critical distinction

**This tests checkpoint selection, not macro-F1-based early stopping.** A later
separate experiment could switch patience/scheduler behavior, but doing so
now would confound the checkpoint-selection comparison.

The active Task 4, Task 4b, Task 6A, and their frozen sources are not modified.
All Task 6B training changes are confined to its own frozen-source snapshot.
