# Task 4: closed-loop warm-start historical replay

## Scientific question

Does continuing the latest validation-loss-selected AstroMiNN ViT between
acquisition cycles improve **closed-loop** learning, relative to the original
Experiment B random-restart trajectories? Task 2 was open-loop (fixed batches);
Task 4 **recomputes every non-random acquisition batch** using its own current
model. The experiment is not a re-evaluation of saved Task-2 manifests.

## Fixed conditions

* Five historical classes: AGN, CV, SN I, SN II and TDE.
* Original L0 and U0 generated from the same historical training manifest with
  split seed 135; compare object-level rows to original Experiment B before jobs.
* Seed-specific *original Experiment B* completed L0 checkpoints for model seeds
  135, 246 and 357. The original checkpoints are **read-only**.
* Four policies: random, entropy, entropy_diversity, entropy_balanced_explore;
  original Experiment B acquisition parameters.
* Label acquisition size 473 per cycle, then 469 at c9; 100% labeled at c9.
  All **nine cycles** are submitted together as short dependency-chained jobs.
* Original loss-selected checkpoint, sample-level training and class weights
  from current L_t; max 50 epochs, patience 10, LR 1e-4; frozen expert towers.
  Newly acquired objects are always trained on together with all prior L_t.
* Every fit resets the optimizer and LR scheduler. Weights are carried, optimizer
  moments are not. This is not a compute-matched restart comparison.
* Fixed historical validation/test manifests. Test scores are report-only and
  **never** determine checkpoints, acquisition, stopping, or model acceptance.

## What happens at each cycle

1. For c1, use the original seed-specific L0 checkpoint for inference. For c>1,
   use Task4's own previous loss-best checkpoint (for that seed and strategy).
2. Score the current U_t and acquire 473 new object IDs with the chosen policy.
   The `random` policy ignores model probabilities but uses exactly the same
   pool and seed schedule as the original reference.
3. Reveal only selected historical labels; persist their selection/order and
   cumulative manifest; never look up hidden U_t labels in acquisition rules.
4. Train on **all cumulative L_t**, initializing ViT from L0 at c1, or the latest
   previous-cycle loss-best ViT at c>1. Frozen pretrained expert towers stay fixed.
5. Select best checkpoint using weighted validation loss; evaluate the fixed
   validation/test objects; persist checkpoint, metrics and state before c+1.
6. Every stage is resumable from existing durable manifests, predictions, and
   training summaries. One seed's cycles must run sequentially; independent seeds
   may run concurrently. A failure halts dependent jobs, never advances a cycle.

## Installation and Delta preparation

The deliverable is an overlay ZIP, not a whole repository. From the **repo root**:

```bash
unzip -o /path/to/task4_closed_loop_overlay_v2.zip
# Test only the newly added files first
python -m pytest tests/jobs/test_task4_package.py -q
# Then run full tests if desired
python -m pytest tests/astronomy/ tests/jobs/ -q
bash scripts/research/historical_replay/task4/task4_submit.sh --prepare-only
```

The prepare-only step generates a **separate** `scripts/run_astrominn_closed_loop.py`
from the checked-in Experiment B runner; never modifies the original runner.
It then freezes all `src/al_pipeline/*.py`, the new runner, and the Task4 job
scripts into the persistent suite directory. During GPU execution, all imports
and entry points use the frozen copy, so subsequent edits for Task 5 on the
working repository will not contaminate these queued trajectories.

Expected suite:

```text
/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_v1/
    registry.toml
    source_snapshot.json
    source/
    logs/
    plan_seed_135.txt
    plan_seed_246.txt
    plan_seed_357.txt
```

If no `configs/delta.toml` is available, the launcher uses the original suite's
`cycle_jobs_v1/registry_hdd.toml` registry as source and changes only `[storage]`.
Outputs go to persistent `/projects/...` and heavyweight `/work/hdd/...` storage.

## Submission (only after prepare-only passes)

```bash
bash scripts/research/historical_replay/task4/task4_submit.sh --submit --cycles 9
squeue -u "$USER"
cat /projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_v1/submitted.tsv
```

Submission creates **108 SLURM jobs** (9 cycles x 4 policies x 3 seeds). Each
short GPU job runs exactly one policy for one cycle. The three seeds form
independent serial chains (36 jobs per seed), so at most three GPUs run
at once. Cycles 1-2 request 15 minutes, cycles 3-9 request 20 minutes.
Jobs are submitted held, chained with `afterok` within each seed, then
released in reverse order. Policies maintain independent learning histories.
A successful c9 acquires the last 469 objects, giving 100% label coverage. No Task 1-3
outputs are modified. `--prepare-only` never queues jobs.

If a job times out, the dependent chain stops. Resubmission should be handled
by inspecting the saved state, training artifacts, and the SLURM error first;
**do not delete `submitted.tsv` and blindly rerun `--submit`**.

The old `task4_extend.sh` is disabled: extension is no longer necessary.
The default plan already queues all nine cycles.

TODO (after Task 4 analysis): optional Task 3 weighting/checkpoint variants in
independent, differently named suites—not in this warm-start experiment.

## Read-only run report / integrity check

```bash
python scripts/research/historical_replay/task4/task4_report.py
```

The report writes `report/evaluation_history.csv` (unsmoothed validation and
**report-only** test metrics) and `report/run_status.csv`. It checks that
inference and training consume exactly the prior cycle's loss-best checkpoint,
that acquisitions never repeat an object, and that batch sizes match the plan.
In future analysis, explicitly compare at **equal label budgets** to the
original random-restart Experiment B trajectories.
