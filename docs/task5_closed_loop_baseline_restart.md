# Task 5: AstroMiNN ViT baseline-restart closed loop

**Purpose:** A controlled paired comparison with Task 4 (warm-start). Each acquisition cycle scores the remaining unlabeled pool using its *own* prior cycle's validation-loss-best ViT, but each training fit starts independently from the **same original, seed-specific L0 checkpoint**. This is neither fixed-trajectory replay nor random initialization. Frozen expert towers, training, loss, class weighting, data split, policies, and SLURM schedule are inherited unchanged from Task 4 v2.

Task 4 and the original Experiment B remain immutable. Task 5 writes to the independent suite `astrominn_closed_loop_baseline_restart_v1` using the original read-only L0 artifacts from `astrominn_historical_replay_v1`. Nine acquisition rounds (473 objects each, then 469) across four policies (`random`, `entropy`, `entropy_diversity`, `entropy_balanced_explore`) and three seeds (135, 246, 357) total **108 one-fit jobs**, just as in Task 4.

## Install on Delta, with CPU-only tests first

In **Windows PowerShell** (outside SSH):

```powershell
scp "$HOME\Downloads\task5_baseline_restart_overlay_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

In the **Delta terminal** (after `source ~/al_delta_setup.sh` if needed):

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
# Important: Task 4 v2 must exist in this checkout.
unzip -o /projects/bcrv/dkourtesis/task5_baseline_restart_overlay_v1.zip -d .
python scripts/research/historical_replay/install_task5_restart.py
python -m pytest tests/jobs/test_task5_restart.py -q
# Optional: python -m pytest tests/jobs/test_task4_package.py -q
bash scripts/research/historical_replay/task5/task5_submit.sh --prepare-only
```

`scripts/research/historical_replay/install_task5_restart.py` makes the dedicated `task5_*` scripts by copying the existing tested Task 4 v2 launch infrastructure with guarded transformations. It will **refuse** to overwrite differing Task 5 files. The derived runner is `scripts/run_astrominn_closed_loop_restart.py`; the existing Experiment B and Task 4 runners are untouched. `--prepare-only` freezes the exact source and registry into `/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_baseline_restart_v1` and audits the historical split and baseline checkpoint availability for all seeds. No GPU jobs are submitted.

Only after every test and preflight succeeds:

```bash
bash scripts/research/historical_replay/task5/task5_submit.sh --submit --cycles 9
squeue -u "$USER"
cat /projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_baseline_restart_v1/submitted.tsv
```

Task 5 uses Task 4's A40 single-GPU, dependency-chained, held-then-reversed-release submission logic. Cycle-to-cycle dependencies are **per seed and policy**; the 12 chains can overlap. 15-minute allocations for cycles 1-2, 20-minute for cycles 3-9. If any job exceeds its limit or fails, the dependent chain remains blocked. **Do not blindly submit again** after `submitted.tsv` exists; inspect the state and SLURM logs.

## Read-only reporting and scientific audits

```bash
python scripts/research/historical_replay/task5/task5_report.py
```

The audit checks that for every completed cycle `c`:

- `inference_checkpoint(c) == L0` for `c=1` and Task 5's own `(c-1)` model for `c>1`;
- `training_initial_checkpoint(c) == original seed-specific L0` **for every cycle**, including `c>1`;
- the actual trainer's `training_summary.json` agrees with that fixed L0 checkpoint;
- learned checkpoint paths, unique selections, and per-cycle acquisition counts match the expected schedule.

Validation and test metrics are exported without smoothing; test is **report-only** and cannot influence acquisition, training or checkpoint selection. Comparisons with Task 4 and Experiment B should be made at equal labeled-object budgets. The restart condition is *not* compute-matched against warm-start; it isolates the effect of repeatedly resetting learned weights to L0 under the same per-fit budgets.
