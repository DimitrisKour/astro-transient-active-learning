# Task 9A / 9B — Isolated ensemble and SWA stabilization pilots

## Scientific scope

Six ORIGINAL Task4 first-rejected acquisition cycles, three seeds × two policies, from the validated Task8B first-rejection plan. Five class order: `AGN,TDE,SN I,SN II,CV`. Each case reuses exactly the original cumulative `L_t`. Neither pilot acquires new objects, changes Task4 or Task8B, selects checkpoints using test data, or alters the 8C closed loops.

### Task 9A — Fixed validation probability ensembles (CPU only)

Uses Task8B saved **validation** object-level logits (`validation_predictions.npz`), converts to probabilities by softmax, validates checkpoint hashes/training-manifest checksums/class order/IDs/targets and averages probabilities for fixed seed lists:

- `E2`: retries 1001, 1002
- `E4`: retries 1001–1004
- `I_E2`: incumbent + retries 1001, 1002
- `I_E4`: incumbent + retries 1001–1004
- `E5`: retries 1001–1005, Task8B reference
- `I_E5`: incumbent + retries 1001–1005, Task8B reference

Reference comparisons: Task4 incumbent; original rejected Task4 candidate; the first retry (among 1001..1005) passing the original 1 percentage-point macro-F1 gate when one exists. Importantly, no validation-best subset or model is chosen for the ensemble. Metrics: macro-F1, balanced accuracy, accuracy, NLL, Brier, 15-bin ECE, object-level flips, per-class recall/precision/F1/count, full confusion counts, paired class-stratified descriptive bootstrap intervals for macro-F1 and BA. CSV reports generated on HDD. **Zero new GPU training.** Compare ensemble gain to deployment cost of K inference passes and K checkpoints (not actually benchmarked here).

**CPU queue (one job):**

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
conda activate al-pipeline
python -m pytest tests/jobs/test_task9a_ensemble.py tests/jobs/test_task9b_swa.py -q
python scripts/adaptive/task9a_ensemble.py verify
mkdir -p /work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task9a_fixed_ensembles_v1/logs
sbatch --parsable \
  --output=/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task9a_fixed_ensembles_v1/logs/%j.out \
  --error=/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task9a_fixed_ensembles_v1/logs/%j.err \
  scripts/adaptive/task9a_ensemble_worker.sbatch "$PWD"
```

Check `sacct -j <jobid>`, then `.../analysis/task9a_fixed_ensembles_v1/summary.json`. No need to submit an `sbatch` if preferred: `python scripts/adaptive/task9a_ensemble.py run` is CPU-only, but using a CPU job protects login resources.

### Task 9B — Ten-epoch SWA continuation (GPU)

For each same original rejected Task4 checkpoint, initialize the same ViT tower and frozen experts. Preserve the original training manifest, augmentations, `per_sample` training weighting, `.85` training-manifest class-weight exponent, AdamW betas/epsilon/weight decay and gradient clipping. Start a **fresh AdamW state** (not an optimizer resume). Continue **exactly 10 epochs** with fixed learning rate **2e-5** (no scheduler, no validation-based stopping). Average the model's ViT weights/buffers after epochs **6,7,8,9,10** with `torch.optim.swa_utils.AveragedModel`. ViT BatchNorm is explicitly prohibited until a BN refresh procedure is implemented; inspect any failure rather than using incorrect BN buffers. Save two single-tower checkpoints: `final_vit_tower.pth` and `swa_vit_tower.pth`. Evaluate **validation only** for both (same fixed split and class order); report SWA versus continuation final model, original rejected model and original incumbent. Count and report class losses and gains separately. Paired bootstrap remains descriptive.

The final nonaveraged continuation is the essential control: its comparison to SWA isolates the contribution of averaging within the same 10-epoch continuation. This pilot does **not** test running SWA over the entire original Task4 training trajectory and cannot rule that approach out.

**Freeze and verify (no jobs):**

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
conda activate al-pipeline
python scripts/adaptive/task9b_swa.py plan
python scripts/adaptive/task9b_swa.py prepare
python scripts/adaptive/task9b_swa.py verify
```

Preflight freezes the exact Task8B `source/src` scientific code and adds only the new SWA module/worker. The plan records input SHA-256s and rejects mutation, duplicate output, or wrong case. All work is placed under `/work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task9b_swa_continuations_v1`.

**Queue two GPU pilots (one per policy):** Cases use sorted keys, so 0 = balanced-exploration seed135, 3 = entropy seed135.

```bash
python scripts/adaptive/task9b_swa.py submit --indices 0,3 --walltime 00:20:00
python scripts/adaptive/task9b_swa.py status
```

Inspect both jobs via `sacct` and the separate logs. This is ~2 GPU jobs, not an array requiring other cases. If both finish, queue the **remaining four** without modifying the code or policy:

```bash
python scripts/adaptive/task9b_swa.py submit --indices 1,2,4,5 --walltime 00:20:00
python scripts/adaptive/task9b_swa.py status
python scripts/adaptive/task9b_swa_report.py
```

The training worker activates `al-pipeline`, preloads Conda `libstdc++.so.6`, and pins the frozen source with `PYTHONPATH`. Each epoch saves an atomic restart state including AdamW, RNG states, accumulated SWA and loader RNG. A Slurm failure requires explicit investigation and **manual, intentional** resubmission; duplicate submit markers prevent unintentional second runs. If truly recovering a previously failed case with a valid saved restart state, submit the exact frozen worker directly with its existing index instead of deleting checkpoints. Finished jobs do not restart training.

Report output: `.../analysis/task9b_swa_continuations_v1/report/`.

## Compute / interpretation

Task9A: one CPU job (about 2 CPUs / 12GB / up to 20 min) with **no training**. Task9B: six short GPU jobs (A40; per-job 20-min safety limit) and two validation inference runs per job. An approximate training-only budget is **0.3–0.8 GPU-hours** across six; walltime, inference and queue overhead are extra and measured after running. If training + evaluation exceeds 20 min, the saved epoch states allow resumption; do not silently increase walltime, restart or alter the policy without recording the change.

Neither method currently merits closed-loop promotion on a single small favorable validation delta. Examine minority and majority class correct-object counts, not only macro-F1. TDE has small validation support. Bootstrap intervals on reused validation observations do not have independent confirmation status.

## Suggested subsequent 20-seed experiment (not included in this package)

If ensemble/SWA performance remains ambiguous, select two **pre-declared** difficult Task4 cases (ideally one per acquisition policy) and train 20 fixed seeds each from exactly the same original incumbent and `L_t`. That's **40 full fits**, approximately 2.3–2.7 training GPU-hours at observed ~3–4 minutes/fit, plus evaluation overhead. Analyze the **full** distributions, success probabilities, recall trade-offs and the first-pass probability as retry budget rises from 1 to 20. Do not cherry-pick the best 20-seed model as the scientific result; it is validation-selected. This would be a different study of diminishing returns and retraining variability.
