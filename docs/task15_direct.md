# Task 15 — DIRECT (ICML 2025), clean retrospective oracle

Reference: Nuggehalli et al., *Improved Algorithm for Deep Active Learning under Imbalance via Optimal Separation*, ICML 2025, https://proceedings.mlr.press/v267/nuggehalli25a.html . Based on Algorithms 1 and 2. This is an explicit object-level adaptation with integer batch rounding, not a claim of byte-identical reproduction.

## Scientific design

- One policy `direct`, seeds 135/246/357, nine 10%-budget cycles (27 training jobs); L0/U0, AstroMiNN and training hyperparameters matched to Task 4 and Task 11, warm-start, validation-only evaluation.
- For each class, sort both eligible U_t and confirmed L_t representations under the same frozen checkpoint by `p_k - max_j p_j` (including j=k), breaking ties with descending p_k. The confirmed labeled reference is inferred using the same checkpoint. No validation/test reference.
- Phase 1 uses **floor(B/2)** positions divided among five classes in seeded class order. Use *VReduce*: draw parallel batches of at most five unlabeled objects in each class's current threshold interval; only after a subbatch is acquired does the retrospective oracle reveal its true labels; then shrink the interval by minimizing the maximum endpoint empirical threshold loss. Model weights are fixed throughout intra-cycle selection.
- Phase 2 is executed **only after all five classes finish VReduce**; divide remaining budget among classes and annotate the closest objects to each learned best empirical one-versus-rest threshold, without replacement. Training happens once after all requested labels are revealed.
- At the end of each cycle save `direct_oracle_journal.json`, `selection.json`, `revealed_labels.json`, and usual training/evaluation artifacts. The journal is written after each oracle request and replayed deterministically on retries.
- This experiment uses a clean historical oracle; noisy labels are **not** enabled. The current simulated oracle is synchronous. Note that a live asynchronous oracle would need further adaptations.
- Integer budget and tie handling are explicitly specified in `scientific_plan.json` and implementation. Pseudocode has ambiguous boundary orientation for multiclass scores; our empirical threshold loss follows increasing class score: rest class on the left, class k on the right.
- Caveat: DIRECT published results were obtained on different datasets; superiority on AstroMiNN is not assumed.

## Transfer (Windows PowerShell)

Save `task15_direct_overlay_v1.zip` to Downloads:

```powershell
scp "$HOME\Downloads\task15_direct_overlay_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

## Prepare + CPU tests (Delta login terminal)

```bash
cd /projects/bcrv/dkourtesis
mkdir -p task15_direct_overlay_v1
unzip -n task15_direct_overlay_v1.zip -d task15_direct_overlay_v1
cd task15_direct_overlay_v1
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
python scripts/adaptive/task15.py plan
python scripts/adaptive/task15.py prepare
python scripts/adaptive/task15.py verify
TASK15_JOB=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task15_direct_vreduce_v1
PYTHONPATH="$TASK15_JOB/source/src" python -m pytest -q tests/test_task15_direct.py
python scripts/adaptive/task15.py preflight
```

`prepare` verifies the **entire Task11 frozen source manifest** and checkpoints before copying and patching it into a separate experiment directory. It never changes Task11–14 artifacts. If a check fails, stop and inspect, don't overwrite the frozen source.

## Acquisition-only GPU pilot (uses subround oracle labels; does not train)

```bash
TASK15_JOB=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task15_direct_vreduce_v1
sbatch --parsable \
  --time=00:35:00 \
  --job-name=T15_pilot_direct \
  --output="$TASK15_JOB/logs/%x_%j.out" \
  --error="$TASK15_JOB/logs/%x_%j.err" \
  "$TASK15_JOB/source/scripts/adaptive/task15_worker.sbatch" \
  "$TASK15_JOB" pilot direct 135
```

Unlike entropy/BALD acquisition-only pilots, this pilot **does query historical labels** for selected objects during acquisition. It never trains, but selected oracle labels and subround transcript are in its pilot result. This is necessary to test the defining DIRECT algorithm. This is controlled retrospective label revelation, not a leak into the scorer prior to acquisition.

Check:

```bash
python scripts/adaptive/task15.py pilot-status
sacct -u "$USER" --starttime 2026-10-31 -X --format=JobID,JobName%35,State,ExitCode,Elapsed | grep T15_
```

Inspect the `pilot/direct/seed_135/result.json` transcript and selection before submitting any training jobs.

## Hold and optionally release training (after pilot confirmation)

```bash
python scripts/adaptive/task15.py submit --walltime 00:35:00
python scripts/adaptive/task15.py status
```

Submission creates **27 HELD jobs**, three independent seed-specific `afterok` chains, no Task11/12/13/14 dependencies. It does **not** release jobs automatically.

First end-to-end validation:

```bash
python scripts/adaptive/task15.py release --policy direct --seed 135 --cycle 1
```

After inspecting full cycle 1 outputs and `direct_oracle_journal.json`, release all remaining jobs:

```bash
python scripts/adaptive/task15.py release --all
```

Report collection (later):

```bash
python scripts/adaptive/task15.py collect --destination /u/dkourtesis/task15_direct_results.zip
```

Then **Windows PowerShell**:

```powershell
scp "dkourtesis@dt-login.delta.ncsa.illinois.edu:/u/dkourtesis/task15_direct_results.zip" "$HOME\Downloads\"
```

No model checkpoints or raw candidate data are included in the results ZIP.
