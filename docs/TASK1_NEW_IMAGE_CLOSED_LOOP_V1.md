# Task 1 — Full historical closed-loop AL, new BTSbot ConvNeXt + 55-feature metadata

## Read this before running

This is an independent, nine-cycle **historical simulation**, **not** online follow-up on new astronomical objects. Historical training labels exist on disk but are revealed to the training manifest **only after object-level acquisition**. No test information influences the acquisition process, the training budget, the checkpoint, or the model used by a subsequent cycle. The initial 476-object L0 and 4,253-object candidate pool are disjoint subsets of the complete 4,729-object historical TRAIN population. Validation: 2,365 objects. Test: 2,365 objects. Never initialize from any fully trained Task 0 checkpoint.

**No data cache is rebuilt per cycle.** The immutable preprocessed cache on HDD already includes ALL original historical TRAIN and VALIDATION stages. Each GPU cycle loads those arrays sequentially into **one shared host-RAM buffer** for acquisition and cumulative training. The dedicated immutable **TEST** cache is loaded only after the validation-loss-selected checkpoint is saved; it is never available to model selection.

Files added by this overlay:
- `src/al_pipeline/astronomy/fusion_image/task1_acquisition.py` — four policies, using the repository's pre-existing `TopKSelector`, `DiversitySelector`, `ExplorationSelector`, and scoring classes
- `src/al_pipeline/astronomy/fusion_image/task1_training.py` — weighted CE + D4 + warmup/cosine resident-RAM training, two checkpoints, automatic sealed test inference
- `scripts/adaptive/task1_closed_loop.py` — read-only provenance verification, L0 and cycle orchestration, reports
- `scripts/adaptive/task1_schedule.py` — independent SLURM afterok chains with incremental submission registry
- `scripts/adaptive/task1_cycle.sbatch` and `scripts/adaptive/submit_task1_closed_loop.sh`
- `tests/astronomy/fusion_image/test_task1_closed_loop_v1.py`

This overlay does **not** modify any frozen Task 0 source file, train cache, or checkpoint.

## Experiment matrix

Three configurations: `random_warm` (randomized ConvNeXt image weights + warm-start from previous cycle), `random_l0` (randomized ConvNeXt image weights + restart every cycle from the SAME seed's initial L0 checkpoint), `pretrained_warm` (GalaxyZoo/BTSbot-pretrained ConvNeXt image weights + warm-start). The underlying model architecture is identical in all configurations. Seed 135, 246, 357.

Four acquisition policies: `random`, `entropy`, `entropy_diversity` (3x high-entropy shortlist + greedy farthest-first in normalized embeddings), `entropy_balanced_explore` (20% uniform exploration + 80% top entropy, matching the existing AL `ExplorationSelector`; the word **balanced** does NOT mean access to hidden class labels or a class-quota algorithm). In every policy the acquisition unit is one **object** represented by its latest available historical TRAIN alert stage; no TEST or candidate labels are provided to prediction/selection functions.

Nine acquisitions: 473 objects for cycles 1–8, 469 for cycle 9. All curves include shared initial cycle 0. Cumulative `L_t` manifests retain **every original TRAIN row** and its multiplicities in source order for selected object IDs. Other unselected objects remain strictly excluded from the training manifest. Random and entropy are **phase 1**; advanced policies are **phase 2**, and require explicit manual release after ALL phase-1 experiments complete.

Two shared initial L0 runs per seed (randomized and pretrained) = 6 fits; 18 phase-1 trajectories × 9 cycles = 162 additional fits; phase 2 similarly 162. There are 330 fits total if every phase is performed. (These do not include model-test passes, which run inside each fit's SLURM job.)

## Model and training protocol

Use the same frozen Task 0 `cosine_aug` recipe: AdamW learning rate 1e-4, weight decay .01, batch size 64, image D4 rotations/flips on TRAIN only, class weights proportional to (L_t alert-row class count)^(-0.85), 3-epoch linear warmup, 80-epoch cosine schedule, max 80 epochs, early-stop patience 15 on weighted validation loss. No EMA. The checkpoint with minimum weighted validation loss is the PRIMARY checkpoint throughout Task 1; a second best-object-macro-F1 checkpoint is saved for diagnostics. Never choose which checkpoint to use from TEST.

Warm-start loads the preceding cycle's PRIMARY model **weights**; optimizer, scheduler, and early stopping reset for every independent fit. L0-restart loads the immutable initial L0 PRIMARY model weights at every cycle, then trains over the full cumulative L_t manifest with a fresh optimizer/scheduler. Test metrics are never considered for accept/reject, architecture adaptation, or dynamic hyperparameter tuning.

The Task 1 scripts intentionally keep the full training recipe consistent with the Task 0 choice even if a later analysis might favor a different checkpoint selection rule. Every run's plan includes exact source-file SHA256 checksums; modifying these scientific source files after `--prepare` requires a **new versioned experiment root** rather than changing existing trajectories.

## Install

On Windows PowerShell, assuming ZIP is in Downloads:

```powershell
scp "$HOME\Downloads\fusion_task1_closed_loop_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

On Delta:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o /projects/bcrv/dkourtesis/fusion_task1_closed_loop_v1.zip -d .
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/astronomy/fusion_image/test_task1_closed_loop_v1.py -q
bash scripts/adaptive/submit_task1_closed_loop.sh --prepare
bash scripts/adaptive/submit_task1_closed_loop.sh --check
```

`--prepare` freezes identities and creates a **label-free** pool index. It does not require the pending test-cache CPU jobs to have completed. **Job submission DOES require** the finalized test cache to match the frozen train/validation/test manifests and preprocessing.

After cache consolidation and verification:

```bash
bash scripts/adaptive/submit_task0_test_cache.sh --validate
bash scripts/adaptive/submit_task1_closed_loop.sh --submit-l0
bash scripts/adaptive/submit_task1_closed_loop.sh --status
```

**Strongly advised**: Inspect the six L0 `complete.json`, `fit/training_summary.json`, and `sealed_test/evaluation_summary.json` files first. In particular, the macro-F1 metrics and all five class counts should be sensible. Then release only random and entropy:

```bash
bash scripts/adaptive/submit_task1_closed_loop.sh --submit-phase1
```

The script submits each cycle as its **own GPU SLURM job**. Every cycle depends on its trajectory's predecessor using `afterok`. L0 is shared rather than retrained for each policy. The submission registry is append-only: `/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task1_closed_loop_v1/slurm_jobs.tsv`.

Slurm may impose a per-user maximum number of pending jobs. If `sbatch` rejects a job due to queue limits, previously submitted job IDs remain recorded. To submit incrementally instead of all 162 at once, specify e.g.:

```bash
bash scripts/adaptive/submit_task1_closed_loop.sh --submit-phase1 --limit 24
# Repeat the SAME command later; already-recorded jobs are not duplicated.
```

Phase 2 is **not** submitted by any phase-1 command. After all phase-1 trajectories finish and their TEST evaluations are sealed:

```bash
bash scripts/adaptive/submit_task1_closed_loop.sh --submit-phase2
```

By default the `task1_validation_report.csv` contains only validation metrics:

```bash
bash scripts/adaptive/submit_task1_closed_loop.sh --report
```

Only after ALL phase-1 trajectories finish, explicitly request a combined TEST report:

```bash
bash scripts/adaptive/submit_task1_closed_loop.sh --report-with-test
```

Each L0 and cycle run saves the selected checkpoint, validation metrics, per-class counts, acquisition object IDs/composition, and TEST metrics automatically under an explicit `sealed_test/` subfolder. The next job's parent check uses completion status and training-checkpoint hashes; it never reads `sealed_test/evaluation_summary.json` metric values. These test files are physically readable to anyone with directory access; the software embargo protects default reporting and downstream decision paths, not access by an operator who opens them manually.

## Important operational notes

- Slurm GPU jobs request up to 35 minutes each (not a guaranteed duration). Cycle 9 may need more time than the 15-minute full-data Task 0 baseline because acquisition and test inference are also performed. After two real L0 / cycle jobs, inspect `sacct` before reducing time requests. CPU cache jobs and GPU fitting jobs have different resource profiles.
- If the job is interrupted **mid-fitting**, the partial output is preserved; automatic retraining does not overwrite it. Inspect the logs and checkpoint before deciding on a versioned retry. If fitting succeeded but test evaluation failed, the same job can be re-invoked: it verifies the completed fit and retries the evaluation (unless partial test files require inspection).
- The `--check` command and local unit tests do not constitute a real live-GPU integration test. Verify the six L0 fits and preferably one cycle before mass submission.
- The old ViT model uses different preprocessing and a different cache. This overlay only evaluates the new ConvNeXt model. The separate Task 0 final old-ViT comparison must still use its own matched-split audit.
- The frozen TEST cache may be built now; its completion is required for the first Task 1 GPU job but not for creating or testing this overlay.
- Model initialization from the **full-history Task 0 checkpoint is expressly prohibited**: it would leak U0 ground-truth training labels into the cycle-0 model.
