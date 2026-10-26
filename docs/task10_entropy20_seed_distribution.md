# Task 10: entropy seed 357 / cycle 5, 20 independent training seeds

## Question
At a known problematic entropy acquisition cycle, does a larger *predeclared* collection of independent training restarts provide sustained improvements over 1–4 tries? How broad is the distribution of class recalls, and what happens to fixed K-member probability ensembles? This is a fixed-L5, single-transition pilot, **not a closed-loop continuation**.

## Frozen inputs
- Task 4 ordinary entropy seed 357, **cycle 5** (first rejection by macro-F1 1 pp rule; cycle 4 incumbent).
- Exactly the original cumulative L5 manifest and original Task 4 cycle-4 initialization checkpoint (hash-checked).
- The Task 4 original training policy from Task8B evidence, including 0.85 class-weight exponent, per-sample weighting, weighted-validation-loss-best epoch, full 50-epoch/10-patience training.
- New seeds **2001 through 2020**, all preselected in order. Each uses a fresh optimizer, RNG, DataLoader seed; no change to data, labels or acquisition.
- Validation-only evaluation; no test artifacts used, no model chosen to update an active-learning run.

## Execution and storage
Dedicated directory on HDD:
`/work/hdd/bcrv/dkourtesis/applecider_adaptive/recovery/astrominn_task10_entropy_s357_c5_seed20_v1`

The source is frozen from the verified Task8B/Task4 trainer, with only Task10 orchestration/report scripts added. Original Task8B and Task4 remain untouched.

Commands from editable active-learning repository, with `conda activate al-pipeline`:

```bash
python -m pytest tests/jobs/test_task10_entropy20.py -q
python scripts/adaptive/task10_entropy20.py plan
python scripts/adaptive/task10_entropy20.py prepare
python scripts/adaptive/task10_entropy20.py verify
python scripts/adaptive/task10_entropy20.py submit --walltime 00:20:00 --max-concurrent 2
python scripts/adaptive/task10_entropy20.py status
# Only after all 20 finish:
python scripts/adaptive/task10_entropy20_report.py
```

One **20-element SLURM GPU array, max two concurrent**, training seeds 2001..2020. Each element is independent; jobs may run in parallel. Submission marker and frozen plan block accidental resubmission. Stdout/err in `logs/T10_<arrayjob>_<task>.{out,err}`. To inspect status use `sacct -j <array job ID> --format=JobID,JobName,State,ExitCode,Elapsed`.

**Failure handling:** do not delete partial training directories and do not resubmit the entire array. The original trainer saves a last-state file but has no integrated exact resume function. Investigate any individual failed slot and its partial artifacts before re-running it; the script refuses automatic overwrite.

## Report
- `individual_and_ensemble_comparisons.csv`: 20 singles and fixed E2/E4/E8/E12/E16/E20 ensembles, with and without incumbent anchor, versus both incumbent and original rejected.
- `class_recall.csv`: per-class recall/F1 and correct-object changes, aligned by validation-object ID.
- `recovery_by_budget.csv`: predeclared order 1,2,3,4,5,8,10,12,16,20; first pass and descriptive best-of-k (optimistically biased).
- `ensemble_definitions.csv`: exact checkpoint membership.
- `summary.json`: success frequency and aggregate cost.

The same validation set has been repeatedly inspected in previous tasks. This is an **exploratory seed-distribution study**; no statistical intervals can establish independent generalization without separate data. The 20 independent seeds are not necessarily independent performance draws if training randomness has shared sources.

## Design note
Existing Task8B retries 1001..1005 for this case are retained as a separate prior pilot. None is silently mixed with the new fixed 20-seed sample. They may be compared afterward, with appropriate selection-bias caveats.
