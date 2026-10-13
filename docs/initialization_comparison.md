# Experiment B: matched initialization comparison

B so far used `restart_from_random`. This was verified in all 12 experiment configurations and all 108 cycle-1–9 training summaries from the completed four-policy, three-seed archive. Every summary records `initialization: random_vit` and `initial_checkpoint: null`. The previous model supplies acquisition scores, while training creates a new ViT from that seed's original random starting point. Published expert towers remain frozen. This differs from Experiment A's warm-start and published-baseline-restart runs.

This suite compares the existing random-restart results with two new training variants:

- `restart_from_l0`: load the existing checkpoint trained only on the initial historical labeled subset for that seed, at every cycle.
- `warm_start`: start cycle 1 from that same L0 checkpoint; subsequent cycles load this new warm-start trajectory's preceding validation-selected checkpoint.

Both variants use each original policy's saved cumulative manifests, byte-for-byte and in their original order. The policies are random, entropy, entropy plus diversity, and entropy plus soft class balancing with exploration (`entropy_balanced_explore`, the existing fourth policy). No new pool inference, selection or label resolution runs. These are conditional training comparisons on the original policy trajectories. They do not estimate how a warm-started acquisition loop would select different objects.

## Scientific controls and source checks

The same loss, class-weight rule, loader seed, epoch limit (50 in the completed suite), patience (10), augmentation, frozen experts and weighted sample-level validation-loss checkpoint rule are retained. Validation and test objects remain excluded from training. Test metrics are for reporting only.

Warm-start carries ViT weights only. A fresh optimizer, scheduler and learning-rate schedule are created at every cycle, including warm-start. This keeps optimizer continuation out of the first ablation. The selected checkpoint from each prior cycle, not its final epoch state, is passed forward. Warm-start inherits more historical training than L0 restart: per-cycle settings match, but total training exposure is not equal.

Preparation validates nested object sets and complete sample-row multisets, original manifest hashes, completed source cycles, recipes, immutable evaluation split counts, agreement between evaluation history and saved summaries, common L0 identity, and the per-seed bootstrap checkpoint SHA from the original experiment. At cycle 9, model-equivalent historical rows must match (native/canonical label aliases and omitted survey metadata are handled explicitly), and final raw replay row content must match across policies/seeds. Other fields and duplicate row multiplicities remain exact.

Runtime audits the actual loaded initial ViT state and checkpoint identity. Explicit checkpoints are mandatory; the published full-data ViT is never a fallback. Historical sample arrays are checked for existence, while manifests, checkpoints, external model source and expert artifacts are hashed. Preparation does not hash all sample-array bytes.

Inputs and code are frozen before submission. Existing AL directories and queued training-diagnostic sources are not modified. Metadata/results live under `/projects/bcrv/dkourtesis/applecider_adaptive/initialization_compare/<id>`, heavy outputs under the corresponding `/work/hdd/...` directory.

## GPU budget and jobs

Three seeds × four trajectories × nine cycles × two new modes would cost 216 fits. Cycle 1 is identical for both modes, so one fit per trajectory is shared. The default plan therefore requests **204 new fits**, reuses **108 existing random-restart fits** and **three existing L0 checkpoints**, and saves **12 duplicate cycle-1 fits**. No L0 retraining or random-restart retraining is scheduled.

There are 27 Slurm arrays: one per seed/cycle. Each element trains exactly one model and evaluates validation/test. Cycle 1 has four fits per seed; cycles 2–9 have eight. Every next-cycle array depends on successful completion of the previous array for that seed. Seeds have independent chains. Default concurrency is one fit per seed, at most three GPUs across this suite (other submitted suites have their own concurrency).

Default time requests:

| Cycles | Requested time per fit |
|---|---|
| 1–2 | 10 minutes |
| 3–5 | 15 minutes |
| 6–9 | 20 minutes |

These keep one model per request rather than placing eight fits inside a long allocation. The completed random-restart fits took 1.9–7.4 minutes of training, depending on cycle and stopping epoch. New warm starts can run to the 50-epoch cap: extrapolating the recorded per-epoch times gives about 5.6–17.6 training minutes across cycles, before loading/evaluation. Shorter requests should be chosen after observing this suite, not by reducing the scientific epoch budget. Jobs exit when their work finishes; the requested limit does not force them to run for the full allocation. No claim is made about queue wait times.

Arrays are submitted held and released in reverse order. Submission errors cancel only the arrays created by that submission attempt. A file lock and durable job record block accidental duplicate submission. Retry requires previous jobs to stop, validates completed checkpoint hashes, and queues only missing fits. Completed training can finish evaluation without retraining; an interrupted training attempt is preserved and restarted in a new attempt directory, since the existing trainer has no epoch-resume implementation. Failed predecessor arrays stop later cycles for that seed.

## Installation and launch on Delta

Extract the code ZIP in the checkout:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -o ~/initialization_compare_v1.zip -d .
bash scripts/adaptive/submit_initialization_compare.sh --prepare-only
```

After preparation succeeds, submit all three seeds with the default times:

```bash
bash scripts/adaptive/submit_initialization_compare.sh
squeue -u "$USER" -o '%.18i %.28j %.2t %.10l %R'
```

Explicit times, if wanted:

```bash
bash scripts/adaptive/submit_initialization_compare.sh 00:10:00 00:15:00 00:20:00
```

A one-seed pilot is supported using its own immutable comparison ID. It runs all four trajectories and both new variants, 68 fits. A later all-seed suite must use a different ID; separate plans are not silently merged or reused.

```bash
INIT_ID=astrominn_init_pilot135 INIT_SEEDS=135 \
  bash scripts/adaptive/submit_initialization_compare.sh --prepare-only
INIT_ID=astrominn_init_pilot135 INIT_SEEDS=135 \
  bash scripts/adaptive/submit_initialization_compare.sh
```

If comparing all seeds in one suite, the default all-seed plan avoids duplicate pilot fits. Further options: `INIT_MAX_CYCLE=5` for an early-cycle-only plan (use a new ID), `INIT_PARALLEL_PER_SEED=2` for at most six concurrent GPUs in this suite, and `INIT_REGISTRY`, `INIT_SOURCE_SUITE`, `INIT_SOURCE_ARTIFACT_ROOT`, `INIT_PERSISTENT_ROOT`, `INIT_DATA_ROOT`, `INIT_LEGACY_CODE_DIR` for paths. Plan-changing settings must remain identical on every subsequent invocation.

After failed/timed-out jobs and their dependents have stopped:

```bash
bash scripts/adaptive/submit_initialization_compare.sh --retry-incomplete
```

Increase relevant time arguments on retry if necessary; time limits are scheduler settings and do not alter the immutable training plan. Do not remove submission records to resubmit.

## Reports and small result export

```bash
PLAN=/projects/bcrv/dkourtesis/applecider_adaptive/initialization_compare/astrominn_initialization_compare_v1/plan.json
python scripts/report_initialization_compare.py --plan "$PLAN"
python scripts/export_initialization_compare.py --plan "$PLAN" \
  --output ~/initialization_compare_results_v1.tar.gz
```

Export excludes raw arrays, checkpoints and frozen source; add `--include-logs` if a failed job needs diagnosis. Reporting works from a downloaded plan and small task-result directory, without the original Delta paths or checkpoints. Matplotlib is optional (`--no-plots`).

Outputs include matched per-seed initialization differences, early cycle-1–5 means, cycle-6→7 changes, cycle-5/6 gaps relative to each method's own full-data endpoint, downward steps on complete trajectories, seed SD, runtime tables, new-fit per-class metrics, and validation/test learning-curve PDFs. Cycle-1 results appear under both logical variants but count only once in actual new-fit totals. Original random-restart metrics are clearly marked as reused. Aggregate plots omit incomplete seed groups rather than mixing different seeds; tables report partial counts explicitly. Three-seed SD is not a confidence interval and no test-curve peak is selected as a stopping rule.

## Scope of this code package

New comparison modules/scripts/tests plus the already fixed `training_diagnostics.py` helper used for hashes, source validation and manifest checks. Existing training/model/AL-loop implementations are unchanged. This does not submit jobs from ChatGPT; launch the commands on Delta.
