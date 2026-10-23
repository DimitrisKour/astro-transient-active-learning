# Task 7A — Temperature-scaled entropy, full warm-start closed-loop pilot

## Scientific question

Does temperature-calibrated predictive entropy select more informative objects than unscaled entropy, **with the training procedure kept identical**?

### Fixed experimental settings

| Setting | Task7A | Existing reference |
|---|---|---|
| Suite | `astrominn_closed_loop_warm_temperature_entropy_v1` | `astrominn_closed_loop_warm_v1` |
| Seeds | 135, 246, 357 | Same |
| Policy | **entropy only** | entropy |
| Initial model | Existing seed-specific Experiment B L0 | Same exact L0 |
| Weight exponent | 0.85 | 0.85 |
| Checkpoint | Weighted validation-loss-best | Same |
| Warm-start | Previous cycle selected checkpoint | Same |
| Sample weighting | Per sample | Same |
| L0/U0, split seed | Unchanged, 135 | Same |
| Acquisition fraction | 0.10, exactly 473 each early cycle | Same |
| Newly queued fits | 3 seeds × 5 cycles = **15** | No new reference fits |
| One fit | One A40 / 8 CPUs / 32 GB / **20 min** requested | Same resource family |
| Calibration fit | Latest-alert *validation object* logits, NLL-optimal one-scalar temperature | T=1 |
| Application of T | **Entropy scoring only** for unlabeled objects | No scaling |
| Pool/test labels used to fit T | **None** | N/A |

`--max-cycles 9 --complete-pool` is a **scientific configuration** in the frozen source, but the initial scheduler queues **only 1–5**, permitting a subsequent unchanged-science extension to cycle 9 if promising.

### Exactly when calibration occurs

For acquisition cycle *c*, load the checkpoint from cycle *c−1* (Experiment B's L0 at *c=1*). Load **that checkpoint's** saved validation `evaluation_summary.json` and `validation_predictions.npz` from the new Task7A trajectory. Verify checkpoint SHA-256 and five-class order, and fit one temperature on all latest-alert object-level logits by minimizing unweighted validation NLL over `T∈[0.05,20]` (same algorithm/bounds as the retrospective Task6A audit). The trained model and raw pool logits are not altered. Entropy sees `softmax(pool_logits/T)` only. The selected object IDs are revealed and added to L; then a normal `.85`, loss-best, warm-start fit is performed. Repeat.

`temperature_calibration.json` is written once per acquisition cycle with T, NLL, file hashes and source checkpoint. A retry verifies these artifacts before using the saved result.

**Known limitation:** this retrospective simulation uses validation data for both within-cycle checkpoint selection and temperature fitting; a fully independent calibration set would be cleaner, but changing the data partition for this pilot would introduce an additional variable. This experiment is designed to establish changes to acquisition quality, not to claim unbiased calibration metrics. The test set remains report-only, not a tuning criterion.

### Safety and isolation

- *Only* Task7A-specific repo scripts are added. The production `simulated.py` and historical runner are patched **in memory into a new frozen Task7A source**; originals stay unchanged.
- `prepare` freezes a manifest of all copied Python files, the registry and the scientific contract. It invokes the no-`--run` CPU preflight for all seeds to materialize the historical split **before** comparing it with original L0/U0.
- 15 one-GPU jobs form **three independent dependency chains**. Each cN depends only on the preceding c(N−1) of the **same seed**; cycle 1 has no dependencies. No new L0 fits are scheduled.
- In `submit`, all jobs are queued held and released in reverse order, with failure/cancellation marker handling. `submit` without `--submit` is a dry run. Submission markers prevent accidental duplicate submission.
- Run-state identities include the temperature flag. The first cycle verifies original L0 checkpoint metadata and SHA when loading its validation evaluation; subsequent cycles verify previous selected checkpoint.
- The report compares **matched seeds and labeled-object budgets**, exports macro-F1, balanced accuracy, per-class recall/F1, temperatures, and same-cycle selected-ID overlap. It **does not use the test set to choose configurations**.

### Delta installation

From PowerShell:

```powershell
scp "$HOME\Downloads\task7a_temperature_entropy_pilot_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

On Delta:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/task7a_temperature_entropy_pilot_v1.zip -d .
python -m pytest tests/jobs/test_task7a_v1.py -q
python scripts/adaptive/task7a.py prepare
python scripts/adaptive/task7a.py verify
python scripts/adaptive/task7a.py plan
```

`prepare` **never submits jobs** and its CPU preflight **does not train**. The frozen source is in `/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_temperature_entropy_v1/source`.

**When ready to queue (after Task6C as requested):**

```bash
python scripts/adaptive/task7a.py submit --walltime 00:20:00
python scripts/adaptive/task7a.py submit --walltime 00:20:00 --submit
squeue -u "$USER" -o '%.18i %.32j %.2t %.10M %R'
```

The first command is a **dry run**; the second is the one that actually submits jobs.

### Reporting

```bash
python scripts/adaptive/task7a_report.py
```

Report output: `/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_temperature_entropy_v1/report/`.

### Decision

Decide whether to extend the **same** immutable Task7A trajectories through cycles 6–9 based on validation early-cycle learning curves, class-specific recall (especially TDE, SN II, CV), acquisition difference, and trajectory stability. Do not select using test metrics. A change in selected IDs alone is insufficient evidence of better AL.
