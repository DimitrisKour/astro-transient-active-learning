# Task 13: TAILOR versus uniform acquisition-policy mixture

Status: source and CPU regression checks prepared; **no Delta GPU jobs have been submitted**.

This is an offline, simulated-label, **frozen-validation-only** AstroMiNN
pilot. It reuses the Task4 historical L0/U0 and seed-specific L0 checkpoints,
the Task11 frozen training and evaluation code, and the class-imbalance reward
of Zhang et al., *Algorithm Selection for Deep Active Learning with Imbalanced
Datasets* (NeurIPS 2023), Algorithm 2.

The comparison is:

- `tailor_diversity`: per-object Thompson selection of acquisition arms;
  a distinct 5-class Dirichlet posterior is maintained for every arm.
- `uniform_mixture`: independently and uniformly select an arm at every
  object position; all other settings and constituent algorithms are identical.

The acquisition arms are `entropy_diversity`,
`entropy_balanced_explore`, and `fisher_batch`. Both controllers consume the
same 473-object (nominal) batch per round; the actual count comes from the
existing historical-object budget rule. Posterior updates occur only **after**
revealing the acquired labels, and use only confirmed labels. TAILOR's reward
is `1 / (5 * max(1, count_of_previously_confirmed_objects_in_class))`.
The Dirichlet prior is `1` per class; past observations are discounted by 0.9.

The study consists of 2 controllers x 3 seeds x 4 cycles = **24 fits**.
Cycle zero is the unchanged seed-specific bootstrap evaluation, not a fit.
Task11 and Task12 are never modified by these scripts.

## Deployment prerequisites

- A checkout of the private Git repository on branch `feat/task13-tailor`.
- The already verified Task11 frozen source and its `frozen_manifest.json`.
- Task4 source outputs and historical replay files; Task4 L0/U0 equality is
  audited before any acquisition.
- The `al-pipeline` conda environment and the historical ViT artifacts.
- Free file space and quota for 24 small run-state trees plus checkpoints.

On a **Delta login node**, use a separate Git worktree so the current
`active-learning` checkout is not switched or modified. Check first that
`../active-learning-task13` does not already contain another worktree:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
git status --short
git fetch origin
git worktree add --detach ../active-learning-task13 origin/feat/task13-tailor
cd ../active-learning-task13
git status --short

module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

python scripts/research/historical_replay/task13/task13.py plan
python scripts/research/historical_replay/task13/task13.py prepare --repo "$PWD"
python scripts/research/historical_replay/task13/task13.py verify
python scripts/research/historical_replay/task13/task13.py preflight --pilot-seed 135
python scripts/research/historical_replay/task13/task13.py pilot-status
```

`prepare` constructs a **new** frozen source under
`/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task13_tailor_pilot_v1`.
Source hashes are checked against Task11 and frozen after the Task13 overlay.
Existing frozen suites remain untouched. `preflight` materializes and audits
the split and verifies the existing L0 checkpoint; it does not train. It also
checks the `/work/hdd` checkpoint filesystem using a disposable 16 MiB
write, `fsync`, rename and read-back probe; the probe is removed automatically.
Every subsequent GPU worker repeats that safety probe before running.

### GPU acquisition pilots (no training)

Run these on a Delta **login node** to submit GPU pilots:

```bash
TASK13_JOB=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_task13_tailor_pilot_v1

sbatch --parsable --time=00:45:00 --job-name=T13_pilot_tailor \
  --output="$TASK13_JOB/logs/%x_%j.out" --error="$TASK13_JOB/logs/%x_%j.err" \
  "$TASK13_JOB/source/scripts/research/historical_replay/task13/task13_worker.sbatch" \
  "$TASK13_JOB" pilot tailor_diversity 135

sbatch --parsable --time=00:45:00 --job-name=T13_pilot_uniform \
  --output="$TASK13_JOB/logs/%x_%j.out" --error="$TASK13_JOB/logs/%x_%j.err" \
  "$TASK13_JOB/source/scripts/research/historical_replay/task13/task13_worker.sbatch" \
  "$TASK13_JOB" pilot uniform_mixture 135
```

The pilots compute candidate and confirmed-anchor inference, exercise the
full acquisition batch, validate distinct object IDs, and save arm-allocation
diagnostics under `pilot/<policy>/seed_135/result.json`. They do **not**
query hidden labels, train, or evaluate.

Inspect the two pilot JSONs and corresponding Slurm logs. If pilots fail or
need more time, diagnose before proceeding. In particular, benchmark Fisher
selection latency because mixed-policy information updates occur after every
selection, not just Fisher selections.

### Submit, inspect, release

Only after both pilots pass:

```bash
python scripts/research/historical_replay/task13/task13.py pilot-status
python scripts/research/historical_replay/task13/task13.py submit --walltime 00:45:00
```

This submits **24 held jobs** with independent `afterok` cycle chains for
each policy and seed. `submitted.json` records each Slurm job ID and parent.
Submitting does not release any jobs.

Inspect:

```bash
python scripts/research/historical_replay/task13/task13.py status
squeue -u dkourtesis -o '%.18i %.25j %.2t %.10M %.10l %R'
```

Once satisfied that all six chains are correct, explicitly release:

```bash
python scripts/research/historical_replay/task13/task13.py release
```

`release` is retry-safe if interrupted. Every worker verifies source hashes
before doing any GPU work. If a cycle fails, its `afterok` descendants cannot
run until the failed stage is recovered and dependencies are repaired.
**Never force-release a dependent cycle that lacks its previous checkpoint.**

### Result export and transfer

Collect compact results on Delta without model weights, raw training samples,
or the full hidden-label pool:

```bash
python scripts/research/historical_replay/task13/task13.py collect \
  --destination /u/dkourtesis/task13_results_20261028.zip
```

On the Windows **PowerShell** laptop terminal:

```powershell
scp dkourtesis@dt-login.delta.ncsa.illinois.edu:/u/dkourtesis/task13_results_20261028.zip "$HOME\Downloads\task13_results_20261028.zip"
```

Upload that ZIP to ChatGPT. It contains acquired object IDs and revealed
labels, cycle and validation summaries, per-class metrics, and the controller
allocation traces needed for AULC and stability comparisons.

## Local CPU checks

From the repo root (with the package installed in editable mode):

```bash
python -m unittest discover -s tests -p test_task13_core.py -v
```

Regression tests verify that giving all batch positions to one arm reproduces
the existing entropy-diversity, balanced exploration, or Fisher selector.
They also check reward values, posterior discounting, repeatability,
uniform-mixture independence of class rewards, and unique selections.

## Limits and interpretation

- Unlike a direct macro-F1 optimizer, original TAILOR rewards the discovery
  of underrepresented classes, not validation performance.
- The same model is trained once per acquisition cycle; there is no ensemble.
- Sequential interleaving preserves each existing arm in the all-one-arm
  case. When different arms are interleaved, they interact through the
  shared selected set; this is unavoidable for a sequential mixed batch.
- Predefine the primary comparisons using object-level validation macro-F1
  and balanced-accuracy AULC over cycles 0–4. Examine SN I false alarms,
  rare-class recall/F1, and negative cycle-to-cycle performance transitions.
- Because the user has only three seeds and nine original cycles, avoid
  treating tiny apparent differences as decisive or repeatedly tuning to
  frozen validation results. Keep the frozen test untouched.
