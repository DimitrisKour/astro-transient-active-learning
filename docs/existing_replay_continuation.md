# Existing historical replay: reporting, final cycle and new policies

Activate `al-pipeline` and run these commands from the patched repository.
The original nine runs have eight cycles and 4,260 of 4,729 training objects.
The final acquisition is 469 objects. Do not manually edit experiment.json.

## Plot the completed eight-cycle experiments

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
python scripts/report_astrominn_multiseed.py \
  --suite historical_replay \
    /projects/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_historical_replay_v1 \
    /work/hdd/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_historical_replay_v1 \
  --strategies random margin entropy --seeds 135 246 357 \
  --expected-cycles 8 --metrics f1_macro accuracy balanced_accuracy \
  --splits validation test \
  --output /projects/bcrv/dkourtesis/applecider_adaptive/reports/historical_cycles1to8
```

This needs no GPU training. It writes PNG/PDF figures and validated CSV tables.

## Prepare a common source snapshot and complete cycle 9

```bash
export POLICY_SUITE_ID=astrominn_historical_replay_v1
bash scripts/adaptive/submit_policy_compare.sh 135 --prepare-only
```

This is preparation only: no jobs or training. It reuses the original registry
at cycle_jobs_v1/registry_hdd.toml, so ignored local config files are not needed.
It freezes the patched source,
checks input paths and shared/HDD writes, and requires the existing bootstrap.
Do not edit source after preparation while any of these jobs remain queued/running.

Submit one final-cycle job for each seed:

```bash
for seed in 135 246 357; do
  sbatch --job-name="alB_s${seed}_c9" \
    --output="/projects/bcrv/dkourtesis/applecider_adaptive/cycle_jobs/astrominn_historical_replay_v1/logs/%x_%j.out" \
    --error="/projects/bcrv/dkourtesis/applecider_adaptive/cycle_jobs/astrominn_historical_replay_v1/logs/%x_%j.err" \
    scripts/adaptive/historical_final_cycle.sbatch "$PWD" "$seed"
done
```

Each job runs only cycle 9 for random, margin and entropy. The explicit budget
extension validates all strategies before changing any identity, verifies the
previous completed training output against the requested training settings,
and archives the old identity as experiment_before_budget_extension_008.json.
It changes only max_cycles and allow_partial_final_cycle. Earlier selections,
manifests, evaluations and checkpoints are retained. Completed cycle 9 is
skipped on retry; partial training may restart if no completed stage exists.
There is no epoch/optimizer-level continuation.

## Run only the two new policies

With POLICY_SUITE_ID still set to astrominn_historical_replay_v1:

```bash
bash scripts/adaptive/submit_policy_compare.sh 135 01:00:00
# Review this seed before launching the remaining two.
bash scripts/adaptive/submit_policy_compare.sh 246 01:00:00
bash scripts/adaptive/submit_policy_compare.sh 357 01:00:00
```

Each chain has nine jobs, each running entropy_diversity and
entropy_balanced_explore only. They share the immutable split and existing
validated bootstrap. A missing or incompatible bootstrap fails rather than
training it. Completed random/entropy states are not rejected or retrained.
The final-cycle baseline jobs and new-policy chains own separate strategy
state directories, but report only after all required runs have completed.

## Plot the four-policy comparison when every required run has cycle 9

Use the plotting command above with:

```text
--strategies random entropy entropy_diversity entropy_balanced_explore
--expected-cycles 9
--output /projects/bcrv/dkourtesis/applecider_adaptive/reports/four_policy_cycles1to9
```

The plotting validator checks common data fingerprints and identical
seed-specific bootstrap hashes. The two new policies have distinct colors.
The original margin results remain available separately.

Use scripts/report_policy_compare.py for the additional curve-summary tables;
it does not itself render plots. Its current summaries scan all strategies
present in the suite, including margin. No change to early-cycle evaluation
criteria is introduced here.
