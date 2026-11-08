#!/bin/bash
# Task 4 is intentionally isolated from the original policy suite.
set -euo pipefail
TASK4_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TASK4_SUITE_ID="astrominn_closed_loop_warm_v1"
TASK4_BASE_SUITE_ID="astrominn_historical_replay_v1"
TASK4_PERSISTENT="/projects/bcrv/dkourtesis/applecider_adaptive"
TASK4_DATA="/work/hdd/bcrv/dkourtesis/applecider_adaptive"
TASK4_JOB_DIR="$TASK4_PERSISTENT/closed_loop/$TASK4_SUITE_ID"
TASK4_SOURCE="$TASK4_JOB_DIR/source"
TASK4_REGISTRY="$TASK4_JOB_DIR/registry.toml"
TASK4_SOURCE_REGISTRY="${TASK4_SOURCE_REGISTRY:-$TASK4_REPO/configs/delta.toml}"
if [[ ! -f "$TASK4_SOURCE_REGISTRY" ]]; then
  TASK4_SOURCE_REGISTRY="$TASK4_PERSISTENT/cycle_jobs_v1/registry_hdd.toml"
fi
TASK4_EARLY_WALLTIME="${TASK4_EARLY_WALLTIME:-00:15:00}"
TASK4_LATE_WALLTIME="${TASK4_LATE_WALLTIME:-00:20:00}"

task4_activate() {
    module load miniforge3-python
    eval "$(conda shell.bash hook)"
    conda activate al-pipeline
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
    export PYTHONUNBUFFERED=1
}

task4_arguments() {
    local seed="$1" policy="${2:-}"
    local strategies=(random entropy entropy_diversity entropy_balanced_explore)
    if [[ -n "$policy" ]]; then
      case "$policy" in
        random|entropy|entropy_diversity|entropy_balanced_explore) strategies=("$policy");;
        *) echo "Unsupported Task4 policy: $policy" >&2; return 2;;
      esac
    fi
    TASK4_ARGS=(
      --registry "$TASK4_REGISTRY" --suite-id "$TASK4_SUITE_ID"
      --bootstrap-source-suite-id "$TASK4_BASE_SUITE_ID"
      --require-existing-bootstrap --training-initialization warm_start
      --strategies "${strategies[@]}"
      --seed "$seed" --split-seed 135
      --initial-fraction 0.10 --minimum-initial-per-class 5
      --acquisition-fraction 0.10 --max-cycles 9 --complete-pool
      --max-epochs 50 --patience 10 --batch-size 512
      --validation-batch-size 2048 --learning-rate 0.0001
      --train-workers 0 --validation-workers 0
      --inference-workers 0 --evaluation-workers 0
      --shortlist-multiplier 3 --balance-alpha 1 --balance-beta 0.5
      --balance-strength 1 --balance-cap 3 --exploration-fraction 0.2
      --assert-excluded-manifest
      "$TASK4_PERSISTENT/runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv"
    )
}
