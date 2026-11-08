#!/bin/bash
set -euo pipefail
TASK6B_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TASK6B_SUITE_ID=astrominn_closed_loop_warm_f1best_085_v1
TASK6B_BASE_SUITE_ID="$TASK6B_SUITE_ID"
TASK6B_REFERENCE_SUITE_ID=astrominn_closed_loop_warm_v1
TASK6B_PERSISTENT=/projects/bcrv/dkourtesis/applecider_adaptive
TASK6B_DATA=/work/hdd/bcrv/dkourtesis/applecider_adaptive
TASK6B_JOB_DIR="$TASK6B_PERSISTENT/closed_loop/$TASK6B_SUITE_ID"
TASK6B_SOURCE="$TASK6B_JOB_DIR/source"
TASK6B_REGISTRY="$TASK6B_JOB_DIR/registry.toml"
TASK6B_SOURCE_REGISTRY="${TASK6B_SOURCE_REGISTRY:-$TASK6B_PERSISTENT/closed_loop/$TASK6B_REFERENCE_SUITE_ID/registry.toml}"

task6b_activate() {
  module load miniforge3-python
  eval "$(conda shell.bash hook)"
  conda activate al-pipeline
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  export PYTHONUNBUFFERED=1
}

task6b_arguments() {
  local seed="$1" policy="$2"
  case "$seed" in 135|246|357) ;; *) echo "Unsupported seed: $seed" >&2; return 2;; esac
  case "$policy" in entropy|entropy_balanced_explore) ;; *) echo "Unsupported policy: $policy" >&2; return 2;; esac
  TASK6B_ARGS=(
    --registry "$TASK6B_REGISTRY" --suite-id "$TASK6B_SUITE_ID"
    --bootstrap-source-suite-id "$TASK6B_SUITE_ID"
    --training-initialization warm_start
    --strategies "$policy" --seed "$seed" --split-seed 135
    --initial-fraction 0.10 --minimum-initial-per-class 5
    --acquisition-fraction 0.10 --max-cycles 9 --complete-pool
    --max-epochs 50 --patience 10 --batch-size 512
    --validation-batch-size 2048 --learning-rate 0.0001
    --train-workers 0 --validation-workers 0
    --inference-workers 0 --evaluation-workers 0
    --shortlist-multiplier 3 --balance-alpha 1 --balance-beta 0.5
    --balance-strength 1 --balance-cap 3 --exploration-fraction 0.2
    --assert-excluded-manifest "$TASK6B_PERSISTENT/runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv"
  )
}
