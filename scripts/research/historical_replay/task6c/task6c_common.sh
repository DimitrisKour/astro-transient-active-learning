#!/bin/bash
set -euo pipefail
TASK6C_PERSISTENT=/projects/bcrv/dkourtesis/applecider_adaptive
TASK6C_DATA=/work/hdd/bcrv/dkourtesis/applecider_adaptive

task6c_config() {
  case "$1" in
    loss)
      TASK6C_SUITE_ID=astrominn_closed_loop_equal_object_loss_085_v1
      TASK6C_BOOTSTRAP_SUITE=astrominn_historical_replay_v1
      TASK6C_REFERENCE=astrominn_closed_loop_warm_v1
      ;;
    macro_f1)
      TASK6C_SUITE_ID=astrominn_closed_loop_equal_object_f1best_085_v1
      TASK6C_BOOTSTRAP_SUITE=astrominn_closed_loop_warm_f1best_085_v1
      TASK6C_REFERENCE=astrominn_closed_loop_warm_f1best_085_v1
      ;;
    *) echo "Unexpected Task6C checkpoint selection $1" >&2; return 2;;
  esac
  TASK6C_JOB_DIR="$TASK6C_PERSISTENT/closed_loop/$TASK6C_SUITE_ID"
  TASK6C_REGISTRY="$TASK6C_JOB_DIR/registry.toml"
}

task6c_activate() {
  module load miniforge3-python
  eval "$(conda shell.bash hook)"
  conda activate al-pipeline
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  export PYTHONUNBUFFERED=1
}

task6c_arguments() {
  local selection="$1" seed="$2" policy="$3"
  task6c_config "$selection"
  case "$seed" in 135|246|357) ;; *) echo "Invalid seed $seed" >&2; return 2;; esac
  case "$policy" in entropy|entropy_balanced_explore) ;; *) echo "Invalid policy $policy" >&2; return 2;; esac
  TASK6C_ARGS=(
    --registry "$TASK6C_REGISTRY" --suite-id "$TASK6C_SUITE_ID"
    --bootstrap-source-suite-id "$TASK6C_BOOTSTRAP_SUITE"
    --require-existing-bootstrap --training-initialization warm_start
    --strategies "$policy" --seed "$seed" --split-seed 135
    --initial-fraction 0.10 --minimum-initial-per-class 5
    --acquisition-fraction 0.10 --max-cycles 9 --complete-pool
    --max-epochs 50 --patience 10 --batch-size 512
    --validation-batch-size 2048 --learning-rate 0.0001
    --train-workers 0 --validation-workers 0
    --inference-workers 0 --evaluation-workers 0
    --shortlist-multiplier 3 --balance-alpha 1 --balance-beta 0.5
    --balance-strength 1 --balance-cap 3 --exploration-fraction 0.2
    --assert-excluded-manifest "$TASK6C_PERSISTENT/runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv"
  )
}
