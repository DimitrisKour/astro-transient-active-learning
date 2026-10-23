#!/bin/bash
set -euo pipefail
TASK7A_SUITE=astrominn_closed_loop_warm_temperature_entropy_v1
TASK7A_BASE=astrominn_historical_replay_v1
TASK7A_PERSISTENT=/projects/bcrv/dkourtesis/applecider_adaptive
TASK7A_DATA=/work/hdd/bcrv/dkourtesis/applecider_adaptive
TASK7A_JOB_DIR="$TASK7A_PERSISTENT/closed_loop/$TASK7A_SUITE"
TASK7A_SOURCE="$TASK7A_JOB_DIR/source"
TASK7A_REGISTRY="$TASK7A_JOB_DIR/registry.toml"

task7a_activate() {
  module load miniforge3-python
  eval "$(conda shell.bash hook)"
  conda activate al-pipeline
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  export PYTHONUNBUFFERED=1
}

task7a_arguments() {
  local seed="$1"
  case "$seed" in 135|246|357) ;; *) echo "Unknown Task7A seed $seed" >&2; return 2;; esac
  TASK7A_ARGS=(
    --registry "$TASK7A_REGISTRY" --suite-id "$TASK7A_SUITE"
    --bootstrap-source-suite-id "$TASK7A_BASE" --require-existing-bootstrap
    --training-initialization warm_start --strategies entropy
    --seed "$seed" --split-seed 135
    --initial-fraction 0.10 --minimum-initial-per-class 5
    --acquisition-fraction 0.10 --max-cycles 9 --complete-pool
    --max-epochs 50 --patience 10 --batch-size 512
    --validation-batch-size 2048 --learning-rate 0.0001
    --train-workers 0 --validation-workers 0
    --inference-workers 0 --evaluation-workers 0
    --shortlist-multiplier 3 --balance-alpha 1 --balance-beta 0.5
    --balance-strength 1 --balance-cap 3 --exploration-fraction 0.2
    --assert-excluded-manifest "$TASK7A_PERSISTENT/runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv"
  )
}
