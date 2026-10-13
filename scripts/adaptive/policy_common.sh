#!/bin/bash
# Sourced by the preparation, submission and GPU job entry points.
set -euo pipefail
POLICY_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_SUITE_ID="${POLICY_SUITE_ID:-astrominn_historical_replay_v1}"
[[ "$POLICY_SUITE_ID" == astrominn_historical_replay_v1 || "$POLICY_SUITE_ID" =~ ^astrominn_policy_compare_[A-Za-z0-9_-]+$ ]] || {
  echo 'Use astrominn_historical_replay_v1 or a suite beginning astrominn_policy_compare_.' >&2; exit 1;
}
POLICY_JOB_DIR="/projects/bcrv/dkourtesis/applecider_adaptive/cycle_jobs/$POLICY_SUITE_ID"
POLICY_DATA_DIR="/work/hdd/bcrv/dkourtesis/applecider_adaptive"
if [[ "$POLICY_SUITE_ID" == astrominn_historical_replay_v1 ]]; then
  POLICY_SOURCE_REGISTRY="/projects/bcrv/dkourtesis/applecider_adaptive/cycle_jobs_v1/registry_hdd.toml"
else
  POLICY_SOURCE_REGISTRY="$POLICY_REPO/configs/delta.toml"
fi

policy_activate() {
  module load miniforge3-python
  eval "$(conda shell.bash hook)"
  conda activate al-pipeline
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  export PYTHONPATH="$POLICY_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONUNBUFFERED=1
  cd "$POLICY_REPO"
}

policy_arguments() {
  POLICY_ARGS=(
    --registry "$POLICY_JOB_DIR/registry_hdd.toml"
    --suite-id "$POLICY_SUITE_ID"
    --strategies entropy_diversity entropy_balanced_explore
    --seed "$1" --split-seed 135
    --initial-fraction 0.10 --minimum-initial-per-class 5
    --acquisition-fraction 0.10 --complete-pool
    --max-epochs 50 --patience 10 --batch-size 512
    --validation-batch-size 2048 --learning-rate 0.0001
    --train-workers 0 --validation-workers 0 --inference-workers 0 --evaluation-workers 0
    --shortlist-multiplier 3 --balance-alpha 1 --balance-beta 0.5
    --balance-strength 1 --balance-cap 3 --exploration-fraction 0.2
    --assert-excluded-manifest
    /projects/bcrv/dkourtesis/applecider_adaptive/runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv
  )
  if [[ "$POLICY_SUITE_ID" == astrominn_historical_replay_v1 ]]; then
    POLICY_ARGS+=(--require-existing-bootstrap)
  fi
}
