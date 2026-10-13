#!/bin/bash -l
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
plan="$1"
frozen_repo="$2"
if [[ "${CONDA_DEFAULT_ENV:-}" != al-pipeline ]]; then
  module load miniforge3-python
  eval "$(conda shell.bash hook)"
  conda activate al-pipeline
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$frozen_repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
artifact_root="$(python - "$plan" <<'PY'
import sys
from pathlib import Path
from al_pipeline.astronomy.training_diagnostics import load_plan
print(load_plan(Path(sys.argv[1]))['artifact_root'])
PY
)"
runtime="$artifact_root/runtime/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/$SLURM_ARRAY_TASK_ID"
mkdir -p "$runtime"/{tmp,matplotlib,torch}
export TMPDIR="$runtime/tmp"
export MPLCONFIGDIR="$runtime/matplotlib"
export TORCH_HOME="$runtime/torch"
cd "$frozen_repo"
echo "Training-only task $SLURM_ARRAY_TASK_ID; plan $plan"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/run_astrominn_training_diagnostic.py \
  --plan "$plan" --task-index "$SLURM_ARRAY_TASK_ID" --device cuda
