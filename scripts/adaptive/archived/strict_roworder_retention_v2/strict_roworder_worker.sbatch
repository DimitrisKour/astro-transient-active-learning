#!/bin/bash -l
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
set -euo pipefail
cycle="$1"
attempt="$2"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
ROOT=/work/hdd/bcrv/dkourtesis/applecider_adaptive/recovery/astrominn_entropy135_strict_roworder_retention_v2
RUNTIME="$ROOT/runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$RUNTIME"/{tmp,torch,matplotlib,jupyter}
export TMPDIR="$RUNTIME/tmp" TORCH_HOME="$RUNTIME/torch" MPLCONFIGDIR="$RUNTIME/matplotlib" JUPYTER_RUNTIME_DIR="$RUNTIME/jupyter"
export PYTHONPATH="$ROOT/source/src${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/strict_roworder_retention.py" job --cycle "$cycle" --attempt "$attempt"
# Dispatch exactly ONE dependent successor once this single fit is complete.
python "$ROOT/scripts/strict_roworder_retention.py" advance
