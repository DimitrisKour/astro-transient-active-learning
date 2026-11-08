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
CONTROLLER="$1"; POLICY="$2"; SEED="$3"; CYCLE="$4"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
DATA=/work/hdd/bcrv/dkourtesis/applecider_adaptive
RUNTIME="$DATA/runs/astrominn_task13_tailor_pilot_v1/shared/extension_runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$RUNTIME"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$RUNTIME/tmp"
export MPLCONFIGDIR="$RUNTIME/matplotlib"
export TORCH_HOME="$RUNTIME/torch"
export JUPYTER_RUNTIME_DIR="$RUNTIME/jupyter"
python "$CONTROLLER/source/task13_extension.py" job "$POLICY" "$SEED" "$CYCLE" --controller "$CONTROLLER"
