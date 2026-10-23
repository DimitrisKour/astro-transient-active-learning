#!/bin/bash -l
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
# Explicit submission only; all execution uses frozen code and a hashed plan.
set -euo pipefail
phase="${1:?5b|5c}"
work="${2:?work-root}"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$work/source/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
runtime="/work/hdd/bcrv/dkourtesis/applecider_adaptive/representation_5bc_runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$runtime"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$runtime/tmp"
export MPLCONFIGDIR="$runtime/matplotlib"
export TORCH_HOME="$runtime/torch"
export JUPYTER_RUNTIME_DIR="$runtime/jupyter"
cd "$work/source"
case "$phase" in
  5b)
    : "${SLURM_ARRAY_TASK_ID:?5B requires an array index}"
    python scripts/adaptive/rep5bc.py run-b --work-root "$work" --task-index "$SLURM_ARRAY_TASK_ID" --device cuda
    ;;
  5c)
    seed="${3:?seed}"; policy="${4:?policy}"; cycle="${5:?cycle}"
    python scripts/adaptive/rep5bc.py run-c --work-root "$work" --seed "$seed" --policy "$policy" --cycle "$cycle"
    ;;
  *) echo 'Unknown representation experiment' >&2; exit 2;;
esac
