#!/bin/bash -l
#SBATCH --job-name=T10_entropy_seed20
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
[[ $# -eq 1 && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'Expected frozen job directory and array index' >&2; exit 2; }
JOB="$1"
SOURCE="$JOB/source"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$SOURCE/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
RT="$JOB/runtime/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RT"/{tmp,torch,matplotlib}
export TMPDIR="$RT/tmp" TORCH_HOME="$RT/torch" MPLCONFIGDIR="$RT/matplotlib"
echo "Task10 seed20 index=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID"
python "$SOURCE/scripts/adaptive/task10_entropy20.py" run --job-dir "$JOB" --index "$SLURM_ARRAY_TASK_ID"
