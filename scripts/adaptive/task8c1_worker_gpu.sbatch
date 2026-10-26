#!/bin/bash -l
#SBATCH --job-name=T8C1_train
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
JOB="$1"; CASE_INDEX="$2"; CYCLE="$3"
export PYTHONPATH="$JOB/source/src"
python "$JOB/source/scripts/adaptive/task8c1.py" run --job-dir "$JOB" --index "$CASE_INDEX" --cycle "$CYCLE"
