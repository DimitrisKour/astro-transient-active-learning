#!/bin/bash -l
#SBATCH --job-name=T8C2_stage
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
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
JOB="$1"; INDEX="$2"; CYCLE="$3"; STAGE="$4"; NUMBER="$5"
export PYTHONPATH="$JOB/source/src"
if [[ "$STAGE" == base ]]; then
  python "$JOB/source/scripts/research/historical_replay/task8c2/task8c2.py" run --job-dir "$JOB" --index "$INDEX" --cycle "$CYCLE" --stage base
else
  python "$JOB/source/scripts/research/historical_replay/task8c2/task8c2.py" run --job-dir "$JOB" --index "$INDEX" --cycle "$CYCLE" --stage retry --retry-number "$NUMBER"
fi
