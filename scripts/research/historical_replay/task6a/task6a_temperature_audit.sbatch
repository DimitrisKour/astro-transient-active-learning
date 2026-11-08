#!/bin/bash -l
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=T6A_temp_audit
set -euo pipefail
script="$1"; out="$2"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$script" run --output "$out"
