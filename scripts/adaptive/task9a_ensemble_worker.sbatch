#!/bin/bash -l
#SBATCH --job-name=T9A_ensemble
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'Provide active-learning repository root' >&2; exit 2; }
REPO="$1"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$REPO/scripts/adaptive/task9a_ensemble.py" verify
python "$REPO/scripts/adaptive/task9a_ensemble.py" run
