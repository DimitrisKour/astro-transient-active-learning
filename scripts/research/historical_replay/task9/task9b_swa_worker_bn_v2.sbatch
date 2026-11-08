#!/bin/bash -l
#SBATCH --job-name=T9B_SWA
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Expected experiment directory and case index' >&2; exit 2; }
JOB="$1"; INDEX="$2"; SOURCE="$JOB/source"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$SOURCE/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
RUNTIME="$JOB/runtime/${SLURM_JOB_ID}"
mkdir -p "$RUNTIME"/{tmp,torch,matplotlib}
export TMPDIR="$RUNTIME/tmp" TORCH_HOME="$RUNTIME/torch" MPLCONFIGDIR="$RUNTIME/matplotlib"
python "$SOURCE/scripts/research/historical_replay/task9/task9b_swa_bn_v2.py" run --job-dir "$JOB" --index "$INDEX"
