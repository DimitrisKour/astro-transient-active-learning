#!/bin/bash -l
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
set -euo pipefail
JOB="$1"; MODE="$2"; SEED="${3:-135}"; CYCLE="${4:-}"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$JOB/source/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
RUNTIME="/work/hdd/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_task12_bald_mc20_v1/shared/runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$RUNTIME"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$RUNTIME/tmp"
export MPLCONFIGDIR="$RUNTIME/matplotlib"
export TORCH_HOME="$RUNTIME/torch"
export JUPYTER_RUNTIME_DIR="$RUNTIME/jupyter"
python "$JOB/source/scripts/research/historical_replay/task12/task12.py" verify --job-dir "$JOB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "$MODE" == 'pilot' ]]; then
  python "$JOB/source/scripts/research/historical_replay/task12/task12.py" pilot --seed "$SEED"
elif [[ "$MODE" == 'job' ]]; then
  python "$JOB/source/scripts/research/historical_replay/task12/task12.py" job "$SEED" "$CYCLE" --job-dir "$JOB"
else
  echo "Unknown Task12 mode $MODE" >&2; exit 2
fi
