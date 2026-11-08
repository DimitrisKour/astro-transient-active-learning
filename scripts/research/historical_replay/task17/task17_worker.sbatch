#!/bin/bash -l
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:35:00
set -euo pipefail
JOB="$1"; MODE="$2"
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
DATA=/work/hdd/bcrv/dkourtesis/applecider_adaptive
RUNTIME="$DATA/runs/astrominn_task17_random_l0_v1/shared/runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$RUNTIME"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$RUNTIME/tmp"
export MPLCONFIGDIR="$RUNTIME/matplotlib"
export TORCH_HOME="$RUNTIME/torch"
export JUPYTER_RUNTIME_DIR="$RUNTIME/jupyter"
python "$JOB/source/scripts/research/historical_replay/task17/task17.py" verify --job-dir "$JOB"
python "$JOB/source/scripts/research/historical_replay/task17/task17.py" filesystem-probe
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$MODE" in
  bootstrap) python "$JOB/source/scripts/research/historical_replay/task17/task17.py" job-bootstrap "$3" --job-dir "$JOB" ;;
  pilot) python "$JOB/source/scripts/research/historical_replay/task17/task17.py" pilot --seed "$3" --job-dir "$JOB" ;;
  cycle) python "$JOB/source/scripts/research/historical_replay/task17/task17.py" job-cycle "$3" "$4" "$5" --job-dir "$JOB" ;;
  *) echo "Unknown Task17 worker mode: $MODE" >&2; exit 2 ;;
esac
