#!/bin/bash -l
#SBATCH --job-name=task1_al
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
set -euo pipefail
REPO="${AL_PIPELINE_REPO:-${SLURM_SUBMIT_DIR:-/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning}}"
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${TASK1_KIND:?}" == "l0" ]];then
  exec python -u scripts/research/fusion_image/task1/task1_closed_loop.py run-l0 \
    --root "${TASK1_ROOT:?}" --initialization "${TASK1_INIT:?}" --seed "${TASK1_SEED:?}"
elif [[ "$TASK1_KIND" == "cycle" ]];then
  exec python -u scripts/research/fusion_image/task1/task1_closed_loop.py run-cycle \
    --root "${TASK1_ROOT:?}" --configuration "${TASK1_CONFIG:?}" \
    --policy "${TASK1_POLICY:?}" --seed "${TASK1_SEED:?}" --cycle "${TASK1_CYCLE:?}"
else
  echo "Unknown Task-1 job kind: $TASK1_KIND" >&2;exit 2
fi
