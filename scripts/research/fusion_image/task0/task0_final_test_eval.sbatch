#!/bin/bash -l
#SBATCH --job-name=task0_final_test
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
: "${TASK0_CASE:?TASK0_CASE is required}"
REPO="${AL_PIPELINE_REPO:-${SLURM_SUBMIT_DIR:-/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning}}"
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u scripts/research/fusion_image/task0/task0_vit_test_compare.py evaluate-new \
  --case "$TASK0_CASE" --device cuda --confirm-frozen-final-test
