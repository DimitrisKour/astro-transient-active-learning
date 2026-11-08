#!/bin/bash -l
#SBATCH --job-name=T8C1_prefill
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
JOB="$1"
export PYTHONPATH="$JOB/source/src"
python "$JOB/source/scripts/research/historical_replay/task8c1/task8c1.py" prefill --job-dir "$JOB" --index "$SLURM_ARRAY_TASK_ID"
