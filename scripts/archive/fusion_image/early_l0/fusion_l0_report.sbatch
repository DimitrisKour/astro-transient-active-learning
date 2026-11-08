#!/bin/bash -l
#SBATCH --job-name=fusion_l0_report
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
set -euo pipefail
: "${FUSION_L0_ROOT:?Missing FUSION_L0_ROOT}"
: "${FUSION_L0_REPO:?Missing FUSION_L0_REPO}"
cd "$FUSION_L0_REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -u scripts/archive/fusion_image/early_l0/fusion_l0_matrix.py report --run-root "$FUSION_L0_ROOT"
