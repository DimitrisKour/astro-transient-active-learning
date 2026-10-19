#!/bin/bash -l
#SBATCH --job-name=fusion_fastcache
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/adaptive/fusion_fastcache.py build
python -u scripts/adaptive/fusion_fastcache.py check
python -u scripts/adaptive/fusion_fastcache.py bench --scan-all
