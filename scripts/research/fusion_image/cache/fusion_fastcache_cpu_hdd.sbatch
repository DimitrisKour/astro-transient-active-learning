#!/bin/bash -l
#SBATCH --job-name=fusion_cache_hdd
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail
ulimit -c 0 || true
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
SOURCE=/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_train_cache_v1
TARGET=/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_hdd_v1
printf 'HOST=%s JOB_ID=%s UTC=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)"
printf 'SOURCE=%s\nTARGET=%s\n' "$SOURCE" "$TARGET"
if [[ ! -f "$SOURCE/cache_manifest.json" ]]; then
  echo 'ERROR: source historical cache manifest missing' >&2
  exit 1
fi
if [[ -e "$TARGET" ]]; then
  echo "ERROR: consolidated target already exists: $TARGET" >&2
  exit 1
fi
if [[ ! -f scripts/research/fusion_image/cache/fusion_fastcache.py ]]; then
  echo 'ERROR: install fusion_fastcache_buffered_v2 first' >&2
  exit 1
fi
echo '=== HDD BUFFERED BUILD ==='
python -u scripts/research/fusion_image/cache/fusion_fastcache.py build --source "$SOURCE" --target "$TARGET"
echo '=== VALIDATION ==='
python -u scripts/research/fusion_image/cache/fusion_fastcache.py check --source "$SOURCE" --target "$TARGET"
echo '=== HDD IO BENCHMARK ==='
python -u scripts/research/fusion_image/cache/fusion_fastcache.py bench --source "$SOURCE" --target "$TARGET" --scan-all
echo '=== DONE ==='
