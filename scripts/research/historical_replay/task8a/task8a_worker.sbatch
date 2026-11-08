#!/bin/bash -l
#SBATCH --job-name=T8A_checkpoint_audit
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail

# Task 8A requires *no* GPU. Delta allocation only grants the GPU partition.
# Explicit stdout/stderr paths should be passed to sbatch pointing to /work/hdd.
SOURCE_ROOT="${1:?Pass repository root}"
OUTPUT_DIR="${2:?Pass HDD-backed audit output directory}"
if [[ ! -f "$SOURCE_ROOT/scripts/research/historical_replay/task8a/task8a.py" ]]; then
    echo 'Missing Task 8A runner in repository' >&2
    exit 2
fi
case "$OUTPUT_DIR" in
    /work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/*) ;;
    *) echo 'Task 8A output must be in the designated HDD analysis directory' >&2; exit 2 ;;
esac
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="$SOURCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
python "$SOURCE_ROOT/scripts/research/historical_replay/task8a/task8a.py" run \
    --persistent /projects/bcrv/dkourtesis/applecider_adaptive \
    --data /work/hdd/bcrv/dkourtesis/applecider_adaptive \
    --output "$OUTPUT_DIR" --bootstrap 600
