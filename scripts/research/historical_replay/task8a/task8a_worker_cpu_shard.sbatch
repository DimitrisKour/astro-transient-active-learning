#!/bin/bash -l
#SBATCH --job-name=T8A_cpu_shard
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
set -euo pipefail

SOURCE_ROOT="${1:?Pass repository root}"
OUTPUT_DIR="${2:?Pass HDD output shard directory}"
SUITE="${3:?Pass exactly one suite: task4, task6b, or task6c}"
case "$SUITE" in task4|task6b|task6c) ;; *) echo "Invalid shard suite: $SUITE" >&2; exit 2;; esac
case "$OUTPUT_DIR" in
    /work/hdd/bcrv/dkourtesis/applecider_adaptive/analysis/task8a_retrospective_acceptance_v1/shards/"$SUITE") ;;
    *) echo 'Unexpected HDD shard output directory' >&2; exit 2 ;;
esac
[[ -f "$SOURCE_ROOT/scripts/research/historical_replay/task8a/task8a.py" ]] || { echo 'Task8A script missing' >&2; exit 2; }
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="$SOURCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
python "$SOURCE_ROOT/scripts/research/historical_replay/task8a/task8a.py" run \
    --persistent /projects/bcrv/dkourtesis/applecider_adaptive \
    --data /work/hdd/bcrv/dkourtesis/applecider_adaptive \
    --output "$OUTPUT_DIR" \
    --suites "$SUITE" \
    --bootstrap 600
