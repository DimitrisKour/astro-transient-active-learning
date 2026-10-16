#!/bin/bash -l
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# Task 5A CPU-only extraction. Submit explicitly; never launch Task4/GPU work.
# Examples:
#   sbatch scripts/adaptive/task5a_cpu.sbatch smoke /path/to/run-dir
#   sbatch scripts/adaptive/task5a_cpu.sbatch extract /path/to/run-dir validation 0
#   sbatch scripts/adaptive/task5a_cpu.sbatch report /path/to/run-dir
set -euo pipefail
phase="${1:?smoke|extract|report}"; run_dir="${2:?run-directory}"
repo="/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning"
cd "$repo"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Keep scratch files outside Task 4 and write all real results under --run-dir.
runtime="/work/hdd/bcrv/dkourtesis/applecider_adaptive/task5a_runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$runtime"/{tmp,matplotlib,torch}
export TMPDIR="$runtime/tmp"
export MPLCONFIGDIR="$runtime/matplotlib"
export TORCH_HOME="$runtime/torch"
python scripts/adaptive/task5a.py --run-dir "$run_dir" verify
case "$phase" in
  smoke)
    [[ "$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["mode"])' "$run_dir/plan.json")" == smoke ]] || {
      echo 'Smoke command refuses a full-data plan.' >&2; exit 2;
    }
    for split in train validation test; do
      python scripts/adaptive/task5a.py --run-dir "$run_dir" extract \
        --split "$split" --shard 0 --batch-size 4 --threads "${SLURM_CPUS_PER_TASK:-4}"
    done
    python scripts/adaptive/task5a.py --run-dir "$run_dir" report
    ;;
  extract)
    split="${3:?split}"; shard="${4:?shard-index}"
    python scripts/adaptive/task5a.py --run-dir "$run_dir" extract \
      --split "$split" --shard "$shard" --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"
    ;;
  array)
    # For 3 splits x N deterministic shards: one SLURM array ID per shard.
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'No SLURM_ARRAY_TASK_ID' >&2; exit 2; }
    read -r mode nshards < <(python -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p["mode"],p["shards"])' "$run_dir/plan.json")
    [[ "$mode" == full ]] || { echo 'Array only accepts full-data plan' >&2; exit 2; }
    id="$SLURM_ARRAY_TASK_ID"
    [[ "$id" =~ ^[0-9]+$ ]] && (( id < 3 * nshards )) || { echo 'Out-of-range array ID' >&2; exit 2; }
    splits=(train validation test)
    split="${splits[$((id / nshards))]}"
    shard="$((id % nshards))"
    python scripts/adaptive/task5a.py --run-dir "$run_dir" extract \
      --split "$split" --shard "$shard" --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"
    ;;
  report)
    python scripts/adaptive/task5a.py --run-dir "$run_dir" report
    ;;
  *) echo "Unknown phase $phase" >&2; exit 2;;
esac
