#!/bin/bash -l
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:24:00
set -euo pipefail
source_root="$1"; cycle="$2"
source "$source_root/scripts/adaptive/task6a_common.sh"
task6a_activate
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
index="${SLURM_ARRAY_TASK_ID:?Must run as a SLURM array job}"
[[ "$index" =~ ^[0-5]$ ]] || { echo "Invalid Task6A array index: $index" >&2; exit 2; }
seeds=(135 246 357)
policies=(entropy entropy_balanced_explore)
seed="${seeds[$((index/2))]}"; policy="${policies[$((index%2))]}"
runtime="$TASK6A_DATA/runs/$TASK6A_SUITE_ID/shared/seed_$seed/runtime/${SLURM_JOB_ID:-manual}_${index}"
mkdir -p "$runtime"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$runtime/tmp" MPLCONFIGDIR="$runtime/matplotlib" TORCH_HOME="$runtime/torch" JUPYTER_RUNTIME_DIR="$runtime/jupyter"
python "$source_root/scripts/adaptive/task6a_prepare.py" --verify-only \
  --job-dir "$TASK6A_JOB_DIR"
task6a_arguments "$seed" "$policy"
echo "TASK6A weight exponent=0.5; loss-best; per-sample; seed=$seed policy=$policy cycle=$cycle"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$source_root/scripts/run_task6a_closed_loop.py" "${TASK6A_ARGS[@]}" --device cuda --cycle "$cycle" --run
