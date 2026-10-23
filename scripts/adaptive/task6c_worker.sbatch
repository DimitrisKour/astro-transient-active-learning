#!/bin/bash -l
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
source_root="$1"; selection="$2"; seed="$3"; cycle="$4"; policy="$5"
source "$source_root/scripts/adaptive/task6c_common.sh"
task6c_config "$selection"
task6c_activate
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ "$cycle" =~ ^[1-9]$ ]] || { echo "Invalid cycle $cycle" >&2; exit 2; }
case "$seed" in 135|246|357) ;; *) echo "Invalid seed $seed" >&2; exit 2;; esac
case "$policy" in entropy|entropy_balanced_explore) ;; *) echo "Invalid policy $policy" >&2; exit 2;; esac
runtime="$TASK6C_DATA/runs/$TASK6C_SUITE_ID/shared/seed_$seed/runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$runtime"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$runtime/tmp" MPLCONFIGDIR="$runtime/matplotlib" TORCH_HOME="$runtime/torch" JUPYTER_RUNTIME_DIR="$runtime/jupyter"
python "$source_root/scripts/adaptive/task6c_prepare.py" verify --selection "$selection" --job-dir "$TASK6C_JOB_DIR"
task6c_arguments "$selection" "$seed" "$policy"
echo "TASK6C selection=$selection; equal_object; class_weight=.85; seed=$seed policy=$policy cycle=$cycle"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$source_root/scripts/run_task6c_closed_loop.py" "${TASK6C_ARGS[@]}" --device cuda --cycle "$cycle" --run
