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
source_root="$1"; seed="$2"; cycle="$3"; policy="$4"
source "$source_root/scripts/adaptive/task6b_common.sh"
task6b_activate
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$cycle" == 0 ]]; then
  [[ "$policy" == bootstrap ]] || { echo "Cycle zero must be bootstrap" >&2; exit 2; }
else
  [[ "$cycle" =~ ^[1-9]$ ]] || { echo "Invalid cycle $cycle" >&2; exit 2; }
  case "$policy" in entropy|entropy_balanced_explore) ;; *) echo "Invalid policy $policy" >&2; exit 2;; esac
fi
case "$seed" in 135|246|357) ;; *) echo "Invalid seed $seed" >&2; exit 2;; esac
runtime="$TASK6B_DATA/runs/$TASK6B_SUITE_ID/shared/seed_$seed/runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$runtime"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$runtime/tmp" MPLCONFIGDIR="$runtime/matplotlib" TORCH_HOME="$runtime/torch" JUPYTER_RUNTIME_DIR="$runtime/jupyter"
python "$source_root/scripts/adaptive/task6b_prepare.py" --verify-only \
  --job-dir "$TASK6B_JOB_DIR"
if [[ "$cycle" == 0 ]]; then
  task6b_arguments "$seed" entropy
else
  task6b_arguments "$seed" "$policy"
fi
echo "TASK6B new L0, weight exponent=.85; selected=object-macro-F1-best; early-stop=loss-best; per-sample; seed=$seed policy=$policy cycle=$cycle"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "$cycle" == 0 ]]; then
  python "$source_root/scripts/run_task6b_closed_loop.py" "${TASK6B_ARGS[@]}" --device cuda --bootstrap-only --run
else
  python "$source_root/scripts/run_task6b_closed_loop.py" "${TASK6B_ARGS[@]}" --device cuda --require-existing-bootstrap --cycle "$cycle" --run
fi
