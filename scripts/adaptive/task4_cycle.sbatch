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
source "$source_root/scripts/adaptive/task4_common.sh"
task4_activate
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
runtime="$TASK4_DATA/runs/$TASK4_SUITE_ID/shared/seed_$seed/runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$runtime"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$runtime/tmp"
export MPLCONFIGDIR="$runtime/matplotlib"
export TORCH_HOME="$runtime/torch"
export JUPYTER_RUNTIME_DIR="$runtime/jupyter"
python "$source_root/scripts/adaptive/task4_prepare.py" \
  --repo "$source_root" --job-dir "$TASK4_JOB_DIR" \
  --source-registry "$TASK4_REGISTRY" \
  --persistent "$TASK4_PERSISTENT" --data "$TASK4_DATA" --verify-only
# Scientific settings must be identical to the planned submission.
task4_arguments "$seed" "$policy"
echo "TASK4: suite=$TASK4_SUITE_ID seed=$seed cycle=$cycle policy=$policy; frozen=$source_root"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$source_root/scripts/run_astrominn_closed_loop.py" "${TASK4_ARGS[@]}" \
  --device cuda --cycle "$cycle" --run
