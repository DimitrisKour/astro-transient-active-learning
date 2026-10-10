#!/bin/bash -l
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
repo="$1"
seed="$2"
cycle="$3"
export POLICY_SUITE_ID="$4"
source "$repo/scripts/adaptive/policy_common.sh"
policy_activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
runtime="$POLICY_DATA_DIR/runs/$POLICY_SUITE_ID/shared/seed_$seed/runtime/$SLURM_JOB_ID"
mkdir -p "$runtime"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$runtime/tmp"
export MPLCONFIGDIR="$runtime/matplotlib"
export TORCH_HOME="$runtime/torch"
export JUPYTER_RUNTIME_DIR="$runtime/jupyter"
python scripts/adaptive/prepare_policy_compare.py \
  --repo "$POLICY_REPO" --job-dir "$POLICY_JOB_DIR" --verify-source
policy_arguments "$seed"
echo "Seed $seed, round $cycle, suite $POLICY_SUITE_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/run_astrominn_historical_replay.py "${POLICY_ARGS[@]}" \
  --device cuda --cycle "$cycle" --run
