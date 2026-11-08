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
source_root="$1"; seed="$2"; cycle="$3"
source "$source_root/scripts/research/historical_replay/task7a/task7a_common.sh"
task7a_activate
case "$seed" in 135|246|357) ;; *) echo 'Invalid Task7A seed' >&2; exit 2;; esac
[[ "$cycle" =~ ^[1-9]$ ]] || { echo 'Invalid acquisition cycle 1-9' >&2; exit 2; }
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
runtime="$TASK7A_DATA/runs/$TASK7A_SUITE/shared/seed_$seed/runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$runtime"/{tmp,matplotlib,torch,jupyter}
export TMPDIR="$runtime/tmp" MPLCONFIGDIR="$runtime/matplotlib" TORCH_HOME="$runtime/torch" JUPYTER_RUNTIME_DIR="$runtime/jupyter"
python "$source_root/scripts/research/historical_replay/task7a/task7a.py" verify --job-dir "$TASK7A_JOB_DIR" --source-only
# GPU science: pool logits stay raw on disk; only entropy scoring uses logits/T.
task7a_arguments "$seed"
echo "TASK7A temperature-scaled ENTROPY ONLY seed=$seed cycle=$cycle suite=$TASK7A_SUITE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$source_root/scripts/run_task7a_closed_loop.py" "${TASK7A_ARGS[@]}" --device cuda --cycle "$cycle" --run
