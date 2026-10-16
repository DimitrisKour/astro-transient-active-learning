#!/bin/bash -l
#SBATCH --job-name=fusion_l0_gpu
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail
: "${FUSION_L0_ROOT:?Missing FUSION_L0_ROOT}"
: "${FUSION_L0_REPO:?Missing FUSION_L0_REPO}"
: "${FUSION_L0_KIND:?Missing FUSION_L0_KIND}"
cd "$FUSION_L0_REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$FUSION_L0_ROOT/runtime/${SLURM_JOB_ID}"/{tmp,torch,matplotlib}
export TMPDIR="$FUSION_L0_ROOT/runtime/${SLURM_JOB_ID}/tmp"
export TORCH_HOME="$FUSION_L0_ROOT/runtime/${SLURM_JOB_ID}/torch"
export MPLCONFIGDIR="$FUSION_L0_ROOT/runtime/${SLURM_JOB_ID}/matplotlib"
python -u - <<'PY'
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if not torch.cuda.is_available(): raise SystemExit('No CUDA GPU allocated')
print(f'GPU: {torch.cuda.get_device_name(0)}')
PY
if [[ "$FUSION_L0_KIND" == pilot ]]; then
    : "${FUSION_L0_INIT:?Missing FUSION_L0_INIT}"
    [[ "$FUSION_L0_INIT" == pretrained || "$FUSION_L0_INIT" == randomized ]] || exit 2
    python -u scripts/adaptive/fusion_l0_matrix.py run \
      --run-root "$FUSION_L0_ROOT" --initialization "$FUSION_L0_INIT" \
      --seed 135 --pilot --device cuda
elif [[ "$FUSION_L0_KIND" == full ]]; then
    : "${SLURM_ARRAY_TASK_ID:?Missing array index}"
    inits=(pretrained pretrained pretrained randomized randomized randomized)
    seeds=(135 246 357 135 246 357)
    index="$SLURM_ARRAY_TASK_ID"
    [[ "$index" =~ ^[0-5]$ ]] || { echo 'Unexpected array index' >&2; exit 2; }
    python -u scripts/adaptive/fusion_l0_matrix.py run \
      --run-root "$FUSION_L0_ROOT" --initialization "${inits[$index]}" \
      --seed "${seeds[$index]}" --device cuda
else
    echo "Unknown kind $FUSION_L0_KIND" >&2; exit 2
fi
