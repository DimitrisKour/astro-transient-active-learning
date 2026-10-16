#!/usr/bin/env bash
#SBATCH --job-name=fusion_full_stage
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00

set -euo pipefail

REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
UPSTREAM=/projects/bcrv/dkourtesis/fusion_model_basics
PYTHON=/u/dkourtesis/.conda/envs/al-pipeline/bin/python
REG=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_v1/registry.toml
NEW_INDEX=/work/hdd/bcrv/ssharmachaudhary/data_ztf_preprocessed/built_all.csv
BASE=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_checks/data_audit_full_optimized_v1
EXPECTED_SHA=9d8b635a35c9dc4f90271098abdfa494abffc24f

cd "$REPO"
for file in "$PYTHON" "$REG" "$NEW_INDEX" scripts/adaptive/audit_fusion_image.py; do
    if [[ ! -r "$file" ]]; then
        echo "Required input is missing or not readable: $file" >&2
        exit 2
    fi
done
CURRENT_SHA="$(git -C "$UPSTREAM" rev-parse HEAD)"
if [[ "$CURRENT_SHA" != "$EXPECTED_SHA" ]]; then
    echo "Wrong upstream SHA: $CURRENT_SHA; expected $EXPECTED_SHA" >&2
    exit 2
fi

export PYTHONPATH="/projects/bcrv/dkourtesis:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUT="$BASE/job_${SLURM_JOB_ID:?This script must run under SLURM}"
mkdir -p "$OUT"
echo "Host: $(hostname) | Job: $SLURM_JOB_ID | Output: $OUT"
echo "Upstream SHA: $CURRENT_SHA"

"$PYTHON" -u scripts/adaptive/audit_fusion_image.py \
    --registry "$REG" \
    --new-index "$NEW_INDEX" \
    --model-sha "$CURRENT_SHA" \
    --check-stages -1 \
    --strict-stages \
    --out-dir "$OUT"

echo "PASS: complete historical stage audit, $OUT/audit.json"
