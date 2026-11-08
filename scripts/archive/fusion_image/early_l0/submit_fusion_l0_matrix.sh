#!/usr/bin/env bash
# --prepare: immutable input and L0 checks only. --submit: 2 pilots -> 6 L0 fits -> report.
set -euo pipefail
MODE="${1:---prepare}"
[[ "$MODE" == --prepare || "$MODE" == --submit ]] || { echo 'Use --prepare or --submit' >&2; exit 2; }
REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_matrix_v1
CHECKS=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_checks
REG=/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop/astrominn_closed_loop_warm_v1/registry.toml
AUDIT="$CHECKS/data_audit_full_optimized_v1/job_22899411/audit.json"
INDEX=/work/hdd/bcrv/ssharmachaudhary/data_ztf_preprocessed/built_all.csv
UPSTREAM=/projects/bcrv/dkourtesis/fusion_model_basics
HF=/work/hdd/bcrv/ssharmachaudhary/model_cache/BTSbot-convnext-pico-galaxyzoo
HF_SHA=8fc996eee3821d05ec42f4b462bf8140f1a70b15e9417eb61361479653b7300c
L0=/projects/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_historical_replay_v1/shared/historical_replay_split/initial_labeled_manifest.csv
SMOKE_TRAIN="$CHECKS/data_audit/smoke_train.csv"
SMOKE_VAL="$CHECKS/data_audit/smoke_validation.csv"
cd "$REPO"
for p in "$REG" "$AUDIT" "$INDEX" "$HF/pytorch_model.bin" "$L0" "$SMOKE_TRAIN" "$SMOKE_VAL"; do
    [[ -r "$p" ]] || { echo "REQUIRED INPUT NOT READABLE: $p" >&2; exit 1; }
done
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/archive/fusion_image/early_l0/fusion_l0_matrix.py prepare \
    --repo "$REPO" --run-root "$ROOT" --registry "$REG" \
    --audit "$AUDIT" --new-index "$INDEX" --upstream-repo "$UPSTREAM" \
    --hf-snapshot "$HF" --hf-weights-sha256 "$HF_SHA" \
    --l0 "$L0" --pilot-train "$SMOKE_TRAIN" --pilot-validation "$SMOKE_VAL" \
    --epochs 30 --patience 6 --batch-size 64 --lr 0.0001 --class-weight-exponent 0.85
if [[ "$MODE" == --prepare ]]; then
    echo 'Prepared. Zero SLURM jobs submitted.'
    exit 0
fi
[[ ! -e "$ROOT/submitted_jobs.tsv" ]] || { echo 'Submission manifest already exists. Refusing duplicate queue.' >&2; exit 1; }
if [[ -d "$ROOT/fits" || -d "$ROOT/pilots" ]]; then
    echo 'Existing pilots or fits found: do not resubmit without inspecting progress' >&2
    exit 1
fi
mkdir -p "$ROOT/logs"
export FUSION_L0_ROOT="$ROOT" FUSION_L0_REPO="$REPO"
pre_id=$(sbatch --parsable \
    --export=ALL,FUSION_L0_KIND=pilot,FUSION_L0_INIT=pretrained \
    --job-name=fusion_pilot_pre --time=00:45:00 \
    --output="$ROOT/logs/pilot_pre_%j.out" --error="$ROOT/logs/pilot_pre_%j.err" \
    scripts/archive/fusion_image/early_l0/fusion_l0_gpu.sbatch)
ran_id=$(sbatch --parsable \
    --export=ALL,FUSION_L0_KIND=pilot,FUSION_L0_INIT=randomized \
    --job-name=fusion_pilot_ran --time=00:45:00 \
    --output="$ROOT/logs/pilot_ran_%j.out" --error="$ROOT/logs/pilot_ran_%j.err" \
    scripts/archive/fusion_image/early_l0/fusion_l0_gpu.sbatch)
arr_id=$(sbatch --parsable \
    --dependency="afterok:$pre_id:$ran_id" --array=0-5%2 \
    --export=ALL,FUSION_L0_KIND=full \
    --job-name=fusion_l0_matrix --time=03:00:00 \
    --output="$ROOT/logs/full_%A_%a.out" --error="$ROOT/logs/full_%A_%a.err" \
    scripts/archive/fusion_image/early_l0/fusion_l0_gpu.sbatch)
report_id=$(sbatch --parsable \
    --dependency="afterok:$arr_id" \
    --export=ALL \
    --output="$ROOT/logs/report_%j.out" --error="$ROOT/logs/report_%j.err" \
    scripts/archive/fusion_image/early_l0/fusion_l0_report.sbatch)
{
  printf 'job\tjob_id\n'
  printf 'pilot_pretrained\t%s\n' "$pre_id"
  printf 'pilot_randomized\t%s\n' "$ran_id"
  printf 'full_matrix\t%s\n' "$arr_id"
  printf 'report\t%s\n' "$report_id"
} > "$ROOT/submitted_jobs.tsv"
echo "Queued pilots $pre_id/$ran_id, 6-fit matrix $arr_id, report $report_id"
echo "Monitor: squeue -u \"\$USER\"; cat $ROOT/submitted_jobs.tsv"
