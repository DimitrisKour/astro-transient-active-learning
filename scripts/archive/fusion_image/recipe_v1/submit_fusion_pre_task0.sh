#!/usr/bin/env bash
# Stage, submit or inspect PRE-Task-0 training-recipe comparisons.
# Cache completion is always verified before any sbatch call.
set -euo pipefail
MODE="${1:---prepare}"
[[ "$MODE" == --prepare || "$MODE" == --submit || "$MODE" == --submit-optional || "$MODE" == --status || "$MODE" == --report ]] || {
  echo 'Use --prepare | --submit | --submit-optional | --status | --report' >&2; exit 2; }
REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_v1
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$MODE" == --report ]]; then
  python scripts/archive/fusion_image/recipe_v1/fusion_pre_task0.py report
  exit 0
fi
if [[ "$MODE" == --status ]]; then
  [[ -f "$ROOT/plan.json" ]] || { echo 'Not prepared yet'; exit 0; }
  [[ -f "$ROOT/submitted_jobs.tsv" ]] && cat "$ROOT/submitted_jobs.tsv"
  squeue -u "$USER" || true
  python scripts/archive/fusion_image/recipe_v1/fusion_pre_task0.py report
  exit 0
fi
python scripts/archive/fusion_image/recipe_v1/fusion_pre_task0.py prepare
if [[ "$MODE" == --prepare ]]; then
  echo 'Prepared, no GPU jobs submitted.'; exit 0
fi
mkdir -p "$ROOT/logs"
if [[ "$MODE" == --submit ]]; then
  CASES=(constant cosine cosine_aug)
else
  # Advanced experiments are consciously opt-in after reviewing the primary trio.
  for prior in constant cosine cosine_aug; do
    [[ -f "$ROOT/$prior/training_summary.json" ]] || {
      echo "Primary comparison $prior not complete. Inspect first." >&2; exit 1; }
  done
  CASES=(cosine_aug_ema cosine_aug_effnum)
fi
# Fail rather than overwrite checkpoints or accidentally submit duplicates.
for case in "${CASES[@]}"; do
  [[ ! -e "$ROOT/$case" ]] || { echo "Existing run directory: $ROOT/$case" >&2; exit 1; }
  if [[ -f "$ROOT/submitted_jobs.tsv" ]] && grep -Eq "^${case}[[:space:]]" "$ROOT/submitted_jobs.tsv"; then
    echo "Already submitted $case. Inspect results before requesting a retry." >&2; exit 1
  fi
done
# One running GPU fit at a time, with afterok dependencies; stop the downstream chain
# if an earlier fit fails, rather than burning GPU time on a broken setup.
PREV=""
for case in "${CASES[@]}"; do
  deps=()
  if [[ -n "$PREV" ]]; then deps+=(--dependency="afterok:$PREV"); fi
  JID=$(sbatch --parsable "${deps[@]}" \
    --export="ALL,PRETASK_CASE=$case" \
    --output="$ROOT/logs/${case}_%j.out" \
    --error="$ROOT/logs/${case}_%j.err" \
    scripts/archive/fusion_image/recipe_v1/fusion_pre_task0.sbatch)
  printf '%s\t%s\n' "$case" "$JID" >> "$ROOT/submitted_jobs.tsv"
  PREV="$JID"
  echo "submitted case=$case job=$JID"
done
