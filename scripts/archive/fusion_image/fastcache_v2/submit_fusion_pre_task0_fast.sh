#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---prepare}"
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_fastcache_v2
REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT=scripts/archive/fusion_image/fastcache_v2/fusion_pre_task0_fast.py
case "$MODE" in
  --check) python "$SCRIPT" check ;;
  --prepare) python "$SCRIPT" prepare; echo 'Prepared, no GPU jobs submitted.' ;;
  --status)
    test -f "$ROOT/plan.json" || { echo 'Not prepared'; exit 0; }
    test ! -f "$ROOT/submitted_jobs.tsv" || cat "$ROOT/submitted_jobs.tsv"
    squeue -u "$USER" || true
    ;;
  --report) python "$SCRIPT" report ;;
  --submit-case)
    FIT="${2:?Expected one recipe: constant, cosine, cosine_aug, cosine_aug_ema, cosine_aug_effnum}"
    case "$FIT" in constant|cosine|cosine_aug|cosine_aug_ema|cosine_aug_effnum) ;; *) echo 'Unknown fit' >&2; exit 2;; esac
    python "$SCRIPT" prepare
    if [[ -e "$ROOT/$FIT" ]]; then
      echo "Existing run dir (possibly partial): $ROOT/$FIT" >&2; exit 1
    fi
    if [[ -f "$ROOT/submitted_jobs.tsv" ]] && grep -Eq "^${FIT}[[:space:]]" "$ROOT/submitted_jobs.tsv"; then
      echo "Already submitted $FIT; do not submit again" >&2; exit 1
    fi
    # Guard: default main comparison is sequential/manual, so inspect first GPU fit.
    if [[ "$FIT" == cosine || "$FIT" == cosine_aug ]]; then
      test -f "$ROOT/constant/training_summary.json" || { echo 'Finish constant and inspect timing before further recipe submissions' >&2; exit 1; }
    fi
    if [[ "$FIT" == cosine_aug_ema || "$FIT" == cosine_aug_effnum ]]; then
      for prev in constant cosine cosine_aug; do
        test -f "$ROOT/$prev/training_summary.json" || { echo "Primary fit $prev not complete" >&2; exit 1; }
      done
    fi
    mkdir -p "$ROOT/logs"
    JID=$(sbatch --parsable --export="ALL,PRETASK_CASE=$FIT" \
      --output="$ROOT/logs/${FIT}_%j.out" --error="$ROOT/logs/${FIT}_%j.err" \
      scripts/archive/fusion_image/fastcache_v2/fusion_pre_task0_fast.sbatch)
    printf '%s\t%s\n' "$FIT" "$JID" >> "$ROOT/submitted_jobs.tsv"
    echo "Submitted $FIT job=$JID"
    ;;
  *) echo 'Use --check | --prepare | --submit-case constant | --status | --report' >&2; exit 2;;
esac
