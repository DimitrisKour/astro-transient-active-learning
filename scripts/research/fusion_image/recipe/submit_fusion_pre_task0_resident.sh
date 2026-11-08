#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---prepare}"
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_resident_v3
REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT=scripts/research/fusion_image/recipe/fusion_pre_task0_resident.py
case "$MODE" in
  --check) python "$SCRIPT" check ;;
  --prepare) python "$SCRIPT" prepare; echo 'Prepared, no GPU jobs submitted.' ;;
  --probe)
    python "$SCRIPT" prepare
    PROBE_ROOT="$ROOT/resident_probe"
    if [[ -e "$PROBE_ROOT/probe.json" ]]; then echo 'Completed probe already exists' >&2; exit 1; fi
    if [[ -f "$ROOT/probe_job_id.txt" ]]; then echo 'Probe previously submitted; inspect status' >&2; exit 1; fi
    mkdir -p "$ROOT/logs"
    JOB=$(sbatch --parsable --output="$ROOT/logs/probe_%j.out" \
      --error="$ROOT/logs/probe_%j.err" scripts/research/fusion_image/recipe/fusion_resident_probe.sbatch)
    printf '%s\n' "$JOB" > "$ROOT/probe_job_id.txt"
    echo "Submitted short resident-RAM GPU probe: $JOB"
    ;;
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
    python - "$ROOT/resident_probe/probe.json" <<'PYPROBE'
import json,sys
from pathlib import Path
f=Path(sys.argv[1])
if not f.is_file():
    raise SystemExit('Resident-RAM probe must complete before training')
r=json.loads(f.read_text())
if r.get('data_access_mode') != 'shared_host_resident_arrays_v3' or r.get('train_rows') != 29161 or r.get('validation_rows') != 14148:
    raise SystemExit('Resident probe identity/data extent mismatch')
times=[v['fetch_seconds'] for v in r['batches']['train'][1:]]
if not times or sorted(times)[len(times)//2] > 0.5:
    raise SystemExit('Probe random-access fetch remains slow; inspect results before training')
print('Resident-RAM probe validated',flush=True)
PYPROBE
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
      scripts/research/fusion_image/recipe/fusion_pre_task0_resident.sbatch)
    printf '%s\t%s\n' "$FIT" "$JID" >> "$ROOT/submitted_jobs.tsv"
    echo "Submitted $FIT job=$JID"
    ;;
  *) echo 'Use --check | --prepare | --probe | --submit-case constant | --status | --report' >&2; exit 2;;
esac
