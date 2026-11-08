#!/usr/bin/env bash
# Deliberately separate from the 15-minute training launcher: final TEST only.
set -euo pipefail
MODE="${1:---status}"
FIT="${2:-}"
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task0_full_historical_grouped_v1
TEST_CACHE=/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_historical_test_fastcache_grouped_v1
REPO="${AL_PIPELINE_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

submit_one() {
  case "$1" in
    pretrained_s135|pretrained_s246|pretrained_s357|randomized_s135|randomized_s246|randomized_s357) ;;
    *) echo "Unknown Task0 case: $1" >&2; exit 2;;
  esac
  [[ -f "$ROOT/final_test_split_audit.json" ]] || {
    echo 'No frozen old-ViT/new-model common-test split audit; refusing to schedule TEST.' >&2; exit 1;
  }
  # Check the audited test population and all six completed fits. No inference here.
  python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
from al_pipeline.astronomy.fusion_image.contracts import sha256
root=Path(sys.argv[1]); p=root/'plan.json'
plan=json.loads(p.read_text()); audit=json.loads((root/'final_test_split_audit.json').read_text())
if audit.get('test_evaluation_permitted') is not True or audit.get('task0_plan_sha256') != sha256(p):
    raise SystemExit('Frozen test audit does not match Task 0 plan')
for case in plan['recipes']:
    q=root/case/'training_summary.json'
    if not q.exists() or json.loads(q.read_text()).get('status')!='completed':
        raise SystemExit('All six Task 0 scientific fits must finish before test submissions')
PY
  # Fast cache is independent, label-free and must already be audited/published.
  python scripts/research/fusion_image/task0/task0_test_cache_v1.py validate
  if [[ -e "$ROOT/final_test_evaluation/$1" ]] || { [[ -f "$ROOT/test_jobs.tsv" ]] && grep -Eq "^${1}[[:space:]]" "$ROOT/test_jobs.tsv"; }; then
    echo "Test evaluation $1 already present/submitted; inspect before retry." >&2; exit 1
  fi
  mkdir -p "$ROOT/logs"
  id=$(sbatch --parsable --time=00:15:00 \
    --export="ALL,TASK0_CASE=$1" \
    --output="$ROOT/logs/final_test_${1}_%j.out" \
    --error="$ROOT/logs/final_test_${1}_%j.err" \
    scripts/research/fusion_image/task0/task0_final_test_eval.sbatch)
  printf '%s\t%s\n' "$1" "$id" >> "$ROOT/test_jobs.tsv"
  echo "Queued frozen FINAL TEST $1 job=$id (15m; immutable test cache in host RAM)"
}
case "$MODE" in
  --submit-case)
    [[ -n "$FIT" ]] || { echo 'Pass one case name' >&2; exit 2; }
    submit_one "$FIT"
    ;;
  --status)
    [[ ! -f "$ROOT/test_jobs.tsv" ]] || cat "$ROOT/test_jobs.tsv"
    squeue -u "$USER" || true
    ;;
  *) echo 'Usage (only AFTER audit + six fits): --submit-case NAME | --status' >&2; exit 2;;
esac
