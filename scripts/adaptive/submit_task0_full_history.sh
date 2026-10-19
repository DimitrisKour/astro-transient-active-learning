#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---check}"
FIT="${2:-}"
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task0_full_historical_v1
REFERENCE=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_resident_v3
REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT=scripts/adaptive/task0_full_history.py

verify_probe() {
  python - "$REFERENCE/resident_probe/probe.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    raise SystemExit(f'Finished resident-RAM probe missing: {p}')
r=json.loads(p.read_text())
if r.get('data_access_mode')!='shared_host_resident_arrays_v3' or r.get('train_rows')!=29161 or r.get('validation_rows')!=14148:
    raise SystemExit('Resident cache probe identity mismatch')
times=[x['fetch_seconds'] for x in r['batches']['train'][1:]]
if not times or sorted(times)[len(times)//2] > 0.5:
    raise SystemExit('Resident cache random fetch too slow')
print('Resident historical data probe: validated')
PY
}

submit_one() {
  local name="$1"
  case "$name" in
    pretrained_s135|pretrained_s246|pretrained_s357|randomized_s135|randomized_s246|randomized_s357) ;;
    *) echo "Unknown Task 0 fit: $name" >&2; exit 2 ;;
  esac
  if [[ -e "$ROOT/$name" ]]; then
    echo "Fit directory $name already exists (possibly partial); refusing overwrite" >&2
    return 1
  fi
  if [[ -f "$ROOT/submitted_jobs.tsv" ]] && grep -Eq "^${name}[[:space:]]" "$ROOT/submitted_jobs.tsv"; then
    echo "Fit $name already submitted; refusing duplicate" >&2
    return 1
  fi
  mkdir -p "$ROOT/logs"
  local job
  job=$(sbatch --parsable --time=00:15:00 \
    --export="ALL,TASK0_CASE=$name" \
    --output="$ROOT/logs/${name}_%j.out" \
    --error="$ROOT/logs/${name}_%j.err" \
    scripts/adaptive/task0_full_history.sbatch)
  printf '%s\t%s\n' "$name" "$job" >> "$ROOT/submitted_jobs.tsv"
  echo "Queued Task 0: $name job=$job limit=15m"
}

case "$MODE" in
  --check)
    python "$SCRIPT" check
    verify_probe
    ;;
  --prepare)
    verify_probe
    python "$SCRIPT" prepare
    echo 'Task 0 frozen plan prepared; no GPU jobs submitted.'
    ;;
  --submit-case)
    [[ -n "$FIT" ]] || { echo 'Supply a case name' >&2; exit 2; }
    verify_probe
    python "$SCRIPT" prepare
    submit_one "$FIT"
    ;;
  --submit-all)
    verify_probe
    python "$SCRIPT" prepare
    # Check ALL six first to prevent partial submission due to an already-queued case.
    for name in pretrained_s135 pretrained_s246 pretrained_s357 randomized_s135 randomized_s246 randomized_s357; do
      if [[ -e "$ROOT/$name" ]] || { [[ -f "$ROOT/submitted_jobs.tsv" ]] && grep -Eq "^${name}[[:space:]]" "$ROOT/submitted_jobs.tsv"; }; then
        echo "Case $name already exists/submitted. Use --submit-case for remaining runs." >&2
        exit 1
      fi
    done
    for name in pretrained_s135 pretrained_s246 pretrained_s357 randomized_s135 randomized_s246 randomized_s357; do
      submit_one "$name"
    done
    ;;
  --status)
    [[ ! -f "$ROOT/submitted_jobs.tsv" ]] || cat "$ROOT/submitted_jobs.tsv"
    squeue -u "$USER" || true
    ;;
  --report)
    python "$SCRIPT" report
    ;;
  *)
    echo 'Usage: --check | --prepare | --submit-case NAME | --submit-all | --status | --report' >&2
    exit 2
    ;;
esac
