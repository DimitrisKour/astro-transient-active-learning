#!/bin/bash
# Optional CPU-only full Task5A queue. Never starts GPU jobs.
# Usage: bash scripts/adaptive/task5a_submit_cpu.sh --run-dir PATH --dry-run|--submit
# Defaults to dry-run and 4 concurrently running CPU array tasks.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir=""; mode="--dry-run"; parallel=4; walltime=00:30:00
while (( $# )); do
  case "$1" in
    --run-dir) run_dir="${2:?}"; shift 2;;
    --dry-run|--submit) mode="$1"; shift;;
    --parallel) parallel="${2:?}"; shift 2;;
    --walltime) walltime="${2:?}"; shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$run_dir" && -f "$run_dir/plan.json" ]] || { echo 'Supply a prepared --run-dir with plan.json' >&2; exit 2; }
[[ "$parallel" =~ ^[1-9][0-9]*$ ]] || { echo 'parallel must be positive integer' >&2; exit 2; }
[[ "$walltime" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || { echo 'walltime must be HH:MM:SS' >&2; exit 2; }
read -r experiment shards < <(python - "$run_dir/plan.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p['mode'],p['shards'])
PY
)
[[ "$experiment" == full && "$shards" =~ ^[1-9][0-9]*$ ]] || { echo 'Only a prepared full plan can be queued' >&2; exit 2; }
total=$((3*shards))
last=$((total-1))
manifest="$run_dir/submitted_cpu.tsv"
[[ ! -e "$manifest" ]] || { echo "Existing submission $manifest; do not queue twice" >&2; exit 1; }
logdir="$run_dir/logs"
worker="$repo/scripts/adaptive/task5a_cpu.sbatch"
[[ -f "$worker" ]] || { echo "Missing worker: $worker" >&2; exit 1; }
if [[ "$mode" == --dry-run ]]; then
  echo "Plan: $run_dir; $total independent CPU extraction shards, max $parallel concurrent"
  echo "sbatch --hold --parsable --array=0-$last%$parallel --time=$walltime --output=$logdir/extract_%A_%a.out $worker array $run_dir"
  echo 'One final CPU report job depends on all array elements succeeding.'
  echo 'DRY RUN: no jobs submitted.'
  exit 0
fi
# Preflight before making a single scheduler call. No conda needed for provenance verification.
PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
  python "$repo/scripts/adaptive/task5a.py" --run-dir "$run_dir" verify
mkdir -p "$logdir"
array_id=""; report_id=""; committed=0
cleanup() {
  if (( ! committed )); then
    [[ -z "$report_id" ]] || scancel "$report_id" || true
    [[ -z "$array_id" ]] || scancel "$array_id" || true
    rm -f -- "$manifest"
  fi
}
trap cleanup EXIT
array_id="$(sbatch --hold --parsable --array="0-$last%$parallel" --time="$walltime" \
  --job-name=T5A_extract --output="$logdir/extract_%A_%a.out" \
  "$worker" array "$run_dir")"
[[ "$array_id" =~ ^[0-9]+$ ]] || { echo "Invalid array job ID: $array_id" >&2; exit 1; }
report_id="$(sbatch --hold --parsable --dependency="afterok:$array_id" --time=00:20:00 \
  --job-name=T5A_report --output="$logdir/report_%j.out" \
  "$worker" report "$run_dir")"
[[ "$report_id" =~ ^[0-9]+$ ]] || { echo "Invalid report job ID: $report_id" >&2; exit 1; }
# Both jobs are held until dependencies AND submission manifest are established.
printf 'role\tjob_id\narray\t%s\nreport\t%s\n' "$array_id" "$report_id" > "$manifest"
scontrol release "$report_id"
scontrol release "$array_id"
committed=1
trap - EXIT
printf 'Submitted CPU-only Task 5A: array=%s (%s shards), report=%s\n' "$array_id" "$total" "$report_id"
