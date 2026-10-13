#!/bin/bash -l
# Usage: bash scripts/adaptive/task4_submit.sh --prepare-only|--submit [--cycles 9]
#          [--early-walltime 00:15:00] [--late-walltime 00:20:00]
# One fit per GPU job, with 12 independent seed/policy dependency chains.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/task4_common.sh"
mode="${1:---prepare-only}"
[[ "$mode" == --prepare-only || "$mode" == --submit ]] || {
  echo 'Usage: task4_submit.sh --prepare-only|--submit [--cycles 9] [--early-walltime HH:MM:SS] [--late-walltime HH:MM:SS]' >&2; exit 2;
}
shift || true
cycles=9
early_walltime="$TASK4_EARLY_WALLTIME"
late_walltime="$TASK4_LATE_WALLTIME"
while (( $# )); do
  case "$1" in
    --cycles) cycles="${2:?}"; shift 2;;
    --early-walltime) early_walltime="${2:?}"; shift 2;;
    --late-walltime) late_walltime="${2:?}"; shift 2;;
    *) echo "Unrecognized argument $1" >&2; exit 2;;
  esac
done
# Strict validation before touching the experiment directories.
[[ "$cycles" =~ ^[1-9]$ ]] || { echo 'Use 1..9 cycles.' >&2; exit 2; }
task4_activate
export PYTHONPATH="$TASK4_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$TASK4_REPO"
python scripts/adaptive/task4_prepare.py --repo "$TASK4_REPO" \
  --job-dir "$TASK4_JOB_DIR" --source-registry "$TASK4_SOURCE_REGISTRY" \
  --persistent "$TASK4_PERSISTENT" --data "$TASK4_DATA"
# Plan is immutable between prepare and queue: an existing differing plan aborts.
plan="$TASK4_JOB_DIR/queue_plan.tsv"
tmpplan="$(mktemp "$TASK4_JOB_DIR/.queue_plan.XXXXXXXX")"
trap 'rm -f "$tmpplan"' EXIT
python "$TASK4_SOURCE/scripts/adaptive/task4_schedule.py" --cycles "$cycles" \
  --early-walltime "$early_walltime" --late-walltime "$late_walltime" > "$tmpplan"
if [[ -e "$plan" ]]; then
  cmp -s "$tmpplan" "$plan" || { echo "Prepared Task4 job plan changed: $plan" >&2; exit 1; }
else
  mv "$tmpplan" "$plan"
fi
# Verify split, all 3 seeds' original bootstraps, and configuration before GPU work.
for seed in 135 246 357; do
  task4_arguments "$seed"
  PYTHONPATH="$TASK4_SOURCE/src" python "$TASK4_SOURCE/scripts/run_astrominn_closed_loop.py" \
    "${TASK4_ARGS[@]}" --device cpu > "$TASK4_JOB_DIR/plan_seed_$seed.txt"
  echo "Seed $seed preflight passed; plan saved in $TASK4_JOB_DIR/plan_seed_$seed.txt"
done
python "$TASK4_SOURCE/scripts/adaptive/task4_preflight.py" \
  --persistent "$TASK4_PERSISTENT" --data "$TASK4_DATA" \
  --base-suite "$TASK4_BASE_SUITE_ID" --suite "$TASK4_SUITE_ID"
if [[ "$mode" == --prepare-only ]]; then
  echo "Preflight passed. $(wc -l < "$plan") jobs planned; nothing submitted."
  exit 0
fi
# Do not re-submit an existing Task4 chain, or a progressed/active trajectory.
manifest="$TASK4_JOB_DIR/submitted.tsv"
[[ ! -e "$manifest" ]] || { echo "Submission exists: $manifest; inspect before retrying." >&2; exit 1; }
python - "$TASK4_PERSISTENT" "$TASK4_SUITE_ID" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])/"runs"/sys.argv[2]
for policy in ("random","entropy","entropy_diversity","entropy_balanced_explore"):
 for seed in (135,246,357):
  state=root/policy/f"seed_{seed}"/"run_state.json"
  if state.exists():
   o=json.loads(state.read_text())
   if o.get("active_cycle") is not None or int(o.get("last_completed_cycle",0)) != 0:
    raise SystemExit(f"Task4 already progressed; do not resubmit blindly: {state}")
PY
# Submit from the immutable prepared plan using the frozen scheduler script.
bash "$TASK4_SOURCE/scripts/adaptive/task4_enqueue.sh" \
  "$plan" "$TASK4_SOURCE" "$TASK4_JOB_DIR" "$cycles"
