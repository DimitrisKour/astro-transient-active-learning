#!/bin/bash -l
# Usage: bash scripts/adaptive/submit_policy_compare.sh SEED [WALLTIME | --prepare-only]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/policy_common.sh"
seed="${1:-135}"
option="${2:-01:00:00}"
case "$seed" in 135|246|357) ;; *) echo 'Use seed 135, 246 or 357.' >&2; exit 1;; esac
policy_activate
quota
python scripts/adaptive/prepare_policy_compare.py --repo "$POLICY_REPO" --job-dir "$POLICY_JOB_DIR"
policy_arguments "$seed"
python scripts/run_astrominn_historical_replay.py "${POLICY_ARGS[@]}" --device cpu \
  | tee "$POLICY_JOB_DIR/plan_seed_$seed.txt"
if [[ "$option" == --prepare-only ]]; then
  echo 'Prepared only; no jobs submitted.'
  exit 0
fi
[[ "$option" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || {
  echo 'Wall time must have HH:MM:SS format.' >&2; exit 1;
}
manifest="$POLICY_JOB_DIR/submitted_seed_$seed.tsv"
[[ ! -e "$manifest" ]] || {
  echo "Submission record already exists: $manifest. Inspect before retrying." >&2; exit 1;
}
cycles="$(python - "$POLICY_JOB_DIR/plan_seed_$seed.txt" <<'PY'
import json,sys
from pathlib import Path
plan, _ = json.JSONDecoder().raw_decode(Path(sys.argv[1]).read_text().lstrip())
print(plan['max_cycles'])
PY
)"
[[ "$cycles" =~ ^[1-9][0-9]*$ ]] || exit 1
# Reject progressed runs before creating a new chain.
python - "$POLICY_JOB_DIR/registry_hdd.toml" "$POLICY_SUITE_ID" "$seed" <<'PY'
import json,sys,tomllib
from pathlib import Path
registry = tomllib.loads(Path(sys.argv[1]).read_text())
for strategy in ('random','entropy','entropy_diversity','entropy_balanced_explore'):
    path = Path(registry['storage']['persistent_root'])/'runs'/sys.argv[2]/strategy/f'seed_{sys.argv[3]}'/'run_state.json'
    if path.exists():
        state = json.loads(path.read_text())
        if state.get('last_completed_cycle',0) != 0 or state.get('active_cycle') is not None:
            raise SystemExit(f'Run already progressed: {path}. Resume existing jobs instead.')
PY
new_jobs=()
released=0
cleanup() {
  rc=$?
  if [[ $rc -ne 0 && $released -eq 0 && ${#new_jobs[@]} -gt 0 ]]; then
    scancel "${new_jobs[@]}" || true
    echo 'Submission failed; cancelled the new chain. Inspect the retained record.' >&2
  fi
}
trap cleanup EXIT
previous=''
for ((cycle=1; cycle<=cycles; cycle++)); do
  opts=(--parsable --hold --time="$option" --job-name="alP_s${seed}_c${cycle}"
    --output="$POLICY_JOB_DIR/logs/%x_%j.out" --error="$POLICY_JOB_DIR/logs/%x_%j.err")
  if [[ -n "$previous" ]]; then opts+=(--dependency="afterok:$previous"); fi
  response="$(sbatch "${opts[@]}" "$POLICY_REPO/scripts/adaptive/policy_cycle.sbatch" \
    "$POLICY_REPO" "$seed" "$cycle" "$POLICY_SUITE_ID")"
  job_id="${response%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $response" >&2; exit 1; }
  new_jobs+=("$job_id")
  printf '%s\t%s\t%s\n' "$seed" "$cycle" "$job_id" >> "$manifest"
  previous="$job_id"
done
# Release descendants first; none can start before round 1 is released.
for ((i=${#new_jobs[@]}-1; i>=0; i--)); do scontrol release "${new_jobs[$i]}"; done
released=1
cat "$manifest"
squeue -u "${USER:-$(id -un)}" -o '%.18i %.24j %.2t %.10l %R' || true
