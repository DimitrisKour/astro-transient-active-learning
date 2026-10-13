#!/bin/bash -l
# Usage: bash scripts/adaptive/submit_training_diagnostics.sh [REPEAT_TIME] [ORDER_TIME]
#        [--prepare-only | --retry-incomplete] [--only all|transition|order]
set -euo pipefail
DIAG_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DIAG_ID="${DIAG_ID:-astrominn_training_diagnostics_v1}"
DIAG_REGISTRY="${DIAG_REGISTRY:-/projects/bcrv/dkourtesis/applecider_adaptive/cycle_jobs_v1/registry_hdd.toml}"
DIAG_PERSISTENT_ROOT="${DIAG_PERSISTENT_ROOT:-/projects/bcrv/dkourtesis/applecider_adaptive}"
DIAG_DATA_ROOT="${DIAG_DATA_ROOT:-/work/hdd/bcrv/dkourtesis/applecider_adaptive}"
DIAG_PARALLEL="${DIAG_PARALLEL:-3}"
DIAG_ORDER_PARALLEL="${DIAG_ORDER_PARALLEL:-1}"
repeat_time=00:15:00
order_time=00:20:00
prepare_only=0
retry=0
only=all
times=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-only) prepare_only=1; shift ;;
    --retry-incomplete) retry=1; shift ;;
    --only) only="${2:?--only needs all, transition or order}"; shift 2 ;;
    --help)
      echo 'Usage: bash scripts/adaptive/submit_training_diagnostics.sh [REPEAT_TIME] [ORDER_TIME] [--prepare-only | --retry-incomplete] [--only all|transition|order]'
      exit 0 ;;
    *) times+=("$1"); shift ;;
  esac
done
[[ ${#times[@]} -le 2 ]] || { echo 'At most two wall times are accepted.' >&2; exit 1; }
[[ ${#times[@]} -lt 1 ]] || repeat_time="${times[0]}"
[[ ${#times[@]} -lt 2 ]] || order_time="${times[1]}"
for walltime in "$repeat_time" "$order_time"; do
  [[ "$walltime" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ && "$walltime" != 00:00:00 ]] || {
    echo 'Wall times must be positive HH:MM:SS values.' >&2; exit 1;
  }
done
[[ "$only" == all || "$only" == transition || "$only" == order ]] || exit 1
[[ "$DIAG_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 1
[[ "$DIAG_PARALLEL" =~ ^[1-9][0-9]*$ && "$DIAG_ORDER_PARALLEL" =~ ^[1-9][0-9]*$ ]] || exit 1
[[ $prepare_only -eq 0 || $retry -eq 0 ]] || {
  echo '--prepare-only and --retry-incomplete cannot be combined.' >&2; exit 1;
}
if [[ "${DIAG_SKIP_ACTIVATE:-0}" != 1 ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" != al-pipeline ]]; then
    module load miniforge3-python
    eval "$(conda shell.bash hook)"
    conda activate al-pipeline
  fi
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi
export PYTHONPATH="$DIAG_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
DIAG_META="$DIAG_PERSISTENT_ROOT/training_diagnostics/$DIAG_ID"
prep=(--repo "$DIAG_REPO" --registry "$DIAG_REGISTRY" --diagnostic-id "$DIAG_ID"
  --persistent-root "$DIAG_PERSISTENT_ROOT" --data-root "$DIAG_DATA_ROOT"
  --repeat-count "${DIAG_REPEAT_COUNT:-20}" --first-repeat-seed "${DIAG_FIRST_REPEAT_SEED:-1001}"
  --trajectory-strategy "${DIAG_TRAJECTORY_STRATEGY:-entropy_balanced_explore}"
  --trajectory-seed "${DIAG_TRAJECTORY_SEED:-135}")
[[ -z "${DIAG_SOURCE_SUITE:-}" ]] || prep+=(--source-suite "$DIAG_SOURCE_SUITE")
[[ -z "${DIAG_SOURCE_ARTIFACT_ROOT:-}" ]] || prep+=(--source-artifact-root "$DIAG_SOURCE_ARTIFACT_ROOT")
[[ -z "${DIAG_LEGACY_CODE_DIR:-}" ]] || prep+=(--legacy-code-dir "$DIAG_LEGACY_CODE_DIR")
[[ -z "${DIAG_REPEAT_LOADER_SEED:-}" ]] || prep+=(--repeat-loader-seed "$DIAG_REPEAT_LOADER_SEED")
python "$DIAG_REPO/scripts/adaptive/prepare_training_diagnostics.py" "${prep[@]}"
plan="$DIAG_META/plan.json"
if [[ $prepare_only -eq 1 ]]; then
  echo "Prepared only; no jobs submitted. Plan: $plan"
  exit 0
fi
exec 9>"$DIAG_META/submission.lock"
flock -n 9 || { echo 'Another diagnostic submission is in progress.' >&2; exit 1; }
record="$DIAG_META/submissions.tsv"
if [[ -e "$record" && $retry -eq 0 ]]; then
  echo "Submission record exists: $record. Use --retry-incomplete after all earlier jobs stop." >&2
  exit 1
fi
if [[ -e "$record" ]]; then
  ids="$(python - "$record" <<'PY'
import csv,sys
with open(sys.argv[1]) as f:
    rows=list(csv.DictReader(f,delimiter='\t'))
print(','.join(r['job_id'] for r in rows))
PY
  )"
  if [[ -n "$ids" ]]; then
    active="$(squeue --noheader --jobs="$ids" --format=%i)"
    [[ -z "$active" ]] || {
      echo 'Earlier diagnostic jobs are still active or pending; refusing duplicate fits.' >&2; exit 1;
    }
  fi
fi
selection="$(python - "$plan" "$only" <<'PY'
import json,sys
from pathlib import Path
from al_pipeline.astronomy.training_diagnostics import load_plan, sha256
plan=load_plan(Path(sys.argv[1])); groups={'pilot':[], 'repeat':[], 'order':[]}
for task in plan['tasks']:
    result=Path(plan['metadata_root'])/'tasks'/task['task_id']/'result.json'
    if result.exists():
        saved=json.loads(result.read_text())
        if saved.get('task') != task or saved.get('plan_sha256') != sha256(Path(sys.argv[1])):
            raise SystemExit(f'Completed task identity differs: {result}')
        continue
    if task['anchor']:
        groups['pilot'].append(task['index'])
    elif task['kind']=='transition' and sys.argv[2] in ('all','transition'):
        groups['repeat'].append(task['index'])
    elif task['kind']=='order' and sys.argv[2] in ('all','order'):
        groups['order'].append(task['index'])
for group,indices in groups.items():
    print(group+':'+','.join(map(str,indices)))
PY
)"
mapfile -t batches <<< "$selection"
pilot_indices="${batches[0]#pilot:}"
repeat_indices="${batches[1]#repeat:}"
order_indices="${batches[2]#order:}"
if [[ -z "$pilot_indices$repeat_indices$order_indices" ]]; then
  echo 'All requested fits already completed; no jobs submitted.'
  exit 0
fi
if [[ ! -e "$record" ]]; then
  (set -o noclobber; printf 'phase\tindices\twalltime\tjob_id\n' > "$record")
fi
new_jobs=()
all_released=0
cleanup() {
  rc=$?
  if [[ $rc -ne 0 && $all_released -eq 0 && ${#new_jobs[@]} -gt 0 ]]; then
    scancel "${new_jobs[@]}" || true
    echo 'Submission/release failed; cancelled only this attempt’s new arrays. Record retained.' >&2
  fi
}
trap cleanup EXIT
frozen="$DIAG_META/source"
pilot_job=''
submit_array() {
  local phase="$1" indices="$2" walltime="$3" parallel="$4"
  local response job_id
  local opts=(--parsable --hold --time="$walltime" --array="$indices%$parallel"
    --job-name="alD_$phase" --output="$DIAG_META/logs/%x_%A_%a.out"
    --error="$DIAG_META/logs/%x_%A_%a.err" --kill-on-invalid-dep=yes)
  if [[ "$phase" != pilot && -n "$pilot_job" ]]; then
    opts+=(--dependency="afterok:$pilot_job")
  fi
  response="$(sbatch "${opts[@]}" "$frozen/scripts/adaptive/training_diagnostic.sbatch" "$plan" "$frozen")"
  job_id="${response%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $response" >&2; exit 1; }
  new_jobs+=("$job_id")
  printf '%s\t%s\t%s\t%s\n' "$phase" "$indices" "$walltime" "$job_id" >> "$record"
  [[ "$phase" != pilot ]] || pilot_job="$job_id"
}
[[ -z "$pilot_indices" ]] || submit_array pilot "$pilot_indices" "$repeat_time" 2
[[ -z "$repeat_indices" ]] || submit_array repeat "$repeat_indices" "$repeat_time" "$DIAG_PARALLEL"
[[ -z "$order_indices" ]] || submit_array order "$order_indices" "$order_time" "$DIAG_ORDER_PARALLEL"
# Release descendants first, so none start until the anchor array is released.
for ((i=${#new_jobs[@]}-1; i>=0; i--)); do scontrol release "${new_jobs[$i]}"; done
all_released=1
cat "$record"
echo 'No acquisition cycles queued. Each array element trains one independent saved-manifest fit.'
squeue -u "${USER:-$(id -un)}" -o '%.18i %.24j %.2t %.10l %R' || true
