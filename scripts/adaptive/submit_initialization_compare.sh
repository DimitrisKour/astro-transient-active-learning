#!/bin/bash -l
# One fit per GPU job, cycle-dependent arrays per seed. Defaults: 10/15/20 min.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
comparison_id="${INIT_ID:-astrominn_initialization_compare_v1}"
registry="${INIT_REGISTRY:-/projects/bcrv/dkourtesis/applecider_adaptive/cycle_jobs_v1/registry_hdd.toml}"
persistent="${INIT_PERSISTENT_ROOT:-/projects/bcrv/dkourtesis/applecider_adaptive}"
data="${INIT_DATA_ROOT:-/work/hdd/bcrv/dkourtesis/applecider_adaptive}"
parallel="${INIT_PARALLEL_PER_SEED:-1}"
max_cycle="${INIT_MAX_CYCLE:-9}"
read -r -a seeds <<< "${INIT_SEEDS:-135 246 357}"
prepare_only=0
retry=0
times=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-only) prepare_only=1; shift ;;
    --retry-incomplete) retry=1; shift ;;
    --help) echo 'Usage: bash scripts/adaptive/submit_initialization_compare.sh [EARLY_TIME] [MIDDLE_TIME] [LATE_TIME] [--prepare-only | --retry-incomplete]'; exit 0 ;;
    *) times+=("$1"); shift ;;
  esac
done
[[ ${#times[@]} -le 3 ]] || { echo 'At most three wall times are accepted.' >&2; exit 1; }
early="${times[0]:-00:10:00}"
middle="${times[1]:-00:15:00}"
late="${times[2]:-00:20:00}"
for t in "$early" "$middle" "$late"; do
  [[ "$t" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ && "$t" != 00:00:00 ]] || { echo 'Positive HH:MM:SS wall times required.' >&2; exit 1; }
done
[[ "$parallel" =~ ^[1-9][0-9]*$ && "$comparison_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 1
[[ "$max_cycle" =~ ^[1-9]$ ]] || exit 1
[[ $prepare_only -eq 0 || $retry -eq 0 ]] || exit 1
if [[ "${INIT_SKIP_ACTIVATE:-0}" != 1 ]]; then
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
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
prep=(--repo "$repo" --registry "$registry" --comparison-id "$comparison_id" --persistent-root "$persistent"
  --data-root "$data" --max-cycle "$max_cycle" --seeds "${seeds[@]}")
[[ -z "${INIT_SOURCE_SUITE:-}" ]] || prep+=(--source-suite "$INIT_SOURCE_SUITE")
[[ -z "${INIT_SOURCE_ARTIFACT_ROOT:-}" ]] || prep+=(--source-artifact-root "$INIT_SOURCE_ARTIFACT_ROOT")
[[ -z "${INIT_LEGACY_CODE_DIR:-}" ]] || prep+=(--legacy-code-dir "$INIT_LEGACY_CODE_DIR")
python "$repo/scripts/adaptive/prepare_initialization_compare.py" "${prep[@]}"
metadata="$persistent/initialization_compare/$comparison_id"
plan="$metadata/plan.json"
if [[ $prepare_only -eq 1 ]]; then
  echo "Prepared only; no jobs submitted. Plan: $plan"; exit 0
fi
exec 9>"$metadata/submission.lock"
flock -n 9 || { echo 'Another submission is running.' >&2; exit 1; }
record="$metadata/submissions.tsv"
if [[ -e "$record" && $retry -eq 0 ]]; then
  echo 'Submission record exists. Use --retry-incomplete after earlier jobs stop.' >&2; exit 1
fi
if [[ -e "$record" ]]; then
  ids="$(python - "$record" <<'PY'
import csv,sys
with open(sys.argv[1]) as f: rows=list(csv.DictReader(f,delimiter='\t'))
print(','.join(r['job_id'] for r in rows))
PY
)"
  if [[ -n "$ids" ]]; then
    active="$(squeue --noheader --jobs="$ids" --format=%i)"
    [[ -z "$active" ]] || { echo 'Earlier comparison jobs are active/pending; refusing duplicates.' >&2; exit 1; }
  fi
fi
selection="$(python - "$plan" <<'PY'
import sys
from pathlib import Path
from al_pipeline.astronomy.initialization_compare import batch_selection
from al_pipeline.astronomy.training_diagnostics import load_plan
path=Path(sys.argv[1]);plan=load_plan(path)
for seed in plan['seeds']:
    for cycle in range(1,plan['max_cycle']+1):
        indices=batch_selection(path,seed,cycle)
        if indices: print(str(seed)+'\t'+str(cycle)+'\t'+','.join(map(str,indices)))
PY
)"
[[ -n "$selection" ]] || { echo 'All comparison fits completed; no jobs submitted.'; exit 0; }
if [[ ! -e "$record" ]]; then
  (set -o noclobber; printf 'seed\tcycle\tindices\twalltime\tjob_id\n' > "$record")
fi
new_jobs=()
all_released=0
cleanup() {
  rc=$?
  if [[ $rc -ne 0 && $all_released -eq 0 && ${#new_jobs[@]} -gt 0 ]]; then
    scancel "${new_jobs[@]}" || true
    echo 'Submission failed; cancelled only newly submitted arrays. Record retained.' >&2
  fi
}
trap cleanup EXIT
previous=''
previous_seed=''
while IFS=$'\t' read -r seed cycle indices; do
  if [[ "$seed" != "$previous_seed" ]]; then previous=''; previous_seed="$seed"; fi
  walltime="$late"
  [[ "$cycle" -gt 2 ]] || walltime="$early"
  [[ "$cycle" -lt 3 || "$cycle" -gt 5 ]] || walltime="$middle"
  opts=(--parsable --hold --time="$walltime" --array="$indices%$parallel" --job-name="alI_s${seed}_c${cycle}"
    --output="$metadata/logs/%x_%A_%a.out" --error="$metadata/logs/%x_%A_%a.err" --kill-on-invalid-dep=yes)
  [[ -z "$previous" ]] || opts+=(--dependency="afterok:$previous")
  response="$(sbatch "${opts[@]}" "$metadata/source/scripts/adaptive/initialization_cycle.sbatch" "$plan" "$metadata/source")"
  job_id="${response%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $response" >&2; exit 1; }
  new_jobs+=("$job_id")
  printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "$cycle" "$indices" "$walltime" "$job_id" >> "$record"
  previous="$job_id"
done <<< "$selection"
for ((i=${#new_jobs[@]}-1; i>=0; i--)); do scontrol release "${new_jobs[$i]}"; done
all_released=1
cat "$record"
echo "One fit per GPU job; at most $parallel concurrent fits per seed. No acquisition queued."
squeue -u "${USER:-$(id -un)}" -o '%.18i %.28j %.2t %.10l %R' || true
