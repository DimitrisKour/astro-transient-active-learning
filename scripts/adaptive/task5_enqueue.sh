#!/bin/bash
# Queue one Task 5 policy/cycle fit per held SLURM job.
# Usage: task5_enqueue.sh PLAN_TSV FROZEN_SOURCE JOB_DIR CYCLES
# This file deliberately does not import conda or touch scientific data.
set -euo pipefail
[[ $# -eq 4 ]] || { echo 'Usage: task5_enqueue.sh PLAN SOURCE JOB_DIR CYCLES' >&2; exit 2; }
plan="$1"; source_root="$2"; job_dir="$3"; cycles="$4"
[[ -f "$plan" && -f "$source_root/scripts/adaptive/task5_cycle.sbatch" ]] || {
  echo 'Missing job plan or frozen cycle script' >&2; exit 1;
}
[[ "$cycles" =~ ^[1-9]$ ]] || { echo 'Cycles must be 1..9' >&2; exit 2; }
manifest="$job_dir/submitted.tsv"
[[ ! -e "$manifest" ]] || { echo "Submission exists: $manifest; inspect before retrying." >&2; exit 1; }
mkdir -p "$job_dir/logs"
created=()
released=0
cleanup() {
  rc=$?
  if [[ $rc -ne 0 && $released -eq 0 && ${#created[@]} -gt 0 ]]; then
    scancel "${created[@]}" || true
    echo 'Queue submission failed; newly created jobs cancelled. Check manifest and logs.' >&2
  fi
}
trap cleanup EXIT
# Every (seed, policy) has its own cycle-to-cycle chain.
# Never serialize different policies or seeds.
declare -A previous_job=()
count=0
while IFS=$'\t' read -r seed cycle policy walltime; do
  [[ "$seed" =~ ^(135|246|357)$ && "$cycle" =~ ^[1-9]$ ]] || {
    echo "Malformed queue plan" >&2; exit 1;
  }
  case "$policy" in
    random|entropy|entropy_diversity|entropy_balanced_explore) ;;
    *) echo "Unexpected acquisition policy $policy" >&2; exit 1;;
  esac
  key="${seed}:${policy}"
  previous="${previous_job[$key]:-}"
  # Cycle 1 is independent for every seed/policy. Later cycles must
  # depend on the immediately preceding cycle of the SAME seed/policy.
  if (( cycle == 1 )); then
    [[ -z "$previous" ]] || { echo "Duplicate cycle-1 chain: $key" >&2; exit 1; }
  else
    [[ -n "$previous" ]] || { echo "Missing previous cycle for $key cycle $cycle" >&2; exit 1; }
  fi
  opts=(--parsable --hold --time="$walltime" \
    --job-name="T5_s${seed}_c${cycle}_${policy}" \
    --output="$job_dir/logs/%x_%j.out" --error="$job_dir/logs/%x_%j.err")
  if [[ -n "$previous" ]]; then opts+=(--dependency="afterok:$previous"); fi
  response="$(sbatch "${opts[@]}" "$source_root/scripts/adaptive/task5_cycle.sbatch" \
             "$source_root" "$seed" "$cycle" "$policy")"
  jobid="${response%%;*}"
  [[ "$jobid" =~ ^[0-9]+$ ]] || { echo "Bad sbatch response $response" >&2; exit 1; }
  created+=("$jobid")
  printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "$cycle" "$policy" "$jobid" "$walltime" >> "$manifest"
  previous_job[$key]="$jobid"
  ((count+=1))
done < "$plan"
expected=$((3 * 4 * cycles))
[[ "$count" -eq "$expected" ]] || { echo "Expected $expected jobs, submitted $count" >&2; exit 1; }
for ((i=${#created[@]}-1; i>=0; i--)); do scontrol release "${created[$i]}"; done
released=1
echo "Submitted $count short jobs (12 independent seed/policy chains; one fit/job)."
echo 'Job IDs and walltimes in submitted.tsv.'
squeue -u "${USER:-$(id -un)}" -o '%.18i %.32j %.2t %.10l %R' || true
