#!/usr/bin/env bash
# Read-only post-mortem for original three pre-task-0 GPU jobs.
set -u
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_v1
[[ -f "$ROOT/submitted_jobs.tsv" ]] || { echo "No submitted_jobs.tsv in $ROOT" >&2; exit 1; }
echo '=== Submitted jobs ==='
cat "$ROOT/submitted_jobs.tsv"
IDS=$(awk 'NF >=2 && $2 ~ /^[0-9]+$/ {printf "%s%s",sep,$2;sep=","}' "$ROOT/submitted_jobs.tsv")
if [[ -n "$IDS" ]]; then
  echo '=== SLURM state and exit code ==='
  sacct -j "$IDS" -X --format=JobID,JobName%22,State%25,ExitCode,Elapsed,Timelimit,MaxRSS || true
  echo '=== SLURM job-step resource usage ==='
  sacct -j "$IDS" --format=JobID,JobName%22,State%25,ExitCode,Elapsed,MaxRSS || true
fi
for case in constant cosine cosine_aug; do
  echo "=== $case: stderr ==="
  found=0
  for f in "$ROOT"/logs/${case}_*.err; do
    [[ -e "$f" ]] || continue
    found=1
    echo "--- $f"; tail -100 "$f"
  done
  [[ "$found" = 1 ]] || echo '(no stderr log)'
  echo "=== $case: stdout ==="
  found=0
  for f in "$ROOT"/logs/${case}_*.out; do
    [[ -e "$f" ]] || continue
    found=1
    echo "--- $f"; tail -70 "$f"
  done
  [[ "$found" = 1 ]] || echo '(no stdout log)'
  echo "=== $case: partial outputs ==="
  if [[ -d "$ROOT/$case" ]]; then
    find "$ROOT/$case" -maxdepth 1 -type f -printf '%f %s bytes\n'
  else
    echo '(no output directory)'
  fi
done
