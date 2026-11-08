#!/usr/bin/env bash
# Submit bounded, resumable L0+validation CPU preprocessing shards and one GPU benchmark.
# First --prepare, then --submit. If shards fail, inspect logs and rerun with --retry.
set -euo pipefail
MODE="${1:---prepare}"
[[ "$MODE" == --prepare || "$MODE" == --submit || "$MODE" == --retry ]] || {
  echo 'Use --prepare, --submit, or --retry' >&2; exit 2; }
REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
PLAN=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_matrix_v1/plan.json
CACHE=/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_l0_cache_v3
PIPELINE=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_cache_pipeline_v3
N_SHARDS=24
MAX_PARALLEL=6
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export FUSION_V2_SOURCE_PLAN="$PLAN" FUSION_V2_CACHE_ROOT="$CACHE"
python -u scripts/archive/fusion_image/early_l0/fusion_cache_v2.py prepare \
  --source-plan "$PLAN" --cache-root "$CACHE" --shards "$N_SHARDS"
if [[ "$MODE" == --prepare ]]; then
  echo 'Prepared L0+validation shard plan; zero jobs submitted.'
  exit 0
fi
if [[ "$MODE" == --submit && -f "$PIPELINE/submitted_jobs.tsv" ]]; then
  echo 'Already submitted. Inspect existing jobs; use --retry only after failures.' >&2
  exit 1
fi
if [[ "$MODE" == --retry && ! -f "$PIPELINE/submitted_jobs.tsv" ]]; then
  echo 'Use --submit for the first attempt.' >&2; exit 1
fi
if [[ "$MODE" == --retry ]]; then
  # Old array/dependency jobs must have stopped before any retry. Prevent
  # simultaneous shard writers and duplicate GPU benchmarks.
  while IFS=$'\t' read -r kind jid; do
    if [[ "$jid" =~ ^[0-9]+$ ]] && squeue -j "$jid" -h 2>/dev/null | grep -q .; then
      echo "Previous $kind job $jid remains active/pending; inspect it before retrying" >&2
      exit 1
    fi
  done < "$PIPELINE/submitted_jobs.tsv"
fi
mkdir -p "$PIPELINE/logs"
if [[ -e "$CACHE/cache_manifest.json" ]]; then
  echo 'Cache is already finalized; do not rebuild. Inspect the existing GPU benchmark.' >&2
  exit 1
fi
missing=()
for ((i=0;i<N_SHARDS;i++)); do
  if [[ ! -f "$CACHE/shard_$(printf '%03d' "$i")/done.json" ]]; then
    missing+=("$i")
  fi
done
if [[ "${#missing[@]}" -gt 0 ]]; then
  indices=$(IFS=,; echo "${missing[*]}")
  echo "Submitting ${#missing[@]} remaining CPU shards; array=${indices}%${MAX_PARALLEL}"
  array_id=$(sbatch --parsable --array="${indices}%${MAX_PARALLEL}" \
    --export=ALL,FUSION_V2_SOURCE_PLAN,FUSION_V2_CACHE_ROOT \
    --output="$PIPELINE/logs/shard_%A_%a.out" \
    --error="$PIPELINE/logs/shard_%A_%a.err" \
    scripts/archive/fusion_image/early_l0/fusion_cache_v2_shard.sbatch)
  dependency="afterok:${array_id}"
else
  array_id=none
  dependency=""
  echo 'All cache shards already completed; submitting finalize only.'
fi
if [[ -n "$dependency" ]]; then
  final_id=$(sbatch --parsable --dependency="$dependency" \
    --export=ALL,FUSION_V2_SOURCE_PLAN,FUSION_V2_CACHE_ROOT \
    --output="$PIPELINE/logs/join_%j.out" --error="$PIPELINE/logs/join_%j.err" \
    scripts/archive/fusion_image/early_l0/fusion_cache_v2_finalize.sbatch)
else
  final_id=$(sbatch --parsable \
    --export=ALL,FUSION_V2_SOURCE_PLAN,FUSION_V2_CACHE_ROOT \
    --output="$PIPELINE/logs/join_%j.out" --error="$PIPELINE/logs/join_%j.err" \
    scripts/archive/fusion_image/early_l0/fusion_cache_v2_finalize.sbatch)
fi
bench_id=$(sbatch --parsable --dependency="afterok:${final_id}" \
    --output="$PIPELINE/logs/bench_%j.out" --error="$PIPELINE/logs/bench_%j.err" \
    scripts/archive/fusion_image/early_l0/fusion_cache_v2_benchmark.sbatch)
{
  printf 'job\tjob_id\n'
  printf 'cpu_shards\t%s\n' "$array_id"
  printf 'cpu_finalize\t%s\n' "$final_id"
  printf 'gpu_full_L0_one_epoch\t%s\n' "$bench_id"
} > "$PIPELINE/submitted_jobs.tsv"
echo "Submitted CPU array $array_id, join $final_id, GPU pilot $bench_id"
echo "Results/logs: $PIPELINE"
