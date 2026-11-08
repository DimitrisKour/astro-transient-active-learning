#!/usr/bin/env bash
# New, independent CPU-only cache of complete historical TRAIN and fixed validation.
set -euo pipefail
MODE="${1:---prepare}"
[[ "$MODE" == --prepare || "$MODE" == --submit || "$MODE" == --retry || "$MODE" == --status ]] || {
  echo 'Usage: --prepare, --submit, --retry, --status' >&2; exit 2;
}
REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
export FUSION_FULL_SOURCE_PLAN=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_matrix_v1/plan.json
export FUSION_FULL_CACHE_ROOT=/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_train_cache_v1
PIPE=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_cache_pipeline_v1
export FUSION_FULL_SHARDS=24
PARALLEL=6
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -u scripts/research/fusion_image/cache/fusion_full_historical_cache_v1.py prepare \
   --source-plan "$FUSION_FULL_SOURCE_PLAN" \
   --cache-root "$FUSION_FULL_CACHE_ROOT" --shards "$FUSION_FULL_SHARDS"
if [[ "$MODE" == --prepare ]]; then
  echo 'Verified full historical TRAIN + validation. NO jobs submitted.'
  exit 0
fi
if [[ "$MODE" == --status ]]; then
  python -u scripts/research/fusion_image/cache/fusion_full_historical_cache_v1.py status \
    --source-plan "$FUSION_FULL_SOURCE_PLAN" \
    --cache-root "$FUSION_FULL_CACHE_ROOT" --shards "$FUSION_FULL_SHARDS"
  if [[ -f "$PIPE/submitted_jobs.tsv" ]]; then cat "$PIPE/submitted_jobs.tsv"; fi
  exit 0
fi
if [[ -f "$FUSION_FULL_CACHE_ROOT/cache_manifest.json" ]]; then
  echo 'Full historical cache already finalized; no new jobs required.'
  exit 0
fi
if [[ "$MODE" == --submit && -f "$PIPE/submitted_jobs.tsv" ]]; then
  echo 'Already submitted; use --retry after prior jobs stop and logs are inspected.' >&2
  exit 1
fi
if [[ "$MODE" == --retry && ! -f "$PIPE/submitted_jobs.tsv" ]]; then
  echo 'No earlier submission; use --submit.' >&2
  exit 1
fi
if [[ "$MODE" == --retry ]]; then
  while IFS=$'\t' read -r kind jid; do
    if [[ "$jid" =~ ^[0-9]+$ ]] && squeue -j "$jid" -h 2>/dev/null | grep -q .; then
      echo "Prior job $kind ($jid) is still pending/running. Do not overlap writers." >&2
      exit 1
    fi
  done < "$PIPE/submitted_jobs.tsv"
fi
mkdir -p "$PIPE/logs"
missing=()
for ((i=0;i<FUSION_FULL_SHARDS;i++)); do
  if [[ ! -f "$FUSION_FULL_CACHE_ROOT/shard_$(printf '%03d' "$i")/done.json" ]]; then
    missing+=("$i")
  fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
  array=$(IFS=,; echo "${missing[*]}")
  echo "Requesting ${#missing[@]} CPU shards, at most $PARALLEL simultaneous. NO GPU jobs."
  shards_id=$(sbatch --parsable --array="${array}%${PARALLEL}" \
    --export=ALL,FUSION_FULL_SOURCE_PLAN,FUSION_FULL_CACHE_ROOT,FUSION_FULL_SHARDS \
    --output="$PIPE/logs/shard_%A_%a.out" --error="$PIPE/logs/shard_%A_%a.err" \
    scripts/research/fusion_image/cache/fusion_full_historical_cache_shard.sbatch)
  dep="afterok:$shards_id"
else
  shards_id=none
  dep=""
fi
if [[ -n "$dep" ]]; then
  join_id=$(sbatch --parsable --dependency="$dep" \
    --export=ALL,FUSION_FULL_SOURCE_PLAN,FUSION_FULL_CACHE_ROOT,FUSION_FULL_SHARDS \
    --output="$PIPE/logs/join_%j.out" --error="$PIPE/logs/join_%j.err" \
    scripts/research/fusion_image/cache/fusion_full_historical_cache_finalize.sbatch)
else
  join_id=$(sbatch --parsable \
    --export=ALL,FUSION_FULL_SOURCE_PLAN,FUSION_FULL_CACHE_ROOT,FUSION_FULL_SHARDS \
    --output="$PIPE/logs/join_%j.out" --error="$PIPE/logs/join_%j.err" \
    scripts/research/fusion_image/cache/fusion_full_historical_cache_finalize.sbatch)
fi
{
  printf 'kind\tjob_id\n'
  printf 'cpu_shards\t%s\n' "$shards_id"
  printf 'cpu_finalize\t%s\n' "$join_id"
} > "$PIPE/submitted_jobs.tsv"
echo "Submitted CPU jobs only. Shards $shards_id; finalize $join_id."
echo "Track with: squeue -u \"\$USER\" or bash scripts/research/fusion_image/cache/submit_fusion_full_historical_cache_v1.sh --status"
