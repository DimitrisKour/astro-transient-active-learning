#!/usr/bin/env bash
# CPU-only test-cache preprocessing. Does not compute or reveal test predictions.
set -euo pipefail
MODE="${1:---check}"
REPO="${AL_PIPELINE_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
SHARDS=/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_historical_test_fastcache_grouped_v1_shards
FINAL=/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_historical_test_fastcache_grouped_v1
PIPE=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_test_cache_pipeline_grouped_v1
N=24
PARALLEL=6
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
RUN=(python scripts/research/fusion_image/task0/task0_test_cache_v1.py)
case "$MODE" in
  --check)
    "${RUN[@]}" check
    ;;
  --prepare)
    "${RUN[@]}" prepare --n-shards "$N"
    ;;
  --status)
    if [[ -f "$SHARDS/plan.json" ]]; then "${RUN[@]}" status; else echo 'Not yet prepared'; fi
    [[ ! -f "$PIPE/submitted_jobs.tsv" ]] || cat "$PIPE/submitted_jobs.tsv"
    squeue -u "$USER" || true
    ;;
  --submit|--retry)
    "${RUN[@]}" prepare --n-shards "$N"
    if [[ -f "$FINAL/fast_cache_manifest.json" ]]; then
      echo 'Immutable test cache already published; use --validate.'; exit 0
    fi
    if [[ "$MODE" == --submit && -f "$PIPE/submitted_jobs.tsv" ]]; then
      echo 'Prior submitted jobs recorded; use --status, then --retry if appropriate.' >&2;exit 1
    fi
    if [[ "$MODE" == --retry ]]; then
      if [[ ! -f "$PIPE/submitted_jobs.tsv" ]]; then echo 'No prior submission; use --submit.' >&2;exit 1;fi
      while IFS=$'\t' read -r kind jid; do
        if [[ "$jid" =~ ^[0-9]+$ ]] && squeue -j "$jid" -h 2>/dev/null | grep -q .; then
          echo "Previous $kind job $jid remains active; do not overlap writers" >&2;exit 1
        fi
      done < "$PIPE/submitted_jobs.tsv"
    fi
    mkdir -p "$PIPE/logs"
    missing=()
    for ((i=0;i<N;i++));do
      if [[ ! -f "$SHARDS/shard_$(printf '%03d' "$i")/done.json" ]];then missing+=("$i");fi
    done
    if (( ${#missing[@]} ));then
      selection=$(IFS=,;echo "${missing[*]}")
      jid=$(sbatch --parsable --array="${selection}%${PARALLEL}" \
          --output="$PIPE/logs/shard_%A_%a.out" \
          --error="$PIPE/logs/shard_%A_%a.err" \
          scripts/research/fusion_image/task0/task0_test_cache_shard.sbatch)
      dependence="afterok:$jid"
    else
      jid=none
      dependence=""
    fi
    if [[ -n "$dependence" ]];then
      join=$(sbatch --parsable --dependency="$dependence" \
          --output="$PIPE/logs/join_%j.out" --error="$PIPE/logs/join_%j.err" \
          scripts/research/fusion_image/task0/task0_test_cache_consolidate.sbatch)
    else
      join=$(sbatch --parsable \
          --output="$PIPE/logs/join_%j.out" --error="$PIPE/logs/join_%j.err" \
          scripts/research/fusion_image/task0/task0_test_cache_consolidate.sbatch)
    fi
    { printf 'kind\tjob_id\n';printf 'cpu_shards\t%s\n' "$jid";printf 'cpu_consolidate\t%s\n' "$join"; } > "$PIPE/submitted_jobs.tsv"
    echo "Queued CPU-only test-cache shards: $jid; consolidation: $join; max concurrency=$PARALLEL"
    ;;
  --validate)
    "${RUN[@]}" validate
    "${RUN[@]}" probe
    ;;
  *) echo 'Usage: --check | --prepare | --submit | --status | --retry | --validate' >&2;exit 2;;
esac
