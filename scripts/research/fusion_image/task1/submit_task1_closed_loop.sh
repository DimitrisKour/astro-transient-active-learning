#!/usr/bin/env bash
# No experiments run on login. This script only validates, reports, or submits.
set -euo pipefail
MODE="${1:---status}"
REPO="${AL_PIPELINE_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task1_closed_loop_grouped_v1
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
case "$MODE" in
  --prepare)python scripts/research/fusion_image/task1/task1_closed_loop.py prepare;;
  --check)python scripts/research/fusion_image/task1/task1_closed_loop.py check;;
  --status)
    python scripts/research/fusion_image/task1/task1_closed_loop.py status
    python scripts/research/fusion_image/task1/task1_schedule.py status
    echo '--- SLURM states for this account ---'
    squeue -u "$USER" -h -o '%.2t' | sort | uniq -c || true
    echo '--- First 15 queued/running jobs ---'
    squeue -u "$USER" -h -o '%.18i %.2t %.34j %.10M %.10l %.35R' | sed -n '1,15p' || true;;
  --submit-l0)python scripts/research/fusion_image/task1/task1_schedule.py l0 "${@:2}";;
  --submit-phase1)python scripts/research/fusion_image/task1/task1_schedule.py phase1 "${@:2}";;
  --submit-phase2)python scripts/research/fusion_image/task1/task1_schedule.py phase2 "${@:2}";;
  --report)python scripts/research/fusion_image/task1/task1_closed_loop.py report;;
  --report-with-test)python scripts/research/fusion_image/task1/task1_closed_loop.py report --include-test;;
  *) echo 'Usage: --prepare | --check | --status | --submit-l0 [--limit N] | --submit-phase1 [--limit N] | --submit-phase2 [--limit N] | --report | --report-with-test' >&2;exit 2;;
esac
