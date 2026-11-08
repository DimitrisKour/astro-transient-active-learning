#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---prepare}"
FIT="${2:-}"
ROOT=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_extensions_v1
REFERENCE=/projects/bcrv/dkourtesis/applecider_adaptive/new_image_pre_task0_recipe_resident_v3
REPO=/projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
cd "$REPO"
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate al-pipeline
export PYTHONPATH="/projects/bcrv/dkourtesis:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT=scripts/research/fusion_image/recipe/fusion_pre_task0_extensions.py

verify_probe() {
  python - "$REFERENCE/resident_probe/probe.json" <<'PYPROBE'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit('Missing validated resident-RAM probe: '+str(p))
r=json.loads(p.read_text())
if (r.get('data_access_mode') != 'shared_host_resident_arrays_v3'
    or r.get('train_rows') != 29161 or r.get('validation_rows') != 14148):
    raise SystemExit('Resident RAM probe identity mismatch')
xs=[b['fetch_seconds'] for b in r['batches']['train'][1:]]
if not xs or sorted(xs)[len(xs)//2] > 0.5:
    raise SystemExit('Resident RAM batch fetching still slow')
print('Validated resident-RAM data-access prerequisite')
PYPROBE
}
verify_reference() {
  python - "$REFERENCE/cosine_aug/training_summary.json" <<'PYREF'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit('Reference cosine_aug run not completed: '+str(p))
s=json.loads(p.read_text())
if s.get('status') != 'completed' or s.get('training_rows') != 29161 or s.get('validation_rows') != 14148:
    raise SystemExit('Reference cosine_aug identity mismatch')
r=s.get('identity',{}).get('recipe',{})
if not r.get('image_augmentation') or r.get('lr_schedule') != 'warmup_cosine':
    raise SystemExit('Reference cosine_aug recipe mismatch')
print('Validated completed cosine_aug reference')
PYREF
}
case "$MODE" in
  --check)
    python "$SCRIPT" check
    verify_probe
    verify_reference
    ;;
  --prepare)
    python "$SCRIPT" prepare
    echo "Frozen new extension plan only; no jobs submitted."
    ;;
  --submit-case)
    case "$FIT" in
      cosine_aug_short|cosine_aug_effnum) ;;
      *) echo "Expected cosine_aug_short or cosine_aug_effnum" >&2; exit 2 ;;
    esac
    python "$SCRIPT" prepare
    verify_probe
    verify_reference
    if [[ -e "$ROOT/$FIT" ]]; then
      echo "Existing fit directory $ROOT/$FIT (partial or complete); inspect before retry." >&2
      exit 1
    fi
    if [[ -f "$ROOT/submitted_jobs.tsv" ]] && grep -Eq "^${FIT}[[:space:]]" "$ROOT/submitted_jobs.tsv"; then
      echo "$FIT was already submitted; refusing duplicate." >&2
      exit 1
    fi
    mkdir -p "$ROOT/logs"
    JOB=$(sbatch --parsable --time=00:15:00 \
       --export="ALL,PRETASK_EXT_CASE=$FIT" \
       --output="$ROOT/logs/${FIT}_%j.out" \
       --error="$ROOT/logs/${FIT}_%j.err" \
       scripts/research/fusion_image/recipe/fusion_pre_task0_extensions.sbatch)
    printf '%s\t%s\n' "$FIT" "$JOB" >> "$ROOT/submitted_jobs.tsv"
    echo "Submitted $FIT, SLURM JobID=$JOB, max wall-time=15 minutes"
    ;;
  --status)
    [[ ! -f "$ROOT/submitted_jobs.tsv" ]] || cat "$ROOT/submitted_jobs.tsv"
    squeue -u "$USER" || true
    ;;
  --report) python "$SCRIPT" report ;;
  *) echo "Use --check | --prepare | --submit-case {cosine_aug_short,cosine_aug_effnum} | --status | --report" >&2; exit 2 ;;
esac
