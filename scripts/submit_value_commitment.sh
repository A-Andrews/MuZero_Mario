#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J value_commitment
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o logs/value_commitment-%A_%a.out
#SBATCH -e logs/value_commitment-%A_%a.err
set -euo pipefail
MODE="${1:?validate, full, or aggregate}"
OUT="${2:?absolute project output directory}"
case "$OUT" in /projects/*) ;; *) exit 2 ;; esac
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ "$MODE" = validate ]; then
    python scripts/prepare_controller_diagnostic.py --template docs/value_commitment_v1.json --out "$OUT"
fi
SOURCE_ROOT="$OUT/source"
export PYTHONPATH="$SOURCE_ROOT"
RUNNER="$SOURCE_ROOT/scripts/diagnose_value_commitment.py"
if [ "$MODE" = validate ]; then
    python -m pytest -q "$SOURCE_ROOT/tests/test_value_commitment.py" \
        "$SOURCE_ROOT/tests/test_failure_branches.py" "$SOURCE_ROOT/tests/test_stall_sampling.py"
    python "$RUNNER" --out "$OUT" --mode prepare
    ROOTS=$(python - "$OUT/plan.json" <<'PY'
import json,sys
plan=json.load(open(sys.argv[1]))
seen=set(); selected=[]
for root in plan['roots']:
    case=root['case']; key=(case['level'],case['group'])
    if key not in seen:
        seen.add(key); selected.append(str(root['root_id']))
print(' '.join(selected))
PY
)
    for ROOT_INDEX in $ROOTS; do
        python "$RUNNER" --out "$OUT" --mode both --root "$ROOT_INDEX" --smoke
    done
elif [ "$MODE" = full ]; then
    python "$RUNNER" --out "$OUT" --mode both --root "${SLURM_ARRAY_TASK_ID:?}"
elif [ "$MODE" = aggregate ]; then
    python "$RUNNER" --out "$OUT" --mode aggregate
else
    exit 2
fi
echo "Completed $MODE job=$SLURM_JOB_ID host=$(hostname)"
