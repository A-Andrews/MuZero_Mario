#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J failure_branches
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o logs/failure_branches-%A_%a.out
#SBATCH -e logs/failure_branches-%A_%a.err
set -euo pipefail
MODE="${1:?validate, full, or aggregate}"
OUT="${2:?absolute project output path}"
case "$OUT" in /projects/*) ;; *) exit 2 ;; esac
cd "${SLURM_SUBMIT_DIR:?submit from repo}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ "$MODE" = validate ]; then
    python scripts/prepare_controller_diagnostic.py --template docs/failure_branches_v1.json --out "$OUT"
fi
SOURCE_ROOT="$OUT/source"
export PYTHONPATH="$SOURCE_ROOT"
if [ "$MODE" = validate ]; then
    python -m pytest -q "$SOURCE_ROOT/tests/test_failure_branches.py" \
        "$SOURCE_ROOT/tests/test_stall_pulses.py" "$SOURCE_ROOT/tests/test_stall_sampling.py"
    python "$SOURCE_ROOT/scripts/diagnose_failure_branches.py" --out "$OUT" --mode prepare
    # Exercise every failure signature, both levels, a successful reference,
    # and the special greedy-versus-gated comparison; never select by branch outcome.
    CASES=$(python - "$OUT/branch_plan.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
seen=set(); selected=[]
for c in p['cases']:
    key=(c['level'],c['group'])
    if key not in seen:
        selected.append(str(c['case_id'])); seen.add(key)
print(' '.join(selected))
PY
)
    for CASE in $CASES; do
        python "$SOURCE_ROOT/scripts/diagnose_failure_branches.py" --out "$OUT" --mode run --case "$CASE" --smoke
    done
elif [ "$MODE" = full ]; then
    python "$SOURCE_ROOT/scripts/diagnose_failure_branches.py" --out "$OUT" --mode run --case "${SLURM_ARRAY_TASK_ID:?array required}"
elif [ "$MODE" = aggregate ]; then
    python "$SOURCE_ROOT/scripts/diagnose_failure_branches.py" --out "$OUT" --mode aggregate
else
    exit 2
fi
echo "Completed $MODE job=$SLURM_JOB_ID host=$(hostname)"
