#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J value_targets
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o logs/value_targets-%A_%a.out
#SBATCH -e logs/value_targets-%A_%a.err
set -euo pipefail
MODE="${1:?test, validate, full, or aggregate}"
OUT="${2:?absolute shared output path}"
case "$OUT" in /projects/*|"${SLURM_SUBMIT_DIR:?}"/diagnostic_outputs/*) ;; *) exit 2 ;; esac
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ "$MODE" = test ]; then
    python -m pytest -q tests/test_value_targets_diagnostic.py tests/test_value_commitment.py
    exit
fi
if [ "$MODE" = validate ]; then
    python scripts/prepare_controller_diagnostic.py --template docs/value_targets_v1.json --out "$OUT"
fi
SOURCE_ROOT="$OUT/source"
export PYTHONPATH="$SOURCE_ROOT"
RUNNER="$SOURCE_ROOT/scripts/diagnose_value_targets.py"
if [ "$MODE" = validate ]; then
    python -m pytest -q "$SOURCE_ROOT/tests/test_value_targets_diagnostic.py" "$SOURCE_ROOT/tests/test_value_commitment.py"
    python "$RUNNER" --out "$OUT" --mode prepare
    # Both levels, every source group, intact controls and the late 6-1 obstacle.
    for ROOT_INDEX in 0 2 3 18 35 37 40; do
        python "$RUNNER" --out "$OUT" --mode run --root "$ROOT_INDEX" --smoke
    done
elif [ "$MODE" = full ]; then
    python "$RUNNER" --out "$OUT" --mode run --root "${SLURM_ARRAY_TASK_ID:?}"
elif [ "$MODE" = aggregate ]; then
    python "$RUNNER" --out "$OUT" --mode aggregate
else
    exit 2
fi
echo "Completed $MODE job=$SLURM_JOB_ID host=$(hostname)"
