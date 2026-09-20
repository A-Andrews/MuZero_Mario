#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J controller_coverage
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o logs/controller_coverage-%A_%a.out
#SBATCH -e logs/controller_coverage-%A_%a.err
set -euo pipefail
MODE="${1:?prepare-smoke or full}"
OUT="${2:?absolute project path}"
case "$OUT" in /projects/*) ;; *) exit 2 ;; esac
cd "${SLURM_SUBMIT_DIR:?submit from repository}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ "$MODE" = prepare-smoke ]; then
    python scripts/prepare_controller_diagnostic.py --template docs/controller_coverage_human_v1.json --out "$OUT"
fi
SOURCE_ROOT="$OUT/source"
export PYTHONPATH="$SOURCE_ROOT"
if [ "$MODE" = prepare-smoke ]; then
    python -m pytest -q "$SOURCE_ROOT/tests/test_controller_diagnostic.py" \
        "$SOURCE_ROOT/tests/test_controller_coverage.py" "$SOURCE_ROOT/tests/test_updated_runthrough.py"
    for LEVEL in Level6-1 Level1-3; do
        python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" --manifest "$OUT/manifest.json" \
            --level "$LEVEL" --out "$OUT/smoke/$LEVEL" --smoke --episodes 1
    done
    python "$SOURCE_ROOT/scripts/check_coverage_control.py" --experiment "$OUT"
elif [ "$MODE" = full ]; then
    LEVEL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_levels"][int(sys.argv[2])])' "$OUT/manifest.json" "${SLURM_ARRAY_TASK_ID:?array job required}")"
    python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" --manifest "$OUT/manifest.json" \
        --level "$LEVEL" --out "$OUT/development/$LEVEL"
else
    exit 2
fi
echo "Completed $MODE job=$SLURM_JOB_ID host=$(hostname)"
