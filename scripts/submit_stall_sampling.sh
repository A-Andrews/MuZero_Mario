#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J stall_sampling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o logs/stall_sampling-%j.out
#SBATCH -e logs/stall_sampling-%j.err
set -euo pipefail
MODE="${1:?prepare-smoke or full}"
OUT="${2:?absolute project output path}"
case "$OUT" in /projects/*) ;; *) exit 2 ;; esac
cd "${SLURM_SUBMIT_DIR:?submit from repo}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ "$MODE" = prepare-smoke ]; then
    python scripts/prepare_controller_diagnostic.py --template docs/stall_sampling_v1.json --out "$OUT"
fi
SOURCE_ROOT="$OUT/source"
export PYTHONPATH="$SOURCE_ROOT"
if [ "$MODE" = prepare-smoke ]; then
    python -m pytest -q "$SOURCE_ROOT/tests/test_stall_sampling.py" \
        "$SOURCE_ROOT/tests/test_controller_diagnostic.py" "$SOURCE_ROOT/tests/test_controller_coverage.py" \
        "$SOURCE_ROOT/tests/test_policy_followup.py"
    for LEVEL in Level1-1 Level6-1 Level1-3; do
        python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" --manifest "$OUT/manifest.json" \
            --out "$OUT/smoke/$LEVEL" --level "$LEVEL" --smoke --episodes 1
    done
    python "$SOURCE_ROOT/scripts/check_coverage_control.py" --experiment "$OUT"
elif [ "$MODE" = full ]; then
    python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" --manifest "$OUT/manifest.json" \
        --out "$OUT/development/${3:?level required}" --level "$3"
else
    exit 2
fi
echo "Completed $MODE job=$SLURM_JOB_ID host=$(hostname)"
