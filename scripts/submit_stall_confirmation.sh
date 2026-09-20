#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J stall_confirm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o logs/stall_confirm-%j.out
#SBATCH -e logs/stall_confirm-%j.err
set -euo pipefail
MODE="${1:?validate, full, or audit}"
OUT="${2:?absolute project output path}"
case "$OUT" in /projects/*) ;; *) exit 2 ;; esac
cd "${SLURM_SUBMIT_DIR:?submit from repo}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ "$MODE" = validate ]; then
    python scripts/prepare_controller_diagnostic.py --template docs/stall_sampling_confirmation_v1.json --out "$OUT"
fi
SOURCE_ROOT="$OUT/source"
export PYTHONPATH="$SOURCE_ROOT"
if [ "$MODE" = validate ]; then
    python -m pytest -q "$SOURCE_ROOT/tests/test_stall_sampling.py" \
        "$SOURCE_ROOT/tests/test_controller_diagnostic.py" "$SOURCE_ROOT/tests/test_controller_coverage.py" \
        "$SOURCE_ROOT/tests/test_policy_followup.py"
    python "$SOURCE_ROOT/scripts/check_stall_confirmation.py" --experiment "$OUT"
    OLD=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_experiment"])' "$OUT/manifest.json")
    for LEVEL in Level1-1 Level6-1 Level1-3; do
        python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" --manifest "$OLD/manifest.json" \
            --out "$OUT/smoke/$LEVEL" --level "$LEVEL" --smoke --episodes 1
    done
    python "$SOURCE_ROOT/scripts/check_stall_confirmation.py" --experiment "$OUT" --smoke
elif [ "$MODE" = full ]; then
    python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" --manifest "$OUT/manifest.json" \
        --out "$OUT/confirmation/${3:?level required}" --level "$3"
elif [ "$MODE" = audit ]; then
    python "$SOURCE_ROOT/scripts/audit_stall_sampling.py" --experiment "$OUT"
else
    exit 2
fi
echo "Completed $MODE job=$SLURM_JOB_ID host=$(hostname)"
