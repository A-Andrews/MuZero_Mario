#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J stall_pulses
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -o logs/stall_pulses-%j.out
#SBATCH -e logs/stall_pulses-%j.err
set -euo pipefail
MODE="${1:?prepare-smoke or full}"
OUT="${2:?absolute project output path}"
case "$OUT" in /projects/*) ;; *) exit 2 ;; esac
cd "${SLURM_SUBMIT_DIR:?submit from repo}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
OLD="/projects/u6oz/atdandrews/MuZero_Mario/outputs/controller_policy/v1b-20260907"
if [ "$MODE" = prepare-smoke ]; then
    python scripts/prepare_controller_diagnostic.py --template docs/stall_pulses_v1.json --out "$OUT"
fi
SOURCE_ROOT="$OUT/source"
export PYTHONPATH="$SOURCE_ROOT"
if [ "$MODE" = prepare-smoke ]; then
    python -m pytest -q "$SOURCE_ROOT/tests/test_stall_pulses.py"
    python "$SOURCE_ROOT/scripts/diagnose_stall_pulses.py" --out "$OUT" --development "$OLD" --prepare
    for LEVEL in Level1-1 Level6-1; do
        python "$SOURCE_ROOT/scripts/diagnose_stall_pulses.py" --out "$OUT" --development "$OLD" --level "$LEVEL" --smoke
    done
elif [ "$MODE" = full ]; then
    python "$SOURCE_ROOT/scripts/diagnose_stall_pulses.py" --out "$OUT" --development "$OLD" --level "${3:?level required}"
else
    exit 2
fi
echo "Completed $MODE job=$SLURM_JOB_ID host=$(hostname)"
