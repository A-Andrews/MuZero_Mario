#!/bin/bash
# Frozen-checkpoint development evaluation and policy diagnostics, compute only.
# First submit prepare-smoke, then full jobs with afterok on that validation job.
# sbatch --time=01:00:00 scripts/submit_controller_policy_diagnostic.sh prepare-smoke OUT
# sbatch --dependency=afterok:JOB scripts/submit_controller_policy_diagnostic.sh full OUT Level1-1
# All substantial outputs, frozen source and copied checkpoints live under OUT.
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J controller_policy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -o logs/controller_policy-%j.out
#SBATCH -e logs/controller_policy-%j.err
set -euo pipefail

MODE="${1:?usage: MODE OUT [LEVEL]}"
OUT="${2:?provide an absolute output path under /projects}"
case "$OUT" in /projects/*) ;; *) echo "OUT must be under /projects" >&2; exit 2 ;; esac
REPO="${SLURM_SUBMIT_DIR:?submit from the repository}"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "Host: $(hostname) Job: ${SLURM_JOB_ID} Mode: $MODE"

if [ "$MODE" = prepare-smoke ]; then
    python scripts/prepare_controller_diagnostic.py \
        --template docs/controller_policy_diagnostic_v1.json --out "$OUT"
fi

SOURCE_ROOT="$OUT/source"
export PYTHONPATH="$SOURCE_ROOT"
MANIFEST="$OUT/manifest.json"

if [ "$MODE" = prepare-smoke ]; then
    python -m pytest -q \
        "$SOURCE_ROOT/tests/test_controller_diagnostic.py" \
        "$SOURCE_ROOT/tests/test_policy_target_diagnostics.py" \
        "$SOURCE_ROOT/tests/test_policy_batchnorm_diagnostic.py" \
        "$SOURCE_ROOT/tests/test_mcts.py" \
        "$SOURCE_ROOT/tests/test_targets.py"
    for LEVEL in Level1-1 Level6-1; do
        python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" \
            --manifest "$MANIFEST" --level "$LEVEL" \
            --out "$OUT/smoke/$LEVEL" --device cuda --smoke --episodes 1
        python "$SOURCE_ROOT/scripts/diagnose_policy_targets.py" \
            --checkpoint "$OUT/checkpoints/$LEVEL.pt" --level "$LEVEL" \
            --banks-dir "$OUT/smoke/$LEVEL" --out "$OUT/smoke/$LEVEL/policy_targets.json" \
            --device cuda --max-episodes 5 --states-per-episode 8 --search-states 8
        python "$SOURCE_ROOT/scripts/diagnose_policy_batchnorm.py" \
            --checkpoint "$OUT/checkpoints/$LEVEL.pt" --level "$LEVEL" \
            --banks-dir "$OUT/smoke/$LEVEL" --out "$OUT/smoke/$LEVEL/policy_batchnorm.json" \
            --device cuda --max-episodes 5 --states-per-episode 8 --batch-size 16
    done
    echo "Validation complete: $(date -u)"
elif [ "$MODE" = full ]; then
    LEVEL="${3:?full requires a level}"
    python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" \
        --manifest "$MANIFEST" --level "$LEVEL" \
        --out "$OUT/development/$LEVEL" --device cuda
    python "$SOURCE_ROOT/scripts/diagnose_policy_targets.py" \
        --checkpoint "$OUT/checkpoints/$LEVEL.pt" --level "$LEVEL" \
        --banks-dir "$OUT/development/$LEVEL" --out "$OUT/development/$LEVEL/policy_targets.json" \
        --device cuda --max-episodes 15 --states-per-episode 24 --search-states 64
    python "$SOURCE_ROOT/scripts/diagnose_policy_batchnorm.py" \
        --checkpoint "$OUT/checkpoints/$LEVEL.pt" --level "$LEVEL" \
        --banks-dir "$OUT/development/$LEVEL" --out "$OUT/development/$LEVEL/policy_batchnorm.json" \
        --device cuda --max-episodes 15 --states-per-episode 24 --batch-size 32
    echo "Development diagnostic complete for $LEVEL: $(date -u)"
else
    echo "Unknown mode: $MODE" >&2
    exit 2
fi
