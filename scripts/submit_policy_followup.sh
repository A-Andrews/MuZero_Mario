#!/bin/bash
# Frozen confirmation, gradient probes, and human action-distribution figures.
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J policy_followup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -o logs/policy_followup-%j.out
#SBATCH -e logs/policy_followup-%j.err
set -euo pipefail
MODE="${1:?provide validate, confirmation, or gradients}"
OUT="${2:?provide absolute project output directory}"
OLD="${3:?provide absolute frozen development directory}"
case "$OUT" in /projects/*) ;; *) exit 2 ;; esac
REPO="${SLURM_SUBMIT_DIR:?submit from repository}"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ "$MODE" = validate ]; then
    python scripts/prepare_controller_diagnostic.py --template docs/controller_policy_confirmation_v1.json --out "$OUT"
fi
SOURCE_ROOT="$OUT/source"
export PYTHONPATH="$SOURCE_ROOT"
if [ "$MODE" = validate ]; then
    python -m pytest -q "$SOURCE_ROOT/tests/test_controller_diagnostic.py" \
        "$SOURCE_ROOT/tests/test_policy_followup.py" "$SOURCE_ROOT/tests/test_mcts.py" \
        "$SOURCE_ROOT/tests/test_targets.py"
    python "$SOURCE_ROOT/scripts/plot_action_distributions.py" --experiment "$OLD" \
        --human-dir "$REPO/outputs/human_trajectories" --out "$OUT/figures"
    for LEVEL in Level1-1 Level6-1; do
        python "$SOURCE_ROOT/scripts/diagnose_policy_gradients.py" \
            --checkpoint "$OUT/checkpoints/$LEVEL.pt" --level "$LEVEL" \
            --banks-dir "$OLD/development/$LEVEL" --out "$OUT/smoke/$LEVEL/gradients.json" \
            --max-episodes 3 --roots-per-episode 2
        python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" --manifest "$OLD/manifest.json" \
            --level "$LEVEL" --out "$OUT/smoke/$LEVEL/controllers" --smoke --episodes 1 \
            --conditions greedy sampled
    done
elif [ "$MODE" = confirmation ]; then
    LEVEL="${4:?provide level}"
    python "$SOURCE_ROOT/scripts/eval_controller_diagnostic.py" --manifest "$OUT/manifest.json" \
        --level "$LEVEL" --out "$OUT/confirmation/$LEVEL"
elif [ "$MODE" = gradients ]; then
    LEVEL="${4:?provide level}"
    python "$SOURCE_ROOT/scripts/diagnose_policy_gradients.py" \
        --checkpoint "$OUT/checkpoints/$LEVEL.pt" --level "$LEVEL" \
        --banks-dir "$OLD/development/$LEVEL" --out "$OUT/gradients/$LEVEL.json"
else
    echo "Unknown mode: $MODE" >&2
    exit 2
fi
echo "Completed $MODE on $(hostname) job $SLURM_JOB_ID"
