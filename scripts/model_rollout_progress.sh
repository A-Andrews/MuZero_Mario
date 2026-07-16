#!/bin/bash
# Greedy-MCTS rollout of the current-best checkpoint over the 12 levels,
# recording max-x progress + completion -> analysis/comparison/model_progress.json
# CPU-only (the agent dies early, so ~120 steps/level; fast enough on CPU and
# avoids the GPU queue).
#
#   sbatch scripts/model_rollout_progress.sh [checkpoint]
#
#SBATCH -A brics.u6oz
#SBATCH -J model_rollout
#SBATCH -p workq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -o logs/model_rollout-%j.out
#SBATCH -e logs/model_rollout-%j.err
#SBATCH --time=04:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(pwd)"
echo "Host: $(hostname)  Started: $(date)  Job: ${SLURM_JOB_ID:-<interactive>}"

CKPT="${1:-$REPO/outputs/runs/autocurriculum-longdecay/checkpoints/latest.pt}"
echo "Checkpoint: $CKPT"

source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS=4

python analysis/comparison/model_rollout_progress.py "$CKPT" \
    --device cpu \
    --out "$REPO/analysis/comparison/model_progress.json"

echo "Done: $(date)"
