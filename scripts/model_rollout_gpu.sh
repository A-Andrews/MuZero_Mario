#!/bin/bash
# GPU greedy-MCTS rollout of a checkpoint over the 12 levels (max-x + score).
# Use for the heavier medium net (192ch/10, 50 sims) where CPU is too slow.
#
#   sbatch scripts/model_rollout_gpu.sh <checkpoint> <out.json>
#
#SBATCH -A brics.u6oz
#SBATCH -J model_rollout_gpu
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -o logs/model_rollout_gpu-%j.out
#SBATCH -e logs/model_rollout_gpu-%j.err
#SBATCH --time=04:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(pwd)"
echo "Host: $(hostname)  Started: $(date)  Job: ${SLURM_JOB_ID:-<interactive>}"

CKPT="${1:?usage: sbatch model_rollout_gpu.sh <checkpoint> <out.json>}"
OUT="${2:?usage: sbatch model_rollout_gpu.sh <checkpoint> <out.json>}"
echo "Checkpoint: $CKPT  ->  $OUT"

source "$REPO/.venv/bin/activate"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python analysis/comparison/model_rollout_progress.py "$CKPT" --device cuda --out "$OUT"
echo "Done: $(date)"
