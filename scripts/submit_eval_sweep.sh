#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -J muzero_eval_sweep
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/muzero_eval_sweep-%j.out
#SBATCH -e logs/muzero_eval_sweep-%j.err
#SBATCH --time=03:00:00
#
# Decide whether root Dirichlet noise — not temperature — is what carries the
# 0.59-0.84 self-play completion rate that greedy replay eval never reproduces.
#
# Usage:
#   sbatch scripts/submit_eval_sweep.sh <ckpt> <level> [extra eval_sweep.py args]
#   sbatch scripts/submit_eval_sweep.sh outputs/runs/level1-2-diag-v1/checkpoints/best.pt Level1-2
#
# The default grid is 3 eps x 3 temperature x 2 pb_c_init = 18 cells; the 2
# noise-free/argmax cells collapse to a single deterministic episode each, so
# it is ~16 x 20 + 2 = 322 episodes. At ~300 steps/episode on Level1-2 that is
# well inside the 3h wall time; on a level where the policy survives to the
# 2000-step cap, trim --episodes or the grid.
set -euo pipefail

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-none}"

# Isambard-AI: no modules needed at runtime — the venv bundles CUDA torch
# (cu126 aarch64 wheels) and imageio-ffmpeg ships a static ffmpeg binary.
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

CKPT="${1:?usage: submit_eval_sweep.sh <ckpt> <level> [args...]}"
LEVEL="${2:?usage: submit_eval_sweep.sh <ckpt> <level> [args...]}"
shift 2 || true

OUT="outputs/eval_sweep/${LEVEL}_$(basename "$(dirname "$(dirname "${CKPT}")")")_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

echo "Checkpoint: ${CKPT}"
echo "Level:      ${LEVEL}"
echo "Output dir: ${OUT}"

srun python scripts/eval_sweep.py "${CKPT}" \
    --levels "${LEVEL}" \
    --out "${OUT}" \
    --device cuda \
    --episodes 20 \
    --max-steps 2000 \
    --eps 0 0.1 0.25 \
    --alpha 0.25 \
    --temperature 0 0.1 0.25 \
    --pb-c-init 1.25 2.5 \
    "$@"

echo "Done: $(date)"
