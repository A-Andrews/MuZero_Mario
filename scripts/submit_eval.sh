#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -J muzero_eval
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/muzero_eval-%j.out
#SBATCH -e logs/muzero_eval-%j.err
#SBATCH --time=02:00:00

set -euo pipefail

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Isambard-AI: no modules needed at runtime — the venv bundles CUDA torch
# (cu126 aarch64 wheels) and imageio-ffmpeg ships a static ffmpeg binary.
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

cd "${SLURM_SUBMIT_DIR}"

CKPT="${1:-checkpoints/latest.pt}"
shift || true
STEP=$(basename "${CKPT}" .pt)
OUT="videos/${STEP}"
mkdir -p "${OUT}"

echo "Checkpoint: ${CKPT}"
echo "Output dir: ${OUT}"

srun python scripts/replay_checkpoint.py "${CKPT}" --out "${OUT}" --device cuda "$@"

echo "Done: $(date)"
