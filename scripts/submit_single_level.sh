#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -J muzero_mario
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 72
#SBATCH --mem-per-gpu 110G
#SBATCH -o logs/muzero_mario-%j.out
#SBATCH -e logs/muzero_mario-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=austin.andrews@reuben.ox.ac.uk

set -euo pipefail

LEVEL="${1:?Usage: sbatch submit_single_level.sh <level> [extra hydra overrides]}"
shift

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Level: ${LEVEL}"

# Isambard-AI: no modules needed at runtime — the venv bundles CUDA torch
# (cu126 aarch64 wheels) and imageio-ffmpeg ships a static ffmpeg binary.
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Do NOT let wandb wrap the console: its wrap_raw redirect buffers stderr and
# a crashing learner's traceback never reaches the .err file (the 2026-08-12
# T6 fleet died 12x with empty logs because of this).
export WANDB_CONSOLE=off
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_runs"
mkdir -p "${SLURM_SUBMIT_DIR}/logs" "${SLURM_SUBMIT_DIR}/wandb_runs"

cd "${SLURM_SUBMIT_DIR}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Git: branch=${GIT_BRANCH} sha=${GIT_SHA}"

# Auto-tag every run with SLURM job id, level, git branch, and git sha so
# wandb runs are traceable back to logs AND to the code version that produced
# them. Caller can still add more tags/notes via extra args.
srun python scripts/train_muzero.py \
    "env.levels=[${LEVEL}]" \
    "++wandb.tags=[slurm-${SLURM_JOB_ID},level-${LEVEL},branch-${GIT_BRANCH},sha-${GIT_SHA}]" \
    "++wandb.group=branch-${GIT_BRANCH}" \
    "$@"

echo "Done: $(date)"
