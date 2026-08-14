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

set -euo pipefail

# Isambard-AI: GH200 nodes are 4 GPUs / 288 CPUs / 460G, so one GPU's fair
# share is 72 CPUs + ~110G. selfplay.num_workers defaults to 20 — with this
# allocation it can go up to ~64 (leave a few cores for learner + server).
# No modules needed at runtime: the venv bundles CUDA torch (cu126 aarch64
# wheels) and imageio-ffmpeg ships a static ffmpeg binary.

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"

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
srun python scripts/train_muzero.py "$@"

echo "Done: $(date)"
