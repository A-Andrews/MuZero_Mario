#!/bin/bash
#SBATCH -A gpu_costa.prj
#SBATCH -J muzero_mario
#SBATCH -p gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem-per-gpu 80G
#SBATCH -o logs/muzero_mario-%j.out
#SBATCH -e logs/muzero_mario-%j.err
#SBATCH --time=48:00:00

set -euo pipefail

LEVEL="${1:?Usage: sbatch submit_single_level.sh <level> [extra hydra overrides]}"
shift

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Level: ${LEVEL}"

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1 || true
module load FFmpeg || true

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_runs"
mkdir -p "${SLURM_SUBMIT_DIR}/logs" "${SLURM_SUBMIT_DIR}/wandb_runs"

cd "${SLURM_SUBMIT_DIR}"
srun python scripts/train_muzero.py "env.levels=[${LEVEL}]" "$@"

echo "Done: $(date)"
