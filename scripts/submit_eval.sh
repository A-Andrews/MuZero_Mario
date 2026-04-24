#!/bin/bash
#SBATCH -A gpu_costa.prj
#SBATCH -J muzero_eval
#SBATCH -p gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/muzero_eval-%j.out
#SBATCH -e logs/muzero_eval-%j.err
#SBATCH --time=02:00:00

set -euo pipefail

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1 || true
module load FFmpeg || true

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
