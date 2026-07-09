#!/bin/bash
# GPU greedy-MCTS rollout of a checkpoint over the 12 levels (max-x + score).
# Use for the heavier medium net (192ch/10, 50 sims) where CPU is too slow.
#
#   sbatch scripts/model_rollout_gpu.sh <checkpoint> <out.json>
#
#SBATCH -A gpu_costa.prj
#SBATCH -J model_rollout_gpu
#SBATCH -p gpu_a100_80gb,gpu_a100_40gb,gpu_v100_32gb,gpu_v100_16gb,gpu_rtx8000_48gb,gpu_rtx6000_24gb,gpu_l4_24gb,gpu_l40s_48gb
#SBATCH -q gpu_bmrc_24hr
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

module purge 2>/dev/null || true
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
module load CUDA/12.1.1 2>/dev/null || true
source "$REPO/.venv/bin/activate"
export LD_LIBRARY_PATH="/well/costa/users/zqa082/conda/skylake/envs/ctm-vgdl-py38/lib:${LD_LIBRARY_PATH:-}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python analysis/comparison/model_rollout_progress.py "$CKPT" --device cuda --out "$OUT"
echo "Done: $(date)"
