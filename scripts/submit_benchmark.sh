#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -J muzero_bench
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 16
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/bench-%j.out
#SBATCH -e logs/bench-%j.err
#SBATCH --time=00:30:00

set -euo pipefail

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Isambard-AI: no modules needed at runtime — the venv bundles CUDA torch
# (cu126 aarch64 wheels).
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

cd "${SLURM_SUBMIT_DIR}"

echo ""
echo "=== Benchmark: 12 workers, 20 decisions, 50 simulations, device=cuda ==="
python scripts/benchmark_inference.py \
    --workers 12 --decisions 20 --num-simulations 50 \
    --device cuda \
    --max-batch 48 --max-wait-ms 1.0

echo ""
echo "Done: $(date)"
