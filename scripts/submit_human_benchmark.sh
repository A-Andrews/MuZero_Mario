#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -J muzero_human_bench
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/muzero_human_bench-%j.out
#SBATCH -e logs/muzero_human_bench-%j.err
#SBATCH --time=06:00:00
#
# Plays each level's BEST checkpoint through the level a handful of times and
# plots those run-throughs against the human ones, writing a vector PDF (plus a
# JSON sidecar) into the repo's images/ directory.
#
# This deliberately answers the narrow question — "can the model get through the
# level, in a human-like number of steps?" — rather than the like-for-like rate
# comparison. It uses best.pt (the checkpoint at each run's peak rolling
# completion rate) because step_*.pt rotates and several runs end in a late
# collapse; every figure and JSON record names the run and training step it
# used, so the checkpoint behind each row is never ambiguous.
#
# Usage:
#   sbatch scripts/submit_human_benchmark.sh                       # all 12 levels, 5 rollouts
#   sbatch scripts/submit_human_benchmark.sh --rollouts 10
#   sbatch scripts/submit_human_benchmark.sh --levels Level1-1 Level3-2
#   sbatch scripts/submit_human_benchmark.sh --deterministic       # 1 greedy rollout/level
#   sbatch scripts/submit_human_benchmark.sh --checkpoint latest.pt
#
# Timing: a rollout is up to --max-steps (2000) env steps at 50 MCTS sims. Levels
# the policy dies on early are quick; ones it survives are not. 12 levels x 5
# rollouts fits the 6h wall with room to spare, but raise --time if you raise
# --rollouts much past 10.
#
# Note the run is written into a git-tracked directory. Commit the PDF/JSON if
# you want the figure versioned; they are small (tens of KB).
set -euo pipefail

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-none}"

# Isambard-AI: no modules needed at runtime — the venv bundles CUDA torch
# (cu126 aarch64 wheels).
source "${SLURM_SUBMIT_DIR:-$PWD}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# matplotlib must not try to build a font cache in $HOME on a compute node.
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR:-$PWD}/outputs/.mplcache"
mkdir -p "${MPLCONFIGDIR}" images logs

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
echo "Git: branch=$(git rev-parse --abbrev-ref HEAD) sha=$(git rev-parse --short HEAD)"

srun python scripts/eval_human_benchmark.py \
    --device cuda \
    --out-dir images \
    "$@"

echo
echo "Artifacts:"
ls -l images/human_vs_agent_runthrough.pdf images/human_vs_agent_runthrough.json 2>/dev/null || true
echo "Done: $(date)"
