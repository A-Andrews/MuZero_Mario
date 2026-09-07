#!/bin/bash
# Lesion study: re-initialise each MuZero component and measure the deficit.
#
#   sbatch scripts/submit_lesion_eval.sh --levels Level3-3 Level6-1
#   sbatch scripts/submit_lesion_eval.sh --levels Level1-1 --conditions intact value policy
#
# The Mario counterpart of the Towers-of-Hanoi lesion experiments; see
# src/muzero/lesion.py for the conventions shared with that study.
#
#SBATCH -A brics.u6oz
#SBATCH -J muzero_lesion
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -o logs/muzero_lesion-%j.out
#SBATCH -e logs/muzero_lesion-%j.err
#SBATCH --time=12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(pwd)"
echo "Host: $(hostname)  Started: $(date)  Job: ${SLURM_JOB_ID:-<interactive>}"
echo "Git: $(git rev-parse --short HEAD 2>/dev/null)"

source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS=1

# Default output is namespaced by job id so parallel submissions never collide.
OUT="outputs/lesion/lesion_eval-${SLURM_JOB_ID:-local}.json"
python scripts/lesion_eval.py --out "$OUT" "$@"
echo "Done: $(date)"
