#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -J muzero_diag_policy
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/muzero_diag_policy-%j.out
#SBATCH -e logs/muzero_diag_policy-%j.err
#SBATCH --time=01:00:00
#
# Answer "the model only ever plays one action" for a set of checkpoints:
# policy-head prior, MCTS visit distribution and the taken action, measured
# both along a greedy rollout and on off-policy human states.
#
# Usage:
#   sbatch scripts/submit_diag_policy.sh <ckpt> <Level> [more ckpt/level pairs...]
set -euo pipefail

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="outputs/diag_policy/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"

while [ "$#" -ge 2 ]; do
  CKPT="$1"; LEVEL="$2"; shift 2
  echo ""
  echo "================================================================"
  echo "== $LEVEL  $CKPT"
  echo "================================================================"
  python scripts/diag_policy_collapse.py "$CKPT" --level "$LEVEL" \
    --device cuda --json "$OUT_DIR/${LEVEL}_$(basename "$(dirname "$(dirname "$CKPT")")").json" || \
    echo "FAILED: $LEVEL $CKPT"
done

echo "Finished: $(date)"
