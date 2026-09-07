#!/bin/bash
# Is the root visit distribution flat because of leaf_batch, too few
# simulations, or a genuinely flat policy head? Sweeps both mechanical
# suspects against the same bank of states.
#
#   sbatch scripts/submit_diag_sharpness.sh <ckpt> <Level> [more pairs...]
#
#SBATCH -A brics.u6oz
#SBATCH -J diag_sharp
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -o logs/diag_sharp-%j.out
#SBATCH -e logs/diag_sharp-%j.err
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
export OMP_NUM_THREADS=1
OUT="outputs/diag_sharpness/${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
echo "Host: $(hostname)  $(date)"
while [ "$#" -ge 2 ]; do
  CKPT="$1"; LEVEL="$2"; shift 2
  echo "================ $LEVEL"
  python scripts/diag_search_sharpness.py "$CKPT" --level "$LEVEL" --device cuda \
      --leaf-batches 1 4 --sims 32 50 200 --json "$OUT/${LEVEL}.json" \
      || echo "FAILED: $LEVEL"
done
echo "Done: $(date)"
