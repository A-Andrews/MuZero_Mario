#!/bin/bash
# T7's open question: is curriculum-human's 0.11-vs-0.02 win over
# curriculum-nohuman uniform across levels, or carried by a few?
#
# Both arms are evaluated at latest.pt (both at exactly 60.0M env steps, so
# they are matched on budget) across all 12 curriculum levels, with the human
# comparison on. **Level2-2 is the control**: it is the one level with zero
# human data, so if arm B beats arm A there too, the win is not coming from
# the demonstrations.
#
#   sbatch scripts/submit_t7_per_level.sh
#
#SBATCH -A brics.u6oz
#SBATCH -J t7_per_level
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -o logs/t7_per_level-%j.out
#SBATCH -e logs/t7_per_level-%j.err
#SBATCH --time=08:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(pwd)"
echo "Host: $(hostname)  Started: $(date)  Job: ${SLURM_JOB_ID:-<interactive>}"
source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$REPO/outputs"

for arm in curriculum-human curriculum-nohuman; do
    echo "################ ${arm}"
    python scripts/replay_checkpoint.py \
        "outputs/runs/${arm}/checkpoints/latest.pt" \
        --out "outputs/t7_per_level/${arm}" \
        --device cuda --compare-human "$@"
done
echo "Done: $(date)"
