#!/bin/bash
# T8 step 1: dump one specialist's teacher corpus for a single level.
#
# Fanned out per level rather than one job over all 23, because cost is
# dominated by the weak levels: a specialist completing at 0.03 burns its whole
# attempt budget for a handful of episodes, and 23 of those in series would blow
# the wall. Per level, the worst case is bounded.
#
#   for L in $(...); do sbatch scripts/submit_dump_specialists.sh "$L"; done
#
#SBATCH -A brics.u6oz
#SBATCH -J dump_spec
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -o logs/dump_spec-%j.out
#SBATCH -e logs/dump_spec-%j.err
#SBATCH --time=10:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(pwd)"
LEVEL="${1:?usage: sbatch submit_dump_specialists.sh <LEVEL> [extra args]}"
shift
echo "Host: $(hostname)  Started: $(date)  Job: ${SLURM_JOB_ID:-<interactive>}  Level: ${LEVEL}"
source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS=1

python scripts/dump_specialist_trajectories.py \
    --levels "${LEVEL}" \
    --report "dump_report_${LEVEL}.json" \
    --max-attempts-per-level 60 \
    "$@"
echo "Done: $(date)"
