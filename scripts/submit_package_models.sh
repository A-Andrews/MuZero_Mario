#!/bin/bash
# Package the per-level specialist checkpoints into a downloadable zip.
#
#   sbatch scripts/submit_package_models.sh              # models only (~590 MB zip)
#   sbatch scripts/submit_package_models.sh --human-data # + the 1.9G human corpus
#
# Runs on a compute node: it reads ~3.2 GB of checkpoints and deflates ~1.1 GB,
# which is exactly the kind of job the login node reaps.
#
#SBATCH -A brics.u6oz
#SBATCH -J package_models
#SBATCH -p workq
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -o logs/package_models-%j.out
#SBATCH -e logs/package_models-%j.err
#SBATCH --time=01:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(pwd)"
echo "Host: $(hostname)  Started: $(date)  Job: ${SLURM_JOB_ID:-<interactive>}"

source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS=1

python scripts/package_models.py "$@"
echo "Done: $(date)"
