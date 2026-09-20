#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J policy_search_figures
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH -o logs/policy_search_figures-%j.out
#SBATCH -e logs/policy_search_figures-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/policy-search-${SLURM_JOB_ID}"
python -m pytest -q tests/test_policy_search_distributions.py
python scripts/plot_policy_search_distributions.py "$@"
