#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J updated_runthrough
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -o logs/updated_runthrough-%j.out
#SBATCH -e logs/updated_runthrough-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/muzero-plot-${SLURM_JOB_ID}"
python -m pytest -q tests/test_updated_runthrough.py
python scripts/plot_updated_runthrough.py \
    --development outputs/controller_policy/v1b-20260907 \
    "$@"
