#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J human_scope_update
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH -o logs/human_scope_update-%j.out
#SBATCH -e logs/human_scope_update-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/human-scope-${SLURM_JOB_ID}"
python -m pytest -q tests/test_updated_runthrough.py tests/test_human_benchmark.py
python scripts/plot_updated_runthrough.py --development outputs/controller_policy/v1b-20260907 \
    --all-levels --out images/human_vs_agent_runthrough_updated_all_levels
python scripts/plot_updated_runthrough.py --development outputs/controller_policy/v1b-20260907 \
    --all-levels --extension outputs/controller_policy/coverage-v1-20260911 \
    --out images/human_vs_agent_runthrough_all_levels_stochastic
