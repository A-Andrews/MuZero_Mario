#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J branch_resume_check
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH -o logs/branch_resume_check-%j.out
#SBATCH -e logs/branch_resume_check-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_failure_branch_resume.py --experiment \
    /projects/u6oz/atdandrews/MuZero_Mario/outputs/controller_policy/failure-branches-v1b-20260916
