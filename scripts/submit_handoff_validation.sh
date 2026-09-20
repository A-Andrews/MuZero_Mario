#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -p workq
#SBATCH -J handoff_check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o logs/handoff_check-%j.out
#SBATCH -e logs/handoff_check-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
ulimit -c 0
OUT="${1:?shared output directory}"
mkdir -p "$OUT"
.venv/bin/python -m pytest -q tests/test_networks.py tests/test_mcts.py tests/test_stochastic_start.py tests/test_stall_sampling.py > "$OUT/tests.txt"
.venv/bin/python scripts/validate_handoff_compute.py --out "$OUT" "${@:2}"
