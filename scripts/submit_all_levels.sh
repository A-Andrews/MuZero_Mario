#!/bin/bash
# Submit one SLURM training job per Mario level.
# Usage: bash scripts/submit_all_levels.sh [extra hydra overrides]
#
# All extra args are forwarded to every job (e.g. training.total_env_steps=2_000_000).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LEVELS=(
    Level1-1
    Level1-2
    Level1-3
    Level2-1
    Level2-2
    Level2-3
    Level3-1
    Level3-2
    Level3-3
    Level4-1
    Level4-2
    Level4-3
)

echo "Submitting ${#LEVELS[@]} jobs..."
for LEVEL in "${LEVELS[@]}"; do
    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/submit_single_level.sh" "${LEVEL}" "$@")
    echo "  ${LEVEL}  ->  job ${JOB_ID}"
done
echo "Done."
