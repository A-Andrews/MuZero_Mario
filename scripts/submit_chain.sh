#!/bin/bash
# Submit N chained (afterany-dependent) autocurriculum jobs that auto-resume
# from outputs/runs/<RUN_NAME>/checkpoints/latest.pt, so training continues
# across SLURM wall-time limits.
#
# Usage:
#   bash scripts/submit_chain.sh <RUN_NAME> <N_JOBS> [extra hydra overrides ...]
#
# Example (a ~7.5-day budget at 60h per job):
#   bash scripts/submit_chain.sh autocurriculum-v2 3 training.total_env_steps=30_000_000
set -euo pipefail

RUN_NAME="${1:?Usage: submit_chain.sh <RUN_NAME> <N_JOBS> [overrides...]}"
N_JOBS="${2:?Usage: submit_chain.sh <RUN_NAME> <N_JOBS> [overrides...]}"
shift 2

DEP=""
for i in $(seq 1 "${N_JOBS}"); do
    if [ -z "${DEP}" ]; then
        JOBID=$(sbatch --parsable scripts/submit_autocurriculum.sh "${RUN_NAME}" "$@")
    else
        # afterany (not afterok): resume even if the previous job hit the wall.
        JOBID=$(sbatch --parsable --dependency="afterany:${DEP}" scripts/submit_autocurriculum.sh "${RUN_NAME}" "$@")
    fi
    echo "submitted job ${i}/${N_JOBS}: ${JOBID}${DEP:+ (after ${DEP})}"
    DEP="${JOBID}"
done
