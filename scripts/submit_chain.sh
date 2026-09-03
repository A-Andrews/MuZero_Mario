#!/bin/bash
# Submit N chained (afterany-dependent) autocurriculum jobs that auto-resume
# from outputs/runs/<RUN_NAME>/checkpoints/latest.pt, so training continues
# across SLURM wall-time limits.
#
# Usage:
#   bash scripts/submit_chain.sh <RUN_NAME> <N_JOBS> [extra hydra overrides ...]
#
# Example (a ~7-day budget at Isambard's 24h-per-job QOS cap):
#   bash scripts/submit_chain.sh autocurriculum-v2 7 training.total_env_steps=30_000_000
#
# Warns if N_JOBS cannot reach the env-step budget (see the leg-budget check
# below). MUZERO_CHAIN_DRY_RUN=1 runs that check and submits nothing;
# MUZERO_ENV_STEPS_PER_SEC overrides the assumed throughput.
set -euo pipefail

RUN_NAME="${1:?Usage: submit_chain.sh <RUN_NAME> <N_JOBS> [overrides...]}"
N_JOBS="${2:?Usage: submit_chain.sh <RUN_NAME> <N_JOBS> [overrides...]}"
shift 2

# Extra sbatch flags (e.g. SBATCH_EXTRA="--gres=gpu:2 --cpus-per-gpu=72")
# override the #SBATCH directives baked into submit_autocurriculum.sh.
# Intentionally unquoted below so multiple flags word-split.
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

# --- leg-budget check -------------------------------------------------------
# Why this exists: T7 was submitted 2026-08-24 with the default 4 legs against
# a 60M env-step budget, and stopped on 2026-08-28 at 39.4M / 36.0M having
# simply run out of legs. Every leg exited "TIMEOUT 0:0" — which *is* the
# designed resume path — so nothing looked wrong: no sentinel, no error, no
# queue entry. TRAINING_COMPLETE cancels *surplus* legs; nothing detected the
# shortfall. Measured throughput is ~110 env-steps/s on both the 1-GPU
# specialist and 2-GPU curriculum configs (bench-2gpu 2026-08-12, confirmed by
# T7's 114/104 and T9's 113). Override with MUZERO_ENV_STEPS_PER_SEC.
RATE="${MUZERO_ENV_STEPS_PER_SEC:-110}"
LEG_SCRIPT="$(dirname "$0")/submit_autocurriculum.sh"

# Wall-clock per leg, from the leg script's own #SBATCH --time ([D-]HH:MM:SS).
WALL_SPEC=$(sed -n 's/^#SBATCH --time=//p' "${LEG_SCRIPT}" | head -1)
WALL_SEC=$(awk -v t="${WALL_SPEC}" 'BEGIN{
    d=0; split(t,a,"-");
    if (length(a)>1) { d=a[1]; t=a[2] } 
    split(t,b,":");
    print d*86400 + b[1]*3600 + b[2]*60 + b[3]
}')

# Budget: an explicit override wins, else the config default.
TOTAL=$(printf '%s\n' "$@" | sed -n 's/^training\.total_env_steps=//p' | tail -1 | tr -d '_')
if [ -z "${TOTAL}" ]; then
    TOTAL=$(sed -n 's/^  total_env_steps: *//p' conf/muzero.yaml | head -1 | tr -d '_')
fi

# Progress already banked, so a resumed chain is not told to re-budget the run.
DONE=0
LATEST="outputs/runs/${RUN_NAME}/checkpoints/latest.pt"
if [ -f "${LATEST}" ] && [ -x .venv/bin/python ]; then
    DONE=$(OMP_NUM_THREADS=1 .venv/bin/python -c "
import torch, sys
try:
    print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['env_step']))
except Exception:
    print(0)
" "${LATEST}" 2>/dev/null || echo 0)
fi

if [ -n "${TOTAL}" ] && [ "${WALL_SEC}" -gt 0 ] && [ "${TOTAL}" -gt "${DONE}" ]; then
    PER_LEG=$(( WALL_SEC * RATE ))
    REMAIN=$(( TOTAL - DONE ))
    NEEDED=$(( (REMAIN + PER_LEG - 1) / PER_LEG ))
    if [ "${NEEDED}" -gt "${N_JOBS}" ]; then
        REACH=$(( DONE + N_JOBS * PER_LEG ))
        echo "" >&2
        echo "WARNING: ${N_JOBS} leg(s) will not reach the env-step budget." >&2
        echo "  budget          ${TOTAL} env steps" >&2
        [ "${DONE}" -gt 0 ] && echo "  already done    ${DONE} (resuming)" >&2
        echo "  per leg         ~${PER_LEG} (${WALL_SPEC} at ${RATE} env-steps/s)" >&2
        echo "  ${N_JOBS} leg(s) reach  ~${REACH}" >&2
        echo "  legs needed     ${NEEDED}" >&2
        echo "  The chain will stop mid-run looking exactly like a finished one." >&2
        echo "  Submit ${NEEDED} legs, or re-run with the extra legs later." >&2
        echo "" >&2
    fi
fi

DEP=""
for i in $(seq 1 "${N_JOBS}"); do
    if [ -n "${MUZERO_CHAIN_DRY_RUN:-}" ]; then
        # Budget check only: print the plan, submit nothing.
        echo "would submit job ${i}/${N_JOBS}${DEP:+ (after ${DEP})}"
        DEP="dry-${i}"
        continue
    fi
    if [ -z "${DEP}" ]; then
        JOBID=$(sbatch --parsable ${SBATCH_EXTRA} scripts/submit_autocurriculum.sh "${RUN_NAME}" "$@")
    else
        # afterany (not afterok): resume even if the previous job hit the wall.
        JOBID=$(sbatch --parsable ${SBATCH_EXTRA} --dependency="afterany:${DEP}" scripts/submit_autocurriculum.sh "${RUN_NAME}" "$@")
    fi
    echo "submitted job ${i}/${N_JOBS}: ${JOBID}${DEP:+ (after ${DEP})}"
    DEP="${JOBID}"
done
