#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -J muzero_mario_ac
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 72
#SBATCH --mem-per-gpu 110G
#SBATCH -o logs/muzero_mario_ac-%j.out
#SBATCH -e logs/muzero_mario_ac-%j.err
# workq_qos caps wall time at 24h — use scripts/submit_chain.sh for longer
# budgets (afterany-chained jobs auto-resume from checkpoints/latest.pt).
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=austin.andrews@reuben.ox.ac.uk

set -euo pipefail

# Usage:
#   sbatch scripts/submit_autocurriculum.sh <RUN_NAME> [extra hydra overrides ...]
#
# Stable run dir: outputs/runs/<RUN_NAME>/  (checkpoints + videos persist here)
# Wandb run id:   <RUN_NAME>                 (resume=allow)
#
# To chain a follow-up job that picks up where this one stopped:
#   sbatch --dependency=afterany:<JOBID> scripts/submit_autocurriculum.sh <RUN_NAME>
# afterany (not afterok) so we resume even if the previous job hit the wall.

RUN_NAME="${1:?Usage: sbatch submit_autocurriculum.sh <RUN_NAME> [extra hydra overrides]}"
shift

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Run name: ${RUN_NAME}"

# Isambard-AI: no modules needed at runtime — the venv bundles CUDA torch
# (cu126 aarch64 wheels) and imageio-ffmpeg ships a static ffmpeg binary.
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Do NOT let wandb wrap the console: its wrap_raw redirect buffers stderr and
# a crashing learner's traceback never reaches the .err file (the 2026-08-12
# T6 fleet died 12x with empty logs because of this).
export WANDB_CONSOLE=off
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_runs"
mkdir -p "${SLURM_SUBMIT_DIR}/logs" "${SLURM_SUBMIT_DIR}/wandb_runs"

cd "${SLURM_SUBMIT_DIR}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Git: branch=${GIT_BRANCH} sha=${GIT_SHA}"

RUN_DIR="outputs/runs/${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
if [ -f "${RUN_DIR}/checkpoints/latest.pt" ]; then
    echo "Found existing checkpoint — will auto-resume."
else
    echo "No existing checkpoint — starting fresh."
fi

srun python scripts/train_muzero.py \
    "run_name=${RUN_NAME}" \
    "hydra.run.dir=${RUN_DIR}" \
    "++wandb.tags=[slurm-${SLURM_JOB_ID},autocurriculum,run-${RUN_NAME},branch-${GIT_BRANCH},sha-${GIT_SHA}]" \
    "++wandb.group=autocurriculum-${RUN_NAME}" \
    "$@"

# The learner writes TRAINING_COMPLETE once training.total_env_steps is
# reached. Chained (afterany) resume legs are useless past that point, so
# cancel them; the chain is linear, so walk job -> dependent -> dependent.
# (scancel of a pending job still *releases* its own afterany dependents,
# hence the walk instead of a single scancel.)
if [ -f "${RUN_DIR}/TRAINING_COMPLETE" ]; then
    echo "Training complete — cancelling dependent chain legs."
    CUR="${SLURM_JOB_ID}"
    for _ in $(seq 1 20); do
        NEXT=$(squeue -u "${USER}" -h -t PD -o "%i %E" 2>/dev/null \
            | awk -v id="${CUR}" '$2 ~ ("afterany:" id "([^0-9]|$)") {print $1; exit}') || NEXT=""
        [ -z "${NEXT}" ] && break
        echo "  scancel ${NEXT}"
        scancel "${NEXT}" || true
        CUR="${NEXT}"
    done
fi

echo "Done: $(date)"
