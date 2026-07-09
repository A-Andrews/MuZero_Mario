#!/bin/bash
#SBATCH -A gpu_costa.prj
#SBATCH -J muzero_mario_ac
#SBATCH -p gpu_a100_80gb
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 11
#SBATCH --mem-per-gpu 110G
#SBATCH -o logs/muzero_mario_ac-%j.out
#SBATCH -e logs/muzero_mario_ac-%j.err
#SBATCH --time=2-12:00:00
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

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1 || true
module load FFmpeg || true

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

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

echo "Done: $(date)"
