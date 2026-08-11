#!/bin/bash
#SBATCH -A brics.u6oz
#SBATCH -J muzero_greedy_scan
#SBATCH -p workq
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/muzero_greedy_scan-%j.out
#SBATCH -e logs/muzero_greedy_scan-%j.err
#SBATCH --time=02:00:00
#
# Is greedy completion a property of the *policy* or a knife-edge of *weights*?
#
# T1 found the fully-greedy cell (eps=0, temp=0, pb_c=1.25) completes Level1-1
# off `best.pt` (train step 696,189), while the in-training greedy replay eval
# logged completed=0 at step 701,222 — same sims (50), same pb_c, same
# max_steps, 5K training steps apart. And in-training replay eval only runs
# every `training.replay_every_train_steps` = 50,000 steps, so the whole
# "greedy replay is 0 at every checkpoint" claim rests on 15 single
# deterministic trajectories across a 780K-step run.
#
# This scans that one greedy cell across every retained checkpoint of a run.
# The env has no stochasticity, so one episode per checkpoint is exact, not a
# sample — the output is a step-indexed 0/1 trace of whether the greedy policy
# finishes the level.
#
# Usage:
#   sbatch scripts/submit_greedy_ckpt_scan.sh <run_name> <level> [extra args]
#   sbatch scripts/submit_greedy_ckpt_scan.sh level1-1-diag-v1 Level1-1
set -euo pipefail

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-none}"

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

RUN="${1:?usage: submit_greedy_ckpt_scan.sh <run_name> <level> [args...]}"
LEVEL="${2:?usage: submit_greedy_ckpt_scan.sh <run_name> <level> [args...]}"
shift 2 || true

CKPT_DIR="outputs/runs/${RUN}/checkpoints"
OUT="outputs/greedy_scan/${LEVEL}_${RUN}_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

# best.pt first, then the retained step_*.pt in numeric order. The `.tmp`
# glob exclusion matters: level1-1-diag-v1 has a stale step_508785.pt.tmp
# (695 bytes, an interrupted save) that is not a loadable checkpoint.
CKPTS=("${CKPT_DIR}/best.pt")
while IFS= read -r c; do CKPTS+=("$c"); done < <(
    find "${CKPT_DIR}" -maxdepth 1 -name 'step_*.pt' | sort -V
)

echo "Run:        ${RUN}"
echo "Level:      ${LEVEL}"
echo "Checkpoints: ${#CKPTS[@]}"
echo "Output dir: ${OUT}"

for ckpt in "${CKPTS[@]}"; do
    tag="$(basename "${ckpt}" .pt)"
    echo "=== ${tag} ($(date +%H:%M:%S))"
    srun python scripts/eval_sweep.py "${ckpt}" \
        --levels "${LEVEL}" \
        --out "${OUT}/${tag}" \
        --device cuda \
        --episodes 1 \
        --max-steps 2000 \
        --eps 0 \
        --temperature 0 \
        --pb-c-init 1.25 \
        --no-video \
        "$@"
done

# One file to read: header from the first summary, then every data row tagged
# with the checkpoint it came from.
COMBINED="${OUT}/scan.csv"
first=1
for d in "${OUT}"/*/; do
    tag="$(basename "${d}")"
    [ -f "${d}/summary.csv" ] || continue
    if [ "${first}" = 1 ]; then
        printf 'checkpoint,%s\n' "$(head -1 "${d}/summary.csv")" > "${COMBINED}"
        first=0
    fi
    tail -n +2 "${d}/summary.csv" | sed "s/^/${tag},/" >> "${COMBINED}"
done

echo "Combined: ${COMBINED}"
cat "${COMBINED}" || true
echo "Done: $(date)"
