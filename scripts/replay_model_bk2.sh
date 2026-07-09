#!/bin/bash
# Turn each run's latest saved greedy-replay .bk2 set into a model_progress JSON
# (no MCTS — just replay the recorded movies), then regenerate the figure.
#
#   sbatch scripts/replay_model_bk2.sh
#
#SBATCH -A costa.prj
#SBATCH -J replay_model_bk2
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem 8G
#SBATCH -o logs/replay_model_bk2-%j.out
#SBATCH -e logs/replay_model_bk2-%j.err
#SBATCH --time=00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(pwd)"
echo "Host: $(hostname)  $(date)"

module purge 2>/dev/null || true
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
source "$REPO/.venv/bin/activate"
export LD_LIBRARY_PATH="/well/costa/users/zqa082/conda/skylake/envs/ctm-vgdl-py38/lib:${LD_LIBRARY_PATH:-}"
INT="$REPO/mario.stimuli"

latest_bk2_dir () {  # highest step_<N> dir that actually contains Level*.bk2
    local d best=""
    for d in $(ls -d "outputs/runs/$1"/videos/step_*/ 2>/dev/null | sort -t_ -k2 -n); do
        if ls "$d"Level*.bk2 >/dev/null 2>&1; then best="$d"; fi
    done
    echo "$best"
}

LD_DIR=$(latest_bk2_dir autocurriculum-longdecay)
M50_DIR=$(latest_bk2_dir autocurriculum-medium50)
V2_DIR=$(latest_bk2_dir autocurriculum-medium50-v2)
echo "longdecay   .bk2: $LD_DIR"
echo "medium50    .bk2: $M50_DIR"
echo "medium50-v2 .bk2: $V2_DIR"

python analysis/comparison/replay_model_bk2.py "$INT" "$LD_DIR"  "$REPO/analysis/comparison/model_progress.json"
python analysis/comparison/replay_model_bk2.py "$INT" "$M50_DIR" "$REPO/analysis/comparison/model_progress_medium50.json"
python analysis/comparison/replay_model_bk2.py "$INT" "$V2_DIR"  "$REPO/analysis/comparison/model_progress_medium50v2.json"

echo "==> regenerating figure"
python analysis/comparison/plot_human_vs_model.py
echo "Done: $(date)"
