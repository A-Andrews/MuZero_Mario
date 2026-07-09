#!/bin/bash
# Fetch + replay the human .bk2 gameplay for the model's 12 levels, producing
# analysis/comparison/human_attempts.csv (one row per human attempt with
# completed / max_x / duration). CPU-only; downloads come from the anonymous
# CONP HTTP RIA store.
#
# BMRC compute nodes reach the internet via an HTTP proxy. git-annex refuses to
# use a proxy unless annex.security.allowed-ip-addresses permits it, so we set
# that — without it every annex get fails ("http proxy settings not used ...").
#
#   sbatch scripts/replay_human_bk2.sh
#
#SBATCH -A costa.prj
#SBATCH -J replay_human_bk2
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem 24G
#SBATCH -o logs/replay_human_bk2-%j.out
#SBATCH -e logs/replay_human_bk2-%j.err
#SBATCH --time=03:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(pwd)"
echo "Host: $(hostname)  Started: $(date)  Job: ${SLURM_JOB_ID:-<interactive>}"

MARIO_ROOT="/well/costa/users/zqa082/mario"
INT_PATH="$REPO/mario.stimuli"
OUT="$REPO/analysis/comparison/human_attempts.csv"
LEVELS="Level1-1,Level1-2,Level1-3,Level2-1,Level2-2,Level2-3,Level3-1,Level3-2,Level3-3,Level4-1,Level4-2,Level4-3"
CA="$REPO/.venv/lib/python3.11/site-packages/certifi/cacert.pem"

export PATH="$HOME/.local/bin:/well/costa/users/zqa082/conda/skylake/envs/mario-fmri/bin:$PATH"
export GIT_SSL_CAINFO="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"

cd "$MARIO_ROOT"
echo "proxy: https_proxy=${https_proxy:-<unset>}"
# The compute-node proxy is on a private IP; git-annex blocks private IPs by
# default ("http proxy settings not used due to annex.security.allowed-ip-
# addresses configuration"). Allow it so annex can use the proxy.
git config annex.security.allowed-ip-addresses all

# --- 0. Fail-fast: one get must succeed, else bail in <2 min (not 6 h) -------
TEST="sub-02/ses-001/gamelogs/sub-02_ses-001_task-mario_level-w1l1_rep-000.bk2"
echo "==> Fail-fast download test ..."
timeout 90 git-annex get "$TEST" --from conp-ria-storage-http 2>&1 | tail -3 || true
if [ ! -e "$(readlink -f "$TEST")" ]; then
    echo "ERROR: test .bk2 still not downloaded; aborting before the long phase." >&2
    exit 1
fi
echo "    test OK"

# --- 1. Build explicit file list for the 12 levels, then one batched get -----
LIST="$(mktemp)"
for w in 1 2 3 4; do for l in 1 2 3; do
    find sub-*/ses-*/gamelogs -name "*level-w${w}l${l}_rep-*.bk2" 2>/dev/null
done; done | sort > "$LIST"
echo "==> Downloading $(wc -l < "$LIST") .bk2 (batched, -J16) ..."
xargs -a "$LIST" git-annex get -J16 --from conp-ria-storage-http 2>&1 | grep -iE 'failed|ok$' | tail -3 || true
PRES=$(while read -r f; do [ -e "$(readlink -f "$f")" ] && echo x; done < "$LIST" | wc -l)
echo "    .bk2 present: $PRES / $(wc -l < "$LIST")"

# --- 2. Replay in parallel (project venv + libffi for retro) -----------------
cd "$REPO"
set -e
module purge 2>/dev/null || true
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
source "$REPO/.venv/bin/activate"
export LD_LIBRARY_PATH="/well/costa/users/zqa082/conda/skylake/envs/ctm-vgdl-py38/lib:${LD_LIBRARY_PATH:-}"

echo "==> Replaying ..."
python analysis/comparison/batch_replay_humans.py \
    --mario-root "$MARIO_ROOT" --int-path "$INT_PATH" --out "$OUT" \
    --levels "$LEVELS" --jobs 16

echo "Done: $(date)"
