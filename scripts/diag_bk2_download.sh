#!/bin/bash
# Fast diagnostic: why did the .bk2 download hang on a compute node?
# Tests (a) network reachability to the CONP RIA host, (b) git-annex remote
# enablement, (c) a single annex get with debug, under several CA settings.
#
#SBATCH -A brics.u6oz
#SBATCH -J diag_bk2
#SBATCH -p workq
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH -o logs/diag_bk2-%j.out
#SBATCH -e logs/diag_bk2-%j.err
#SBATCH --time=00:15:00

set -uo pipefail
echo "Host: $(hostname)  $(date)"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
CA="$REPO/.venv/lib/python3.11/site-packages/certifi/cacert.pem"
# Requires datalad + git-annex on PATH (see fetch_human_data.sh header).
export PATH="$HOME/.local/bin:$PATH"
cd "${MARIO_ROOT:-$HOME/data/mario}"

echo "=================== (a) reachability ==================="
echo "--- curl default CA ---"
curl -sS -m 20 -I https://sftp.conp.ca/ 2>&1 | head -5
echo "--- curl with certifi CA ---"
curl -sS -m 20 --cacert "$CA" -I https://sftp.conp.ca/ 2>&1 | head -5
echo "--- curl github (known-good control) ---"
curl -sS -m 20 --cacert "$CA" -I https://github.com 2>&1 | head -3

echo "=================== (b) annex remotes ==================="
git-annex info 2>&1 | sed -n '/semitrusted/,/transfers/p' | grep -i 'conp\|http\|here' | head

F="sub-02/ses-001/gamelogs/sub-02_ses-001_task-mario_level-w1l1_rep-000.bk2"
echo "=================== (c) get attempts ==================="
echo "--- c1: SSL_CERT_FILE=certifi ---"
rm -f "$(readlink -f "$F")" 2>/dev/null || true
SSL_CERT_FILE="$CA" GIT_SSL_CAINFO="$CA" timeout 60 git-annex get "$F" --from conp-ria-storage-http 2>&1 | tail -6
echo "present c1: $([ -e "$(readlink -f "$F")" ] && echo YES || echo NO)"

if [ ! -e "$(readlink -f "$F")" ]; then
  echo "--- c2: SSL_CERT_DIR=/etc/ssl/certs ---"
  SSL_CERT_DIR=/etc/ssl/certs timeout 60 git-annex get "$F" --from conp-ria-storage-http 2>&1 | tail -6
  echo "present c2: $([ -e "$(readlink -f "$F")" ] && echo YES || echo NO)"
fi

if [ ! -e "$(readlink -f "$F")" ]; then
  echo "--- c3: git-annex --debug (first 40 lines) ---"
  SSL_CERT_FILE="$CA" timeout 60 git-annex get "$F" --from conp-ria-storage-http --debug 2>&1 | grep -iE 'http|tls|ssl|cert|fail|error|certificate|exception' | head -40
fi
echo "Done: $(date)"
