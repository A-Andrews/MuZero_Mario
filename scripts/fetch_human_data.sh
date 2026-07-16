#!/bin/bash
# Fetch the human mario.scenes behavioural data needed to compare against the
# MuZero agent. The per-scene info we need (scene_id, outcome=completed/death,
# duration in seconds) lives in the small BIDS events files
# func/*_desc-scenes_events.tsv (~37KB each, ~20MB total) — NOT in the 81MB
# gamelogs.tar archives, which we intentionally skip.
#
# Submit (do not run on the login node):
#   sbatch scripts/fetch_human_data.sh
#
#SBATCH -A brics.u6oz
#SBATCH -J fetch_mario_scenes
#SBATCH -p workq
#SBATCH -c 4
#SBATCH --mem 8G
#SBATCH -o logs/fetch_human_data-%j.out
#SBATCH -e logs/fetch_human_data-%j.err
#SBATCH --time=06:00:00

set -euo pipefail

echo "Host: $(hostname)  Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-<interactive>}"

# Prerequisites on Isambard-AI (not installed by default): `pip install datalad`
# into ~/.local or the venv, plus a git-annex binary on PATH (conda-forge has
# aarch64 builds; there is no system package and no sudo).
export PATH="$HOME/.local/bin:$PATH"

# Point git/curl at the venv's certifi bundle so downloads work identically on
# login and compute nodes regardless of the system CA setup.
CA_BUNDLE="${SLURM_SUBMIT_DIR:-$(pwd)}/.venv/lib/python3.11/site-packages/certifi/cacert.pem"
export GIT_SSL_CAINFO="$CA_BUNDLE"
export SSL_CERT_FILE="$CA_BUNDLE"
export CURL_CA_BUNDLE="$CA_BUNDLE"
export REQUESTS_CA_BUNDLE="$CA_BUNDLE"

echo "datalad:   $(command -v datalad)  $(datalad --version 2>&1 | head -1)"
echo "git-annex: $(command -v git-annex)  $(git-annex version 2>&1 | head -1)"

DEST="${MARIO_SCENES_DIR:-$HOME/data/mario.scenes}"
mkdir -p "$(dirname "$DEST")"

# Clean up a leftover partial clone from a previous failed attempt.
if [ -d "$DEST" ] && [ ! -d "$DEST/.git" ]; then
    echo "==> Removing partial/empty $DEST from a prior failed run"
    rm -rf "$DEST"
fi

# 1. Clone the dataset (metadata + annex symlinks; no large content yet).
if [ ! -d "$DEST/.git" ]; then
    echo "==> Installing mario.scenes (recursive, metadata only) ..."
    datalad install -r -s https://github.com/courtois-neuromod/mario.scenes "$DEST"
else
    echo "==> $DEST already installed; updating subdatasets ..."
    datalad -C "$DEST" get -r -n .   # install any not-yet-cloned subdatasets, no data
fi

cd "$DEST"

# 2. Enumerate the BIDS scene-events sidecars and fetch just those.
echo "==> Locating *_desc-scenes_events.tsv files ..."
mapfile -t EVENTS < <(find . -path '*/func/*_desc-scenes_events.tsv' | sort)
echo "    found ${#EVENTS[@]} events files"

if [ "${#EVENTS[@]}" -eq 0 ]; then
    echo "ERROR: no func/*_desc-scenes_events.tsv found. Dataset layout differs." >&2
    echo "Top-level tree for debugging:" >&2
    find . -maxdepth 3 -type d | head -50 >&2
    exit 1
fi

echo "==> datalad get (events TSVs only) ..."
printf '%s\n' "${EVENTS[@]}" | xargs -r datalad get

# 3. Verify content is present (annex symlinks now point at real files).
PRESENT=0
for f in "${EVENTS[@]}"; do
    [ -s "$(readlink -f "$f")" ] && PRESENT=$((PRESENT + 1))
done
echo "==> ${PRESENT}/${#EVENTS[@]} events files have content on disk."
echo "Sample paths:"
printf '%s\n' "${EVENTS[@]}" | head -5

echo "Done: $(date)"
