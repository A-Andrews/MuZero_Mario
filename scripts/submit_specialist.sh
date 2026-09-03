#!/bin/bash
# One chained single-level specialist run (T6), using the exact recipe
# validated by level1-1-diag-v1 (0 -> 0.84) and level1-2-diag-v1 (0 -> 0.68).
#
# Why this exists: submit_all_levels.sh submits one *unchained*
# submit_single_level.sh job per level, and that script sets no run_name, so
# it cannot auto-resume. At the 24h workq_qos wall against ~36h of work per
# level, every run would die two-thirds finished with no resume path. This
# wraps submit_chain.sh instead, which gives the stable run dir + latest.pt
# auto-resume + TRAINING_COMPLETE self-cancellation.
#
# Differences from submit_curriculum.sh (the 12-level sibling): one level,
# autocurriculum off, 1 GPU / 32 workers rather than the 2-GPU 64-worker
# split, and the diag runs' shorter temperature schedule. Everything else is
# deliberately identical to the diag runs so the per-level completion rates
# are directly comparable to the 0.84 / 0.68 baselines — including
# training.lr_decay_steps, left at the conf default of 400_000, which is what
# those runs used.
#
# Two deliberate departures from the diag recipe, both from T1 (2026-08-11):
#   - env.noop_max/skip_to_control (T2.3) come from the conf defaults, so
#     these runs have stochastic starts and the diag runs did not. Episode
#     lengths therefore are not comparable, completion rates still are.
#   - the root-Dirichlet anneal (T2.1) below decays eps 0.25 -> 0.05 by train
#     step 500K, where the temperature schedule bottoms out. Without it the
#     policy head is never forced to stand on its own — which is survivable
#     for a specialist you always evaluate with noise on, but fatal for T8,
#     since these checkpoints are the distillation teachers.
#
# Usage:
#   bash scripts/submit_specialist.sh <LEVEL> [N_LEGS] [extra overrides...]
#   bash scripts/submit_specialist.sh Level1-1 2
#
# The whole T6 fleet (12 levels in parallel; the QOS allows 256 jobs):
#   for L in Level1-1 Level1-2 Level1-3 Level2-1 Level2-2 Level2-3 \
#            Level3-1 Level3-2 Level3-3 Level4-1 Level4-2 Level4-3; do
#       bash scripts/submit_specialist.sh "$L" 2
#   done
#
# Run dir: outputs/runs/spec-<level>/  (wandb run id is the same string).
# Note the wandb *group* is "autocurriculum-spec-<level>" — an artifact of
# reusing submit_autocurriculum.sh as the chain leg; filter on the
# job_type=specialist tag instead.
set -euo pipefail

LEVEL="${1:?Usage: submit_specialist.sh <LEVEL> [N_LEGS] [overrides...]}"
N_LEGS="${2:-2}"
shift $(( $# >= 2 ? 2 : $# ))

# Valid levels are whatever state files the integration actually ships,
# so worlds 5-8 work without editing this list again.
VALID=($(ls mario.stimuli/SuperMarioBros-Nes/Level*.state 2>/dev/null \
        | sed 's|.*/||; s|\.state$||'))
if [ "${#VALID[@]}" -eq 0 ]; then
    echo "error: no .state files under mario.stimuli/SuperMarioBros-Nes/" >&2
    exit 1
fi
ok=0
for v in "${VALID[@]}"; do [ "${v}" = "${LEVEL}" ] && ok=1 && break; done
if [ "${ok}" != 1 ]; then
    echo "error: unknown level '${LEVEL}'" >&2
    echo "       expected one of: ${VALID[*]}" >&2
    exit 1
fi

# spec-level1-1 rather than spec-Level1-1: run_name becomes a wandb run id and
# a directory name, both of which are nicer lowercase.
RUN_NAME="spec-$(echo "${LEVEL}" | tr '[:upper:]' '[:lower:]')"

echo "Level:    ${LEVEL}"
echo "Run name: ${RUN_NAME}"
echo "Legs:     ${N_LEGS}"

bash "$(dirname "$0")/submit_chain.sh" "${RUN_NAME}" "${N_LEGS}" \
    model=muzero_mario_medium \
    "env.levels=[${LEVEL}]" \
    autocurriculum.enabled=false \
    muzero.discount=0.999 \
    env.completion_bonus=200 \
    mcts.num_simulations=50 \
    selfplay.num_workers=32 \
    training.total_env_steps=15_000_000 \
    'selfplay.temperature_schedule=[[0,1.0],[100000,0.5],[300000,0.25],[500000,0.1]]' \
    'mcts.root_exploration_eps_schedule=[[0,0.25],[500000,0.05]]' \
    ++wandb.job_type=specialist \
    "$@"
