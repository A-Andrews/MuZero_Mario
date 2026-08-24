#!/bin/bash
# A specialist run with the human-demo rescue turned on, for levels where
# pure self-play never completes the level even once (T6: Level1-3, Level4-3).
#
# Why this exists (diagnosis 2026-08-18, from the T6 fleet #2 wandb history):
# the Level1-3 and Level4-3 specialists ran ~97k and ~107k episodes at 15M env
# steps and completed the level exactly zero times. Both are pinned at a pit
# gap — 73% of Level1-3 episodes end at x=750-999 (median 889) and 91% of
# Level4-3 episodes end at x=500-749 (median 729), by *death*, not timeout
# (median episode length 149 and 130 agent steps). `mcts_root_q_mean` is flat
# for the whole run (27.4 -> 27.3 and 22.7 -> 20.6) where a healthy run triples
# it (Level3-2: 39.8 -> 125.3), because a pit gap is a discontinuous reward
# cliff: shaped progress reward rises to the gap edge and any failed jump is
# instant death, so with env.done_on_life_loss=true "walk to the edge and stop"
# is a genuine local optimum that the value head correctly certifies.
#
# Escaping needs one lucky deep excursion to the flag. Level3-3 — the same
# level archetype — got one at 29% into its run and went 0.1% -> 28%
# completion within a single decile, its per-decile max final_x climbing
# 1165 -> 1848 -> 2498 beforehand. Level1-3's *fell* instead: its deepest
# excursion ever (1726) came in its first decile while eps was still ~0.25,
# decaying to 1046 by the end. Random exploration was never going to find it.
#
# Two changes, both aimed squarely at that:
#   1. imitation.* — humans complete exactly these levels (w1l3: 78 completions
#      / 125k agent steps; w4l3: 60 / 142k), so BC pretrain + a mixed human
#      stream puts the completion bonus into the value head's training data
#      instead of waiting for self-play to stumble onto it.
#   2. mcts.root_exploration_eps_gate_on_completion — the anneal clock does not
#      start until the run completes a level once, so a run that has never seen
#      the reward keeps its full exploration budget. Belt and braces: if the
#      human data alone is enough this changes nothing, and if it isn't, at
#      least exploration is not withdrawn from a policy that never escaped.
#
# Everything else is deliberately identical to submit_specialist.sh so the
# completion rates stay directly comparable to the T6 numbers.
#
# Usage:
#   bash scripts/submit_specialist_imitation.sh <LEVEL> [N_LEGS] [overrides...]
#   bash scripts/submit_specialist_imitation.sh Level1-3 2
#
# Run dir: outputs/runs/spec-imit-<level>/. The run name is deliberately NOT
# the T6 spec-<level> one: imitation must only ever be enabled on a fresh
# run_name, since resuming a non-imitation checkpoint below pretrain_steps
# would BC-pretrain an already-trained net.
set -euo pipefail

LEVEL="${1:?Usage: submit_specialist_imitation.sh <LEVEL> [N_LEGS] [overrides...]}"
N_LEGS="${2:-2}"
shift $(( $# >= 2 ? 2 : $# ))

VALID=(Level1-1 Level1-2 Level1-3 Level2-1 Level2-2 Level2-3
       Level3-1 Level3-2 Level3-3 Level4-1 Level4-2 Level4-3)
ok=0
for v in "${VALID[@]}"; do [ "${v}" = "${LEVEL}" ] && ok=1 && break; done
if [ "${ok}" != 1 ]; then
    echo "error: unknown level '${LEVEL}'" >&2
    echo "       expected one of: ${VALID[*]}" >&2
    exit 1
fi

RUN_NAME="spec-imit-$(echo "${LEVEL}" | tr '[:upper:]' '[:lower:]')"

if [ -d "outputs/runs/${RUN_NAME}/checkpoints" ]; then
    echo "note: ${RUN_NAME} already has checkpoints — this will resume, not restart." >&2
    echo "      That is only correct past imitation.pretrain_steps." >&2
fi

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
    mcts.root_exploration_eps_gate_on_completion=true \
    imitation.enabled=true \
    imitation.levels=match_env \
    imitation.pretrain_steps=50_000 \
    'imitation.mix_ratio_schedule=[[0,0.25],[300000,0.05]]' \
    ++wandb.job_type=specialist-imitation \
    "$@"
