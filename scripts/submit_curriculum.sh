#!/bin/bash
# Full 12-level autocurriculum run with the recipe validated on
# level1-1-diag-v1 (0 -> 0.84 completion) and level1-2-diag-v1 (2026-07):
#
#   - discount 0.999 + completion_bonus 200: the completion signal actually
#     reaches the value function over ~1300-step levels (0.997 discounted it
#     to ~2% at level start — the cause of every earlier flat run).
#   - LR that really decays (cosine to the 1e-4 floor): the always-high LR
#     caused level1-1-diag-v1's late collapse (0.72 -> 0.1); recovery at the
#     floor was immediate. Decay length rescaled to the multi-level
#     train-step count (~3-6M at 2-GPU replay ratios).
#   - 2-GPU split (SBATCH_EXTRA below): learner on cuda:0, inference server
#     auto-detects cuda:1, so the learner stops losing its GPU to self-play
#     inference (~0.035 grad steps/env step on one GPU). 64 workers fit the
#     144-CPU share. inference_server.max_batch=256 covers 64 workers x
#     leaf_batch 4 rows in flight.
#   - Buffer 1M transitions (~37 GB of the 220G share): 300K across 12
#     levels left ~25K/level — recently-neglected levels washed out of the
#     buffer (forgetting risk). min_weight 0.05 keeps mastered levels
#     contributing fresh data.
#   - Temperature schedule rescaled to the longer train-step horizon; a
#     global schedule still fights the curriculum late (new focus levels
#     want exploration mastered ones don't) — revisit per-level temperature
#     if stragglers stall.
#
# env.levels / autocurriculum.enabled come from conf defaults (all 12, on).
# best.pt + best.json track the pooled rolling completion rate.
#
# Usage:
#   bash scripts/submit_curriculum.sh [RUN_NAME] [N_LEGS] [extra overrides...]
# Defaults: curriculum-v1, 4 legs (24h QOS cap each; ~60M env steps at 2-GPU
# pace — surplus legs self-cancel via the TRAINING_COMPLETE sentinel).
#
# Short benchmark first (recommended before the real chain):
#   bash scripts/submit_curriculum.sh bench-2gpu 1 training.total_env_steps=200_000
set -euo pipefail

RUN_NAME="${1:-curriculum-v1}"
N_LEGS="${2:-4}"
shift $(( $# >= 2 ? 2 : $# ))

export SBATCH_EXTRA="--gres=gpu:2 --cpus-per-gpu=72"

bash "$(dirname "$0")/submit_chain.sh" "${RUN_NAME}" "${N_LEGS}" \
    model=muzero_mario_medium \
    muzero.discount=0.999 \
    env.completion_bonus=200 \
    mcts.num_simulations=50 \
    selfplay.num_workers=64 \
    inference_server.max_batch=256 \
    buffer.capacity_transitions=1_000_000 \
    autocurriculum.min_weight=0.05 \
    training.total_env_steps=60_000_000 \
    training.lr_decay_steps=1_500_000 \
    'selfplay.temperature_schedule=[[0,1.0],[500000,0.5],[1500000,0.25],[3000000,0.1]]' \
    ++wandb.job_type=curriculum \
    "$@"
