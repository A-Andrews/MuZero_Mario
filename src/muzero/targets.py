"""Build K-step unrolled training targets from a trajectory segment.

Given a trajectory of length T and a start index t, produce:
    obs_stack:    (C, H, W) — stacked observation at step t, used as CNN input
    actions:      (K,)  — actions to unroll via dynamics at steps t..t+K-1
    rewards:      (K,)  — reward targets at steps t+1..t+K (reward from the
                           transition into that step)
    policies:     (K+1, A)   — MCTS visit distribution at steps t..t+K
    returns:      (K+1,) — n-step TD targets at steps t..t+K

Out-of-bounds steps are padded: random actions, reward 0, uniform policy,
return 0. This matches MuZero's absorbing-state convention.
"""
from __future__ import annotations

import numpy as np


def build_targets(trajectory, t, K, num_actions):
    """Assemble targets for the segment starting at index `t`.

    Trajectory must expose:
        .obs_stacks[t]    -> (C, H, W) uint8 (pre-stacked)
        .actions[t]       -> int
        .rewards[t]       -> float  (reward received entering step t+1)
        .policies[t]      -> (A,) float32 visit distribution
        .returns[t]       -> float32 TD target at step t
        .length           -> int T

    Returns five numpy arrays: (obs, actions, rewards, policies, returns).
    The obs is returned in its stored dtype (uint8); the caller is expected to
    normalise to float32/[0,1] to avoid a redundant copy.
    """
    T = trajectory.length
    A = num_actions

    obs = trajectory.obs_stacks[t]

    actions = np.zeros(K, dtype=np.int64)
    rewards = np.zeros(K, dtype=np.float32)
    policies = np.zeros((K + 1, A), dtype=np.float32)
    returns = np.zeros(K + 1, dtype=np.float32)

    for i in range(K + 1):
        idx = t + i
        if idx < T:
            policies[i] = trajectory.policies[idx]
            returns[i] = trajectory.returns[idx]
        else:
            policies[i] = 1.0 / A  # uniform at absorbing state
            returns[i] = 0.0

    for i in range(K):
        idx = t + i
        if idx < T:
            actions[i] = trajectory.actions[idx]
            rewards[i] = trajectory.rewards[idx]
        else:
            actions[i] = np.random.randint(0, A)
            rewards[i] = 0.0

    return obs, actions, rewards, policies, returns
