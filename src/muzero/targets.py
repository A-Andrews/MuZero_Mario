"""Build K-step unrolled training targets from a trajectory segment.

Given a trajectory of length T and a start index t, produce:
    obs_stack:    (C, H, W) — stacked observation at step t, used as CNN input
    actions:      (K,)  — actions to unroll via dynamics at steps t..t+K-1
    rewards:      (K,)  — reward targets at steps t+1..t+K (reward from the
                           transition into that step)
    policies:     (K+1, A)   — MCTS visit distribution at steps t..t+K
    returns:      (K+1,) — n-step TD targets at steps t..t+K
    next_obs:     (K, C, H, W) uint8 — observations at steps t+1..t+K, used as
                           consistency-loss targets for the unrolled hidden
                           states (EfficientZero-style)
    next_obs_mask:(K,) float32 — 1.0 where t+k is a real step, 0.0 where the
                           unroll ran past the end of the trajectory

Out-of-bounds steps are padded: random actions, reward 0, uniform policy,
return 0. This matches MuZero's absorbing-state convention. Padded next_obs
entries repeat the last real observation and are masked out.
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

    Returns seven numpy arrays:
    (obs, actions, rewards, policies, returns, next_obs, next_obs_mask).
    Observations are returned in their stored dtype (uint8); the caller is
    expected to normalise to float32/[0,1] to avoid a redundant copy.
    """
    T = trajectory.length
    A = num_actions

    obs = trajectory.obs_stacks[t]

    actions = np.zeros(K, dtype=np.int64)
    rewards = np.zeros(K, dtype=np.float32)
    policies = np.zeros((K + 1, A), dtype=np.float32)
    returns = np.zeros(K + 1, dtype=np.float32)
    next_obs = np.empty((K,) + trajectory.obs_stacks.shape[1:], dtype=trajectory.obs_stacks.dtype)
    next_obs_mask = np.zeros(K, dtype=np.float32)

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
        # Consistency target for unroll step i+1 is the observation at t+i+1.
        nxt = t + i + 1
        if nxt < T:
            next_obs[i] = trajectory.obs_stacks[nxt]
            next_obs_mask[i] = 1.0
        else:
            next_obs[i] = trajectory.obs_stacks[T - 1]
            next_obs_mask[i] = 0.0

    return obs, actions, rewards, policies, returns, next_obs, next_obs_mask


def build_reanalyze_targets(trajectory, t, K, n_step, discount):
    """Assemble what the learner needs to recompute fresh n-step value targets
    at sample time (value reanalyze):

        value_target[k] = reward_window[k]
                          + value_obs_factor[k] * V_current(value_obs[k])

    for each unroll position s = t+k, k in 0..K, matching the bootstrap
    semantics of `compute_n_step_returns`:
      * terminal trajectory: bootstrap at s+n_step if it is still inside the
        episode, else no bootstrap (terminal value 0);
      * truncated trajectory: bootstrap at the last *observed* step within the
        horizon (min(n_step, T-1-s) ahead) at its correct discount;
      * absorbing positions (s >= T): target 0 (factor 0, empty window).

    Returns:
        value_obs:        (K+1, C, H, W) uint8 — bootstrap observations
                          (last real obs where unused)
        value_obs_factor: (K+1,) float32 — discount^horizon, 0 if no bootstrap
        reward_window:    (K+1,) float32 — discounted reward sum to the horizon
    """
    T = trajectory.length
    terminal = bool(getattr(trajectory, "terminal", True))

    value_obs = np.empty((K + 1,) + trajectory.obs_stacks.shape[1:], dtype=trajectory.obs_stacks.dtype)
    value_obs_factor = np.zeros(K + 1, dtype=np.float32)
    reward_window = np.zeros(K + 1, dtype=np.float32)

    for i in range(K + 1):
        s = t + i
        if s >= T:
            value_obs[i] = trajectory.obs_stacks[T - 1]
            continue
        horizon = min(n_step, T - s) if terminal else min(n_step, T - 1 - s)
        acc = 0.0
        g = 1.0
        for j in range(horizon):
            acc += g * trajectory.rewards[s + j]
            g *= discount
        reward_window[i] = acc
        if terminal:
            if s + n_step < T:
                value_obs[i] = trajectory.obs_stacks[s + n_step]
                value_obs_factor[i] = discount ** n_step
            else:
                value_obs[i] = trajectory.obs_stacks[T - 1]
        else:
            value_obs[i] = trajectory.obs_stacks[s + horizon]
            value_obs_factor[i] = discount ** horizon

    return value_obs, value_obs_factor, reward_window
