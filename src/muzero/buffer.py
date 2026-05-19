"""Trajectory-indexed prioritised replay buffer for MuZero-Mario.

Stores full episode trajectories. Pre-stacked observations are kept as uint8
(cheap compared to float32) and converted to float32/[0,1] at sample-assembly
time.

Priority formula (matches Muzero-Hanoi): |v_pred - return|^alpha. Initial
priority when a trajectory is ingested is the maximum priority currently in
the buffer (so new trajectories are likely to be sampled at least once).

Sampling uses a cached flat priority array indexed by (traj_id, t). The cache
is rebuilt only on add/evict; `update_priorities` edits the cache in place.
Trajectories are stored in a dict keyed by stable id so priority updates for
already-evicted trajectories are silently skipped.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np

from src.muzero.targets import build_targets


@dataclass
class Trajectory:
    obs_stacks: np.ndarray   # (T, C, H, W) uint8
    actions: np.ndarray      # (T,) int64
    rewards: np.ndarray      # (T,) float32
    policies: np.ndarray     # (T, A) float32
    root_values: np.ndarray  # (T,) float32
    returns: np.ndarray      # (T,) float32
    priorities: np.ndarray   # (T,) float32
    level: str = ""

    @property
    def length(self) -> int:
        return int(self.obs_stacks.shape[0])


class TrajectoryBuffer:
    def __init__(
        self,
        capacity_transitions: int,
        unroll_K: int,
        num_actions: int,
        priority_alpha: float = 1.0,
        priority_beta_start: float = 0.4,
        priority_beta_end: float = 1.0,
        priority_beta_anneal_steps: int = 500_000,
        eps_priority: float = 1e-3,
    ):
        self.capacity = capacity_transitions
        self.K = unroll_K
        self.num_actions = num_actions
        self.alpha = priority_alpha
        self.beta_start = priority_beta_start
        self.beta_end = priority_beta_end
        self.beta_anneal = priority_beta_anneal_steps
        self.eps = eps_priority

        self.trajectories: Dict[int, Trajectory] = {}
        self._ids: Deque[int] = deque()
        self._size = 0
        self._next_id = 0

        # Flat-index cache: rebuilt on add/evict, updated in-place on priority changes.
        self._flat_priorities: np.ndarray = np.zeros(0, dtype=np.float64)
        self._flat_traj_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self._flat_positions: np.ndarray = np.zeros(0, dtype=np.int64)
        self._offset_by_id: Dict[int, int] = {}
        self._max_priority: float = self.eps

    # -- sizing --------------------------------------------------------------

    def size_transitions(self) -> int:
        return self._size

    def size_trajectories(self) -> int:
        return len(self._ids)

    def __len__(self):
        return self._size

    # -- add / evict ---------------------------------------------------------

    def add(self, traj: Trajectory):
        # New trajectories enter at the current max priority so they are
        # guaranteed to be sampled at least once before competing on TD error.
        if self._ids:
            init_prio = max(self._max_priority, self.eps)
            traj.priorities = np.full_like(traj.priorities, init_prio)

        tid = self._next_id
        self._next_id += 1
        self.trajectories[tid] = traj
        self._ids.append(tid)
        self._size += traj.length

        while self._size > self.capacity and len(self._ids) > 1:
            old_tid = self._ids.popleft()
            evicted = self.trajectories.pop(old_tid)
            self._size -= evicted.length

        self._rebuild_flat()

    # -- flat index cache ----------------------------------------------------

    def _rebuild_flat(self):
        if not self._ids:
            self._flat_priorities = np.zeros(0, dtype=np.float64)
            self._flat_traj_ids = np.zeros(0, dtype=np.int64)
            self._flat_positions = np.zeros(0, dtype=np.int64)
            self._offset_by_id = {}
            self._max_priority = self.eps
            return

        lengths = [self.trajectories[tid].length for tid in self._ids]
        total = sum(lengths)
        priorities = np.empty(total, dtype=np.float64)
        traj_ids = np.empty(total, dtype=np.int64)
        positions = np.empty(total, dtype=np.int64)
        offset_by_id: Dict[int, int] = {}

        off = 0
        for tid, L in zip(self._ids, lengths):
            offset_by_id[tid] = off
            traj = self.trajectories[tid]
            priorities[off:off + L] = traj.priorities
            traj_ids[off:off + L] = tid
            positions[off:off + L] = np.arange(L, dtype=np.int64)
            off += L

        self._flat_priorities = priorities
        self._flat_traj_ids = traj_ids
        self._flat_positions = positions
        self._offset_by_id = offset_by_id
        self._max_priority = max(float(priorities.max()), self.eps)

    # -- sampling ------------------------------------------------------------

    def beta_at(self, train_step: int) -> float:
        if self.beta_anneal <= 0:
            return self.beta_end
        frac = min(1.0, train_step / self.beta_anneal)
        return float(self.beta_start + (self.beta_end - self.beta_start) * frac)

    def sample(self, batch_size: int, train_step: int):
        assert self._size > 0, "empty buffer"
        weights = np.maximum(self._flat_priorities, self.eps)
        if self.alpha != 1.0:
            # x ** 1.0 == x exactly in IEEE754, so the common alpha==1.0 path
            # skips an extra full-length array allocation+copy every step
            # without changing the sampling distribution or RNG draws.
            weights = weights ** self.alpha
        probs = weights / weights.sum()
        chosen = np.random.choice(len(probs), size=batch_size, p=probs, replace=True)

        beta = self.beta_at(train_step)
        is_weights = (1.0 / (len(probs) * probs[chosen])) ** beta
        is_weights = is_weights / is_weights.max()
        is_weights = is_weights.astype(np.float32)

        # Batch-fetch indices to avoid per-sample numpy scalar conversions.
        picked_tids = self._flat_traj_ids[chosen]
        picked_ts = self._flat_positions[chosen]

        obs_batch, act_batch, rew_batch, pol_batch, ret_batch = [], [], [], [], []
        sample_locations: List[Tuple[int, int]] = []
        for i in range(batch_size):
            tid = int(picked_tids[i])
            t = int(picked_ts[i])
            traj = self.trajectories[tid]
            obs, actions, rewards, policies, returns = build_targets(
                traj, t, self.K, self.num_actions
            )
            # obs is uint8 (C, H, W); normalise here on the single fresh copy.
            obs_batch.append(obs.astype(np.float32) / 255.0)
            act_batch.append(actions)
            rew_batch.append(rewards)
            pol_batch.append(policies)
            ret_batch.append(returns)
            sample_locations.append((tid, t))

        return {
            "obs": np.stack(obs_batch, axis=0),          # (B, C, H, W) float32
            "actions": np.stack(act_batch, axis=0),       # (B, K)
            "rewards": np.stack(rew_batch, axis=0),       # (B, K)
            "policies": np.stack(pol_batch, axis=0),      # (B, K+1, A)
            "returns": np.stack(ret_batch, axis=0),       # (B, K+1)
            "is_weights": is_weights,                     # (B,)
            "sample_locations": sample_locations,         # list[(traj_id, t)]
        }

    def update_priorities(self, sample_locations: List[Tuple[int, int]], new_priorities: np.ndarray):
        """Update priorities at sampled locations with fresh |v_pred - target| values.

        Silently skips trajectories that were evicted between sample() and this call.
        """
        new_max = self._max_priority
        for (traj_id, t), p in zip(sample_locations, new_priorities):
            if traj_id not in self.trajectories:
                continue  # evicted
            traj = self.trajectories[traj_id]
            if t >= traj.length:
                continue
            p_val = max(float(p), self.eps)
            traj.priorities[t] = p_val
            offset = self._offset_by_id.get(traj_id)
            if offset is not None:
                self._flat_priorities[offset + t] = p_val
            if p_val > new_max:
                new_max = p_val
        self._max_priority = new_max
