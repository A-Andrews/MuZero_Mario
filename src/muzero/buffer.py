"""Trajectory-indexed prioritised replay buffer for MuZero-Mario.

Stores full episode trajectories. Pre-stacked observations are kept as uint8
(cheap compared to float32) and converted to float32/[0,1] at sample-assembly
time.

Priority formula (matches Muzero-Hanoi): |v_pred - return|^alpha. Ingested
trajectories keep the per-step priorities computed by the self-play worker
(|n-step return - MCTS root value|), floored at eps.

Sampling uses a cached flat priority array indexed by (traj_id, t). The cache
is rebuilt only on add/evict; `update_priorities` edits the cache in place.
Trajectories are stored in a dict keyed by stable id so priority updates for
already-evicted trajectories are silently skipped.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

from src.muzero.schedules import validate_schedule, value_at

import numpy as np

from src.muzero.targets import build_reanalyze_targets, build_targets


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
    # True when the episode ended on a real terminal (death / completion),
    # False when it was truncated (max_trajectory_length). Decides the
    # bootstrap semantics of reanalyzed value targets.
    terminal: bool = True

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
        reanalyze: bool = False,
        n_step: int = 0,
        discount: float = 1.0,
    ):
        self.capacity = capacity_transitions
        self.K = unroll_K
        self.num_actions = num_actions
        self.alpha = priority_alpha
        self.beta_start = priority_beta_start
        self.beta_end = priority_beta_end
        self.beta_anneal = priority_beta_anneal_steps
        self.eps = eps_priority
        # When True, sample() additionally emits the bootstrap observations /
        # discount factors / reward windows the learner needs to recompute
        # n-step value targets with a fresh network (value reanalyze).
        self.reanalyze = bool(reanalyze)
        self.n_step = int(n_step)
        self.discount = float(discount)

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

    def add(self, traj: Trajectory, rebuild: bool = True):
        # Keep the worker-computed per-step priorities (|n-step return − MCTS
        # root value|, as in the MuZero paper) so the first pass over a new
        # trajectory targets the actually-surprising steps; only floor them at
        # eps so every step stays sampleable.
        traj.priorities = np.maximum(traj.priorities, self.eps)

        tid = self._next_id
        self._next_id += 1
        self.trajectories[tid] = traj
        self._ids.append(tid)
        self._size += traj.length

        while self._size > self.capacity and len(self._ids) > 1:
            old_tid = self._ids.popleft()
            evicted = self.trajectories.pop(old_tid)
            self._size -= evicted.length

        # rebuild=False lets bulk loaders (human_data.load_human_buffer) add
        # thousands of trajectories and rebuild the flat cache once at the end
        # instead of O(N^2) per-add rebuilds.
        if rebuild:
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
        nobs_batch, nmask_batch = [], []
        vobs_batch, vfac_batch, rwin_batch = [], [], []
        sample_locations: List[Tuple[int, int]] = []
        for i in range(batch_size):
            tid = int(picked_tids[i])
            t = int(picked_ts[i])
            traj = self.trajectories[tid]
            obs, actions, rewards, policies, returns, next_obs, next_obs_mask = build_targets(
                traj, t, self.K, self.num_actions
            )
            # obs is uint8 (C, H, W); normalise here on the single fresh copy.
            obs_batch.append(obs.astype(np.float32) / 255.0)
            act_batch.append(actions)
            rew_batch.append(rewards)
            pol_batch.append(policies)
            ret_batch.append(returns)
            # next_obs stays uint8 — it is K× the size of obs, so it ships to
            # the GPU compact and is normalised there.
            nobs_batch.append(next_obs)
            nmask_batch.append(next_obs_mask)
            if self.reanalyze:
                v_obs, v_fac, r_win = build_reanalyze_targets(
                    traj, t, self.K, self.n_step, self.discount
                )
                vobs_batch.append(v_obs)
                vfac_batch.append(v_fac)
                rwin_batch.append(r_win)
            sample_locations.append((tid, t))

        out = {
            "obs": np.stack(obs_batch, axis=0),          # (B, C, H, W) float32
            "actions": np.stack(act_batch, axis=0),       # (B, K)
            "rewards": np.stack(rew_batch, axis=0),       # (B, K)
            "policies": np.stack(pol_batch, axis=0),      # (B, K+1, A)
            "returns": np.stack(ret_batch, axis=0),       # (B, K+1)
            "next_obs": np.stack(nobs_batch, axis=0),     # (B, K, C, H, W) uint8
            "next_obs_mask": np.stack(nmask_batch, axis=0),  # (B, K) float32
            "is_weights": is_weights,                     # (B,)
            "sample_locations": sample_locations,         # list[(traj_id, t)]
        }
        if self.reanalyze:
            # uint8 like next_obs — normalised on the GPU by the learner.
            out["value_obs"] = np.stack(vobs_batch, axis=0)          # (B, K+1, C, H, W)
            out["value_obs_factor"] = np.stack(vfac_batch, axis=0)   # (B, K+1)
            out["reward_window"] = np.stack(rwin_batch, axis=0)      # (B, K+1)
        return out

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


class MixedBuffer:
    """Sample-time mix of the self-play buffer and a pinned human buffer.

    Presents the exact TrajectoryBuffer surface the learner and its batch
    prefetcher use. Each sample() draws ceil(batch_size * mix_ratio) steps from
    the human buffer and the rest from the main buffer, concatenating every
    array key. `sample_locations` entries become ("human"|"main", traj_id, t)
    triples; the learner passes them back opaquely and update_priorities routes
    each to its source buffer — so prioritized replay applies to the human data
    too (after the first pass, human priorities become |v_pred - target| like
    everything else).

    Sizing queries report the *main* buffer only: `min_replay_transitions`
    keeps meaning "self-play data collected", and human transitions are
    reported separately via human_transitions().

    `mix_schedule` optionally anneals the ratio over training: a list of
    (train_step, ratio) points, linearly interpolated and clamped to the end
    values outside the range. It is evaluated against the `train_step` passed
    to sample(), i.e. the learner's *global* training step — the caller is
    responsible for shifting thresholds past any pretrain phase.
    `force_mix_ratio` overrides both the constant and the schedule (pretrain
    uses it to pin 100% human batches).
    """

    def __init__(
        self,
        main: TrajectoryBuffer,
        human: TrajectoryBuffer,
        mix_ratio: float,
        mix_schedule=None,
    ):
        self.main = main
        self.human = human
        self.set_mix_ratio(mix_ratio)
        self.mix_schedule = self._validate_schedule(mix_schedule)
        self._forced_ratio: float | None = None

    def set_mix_ratio(self, mix_ratio: float):
        assert 0.0 <= mix_ratio <= 1.0, f"mix_ratio must be in [0,1], got {mix_ratio}"
        self.mix_ratio = float(mix_ratio)

    @staticmethod
    def _validate_schedule(schedule):
        return validate_schedule(schedule, lo=0.0, hi=1.0)

    def force_mix_ratio(self, ratio: float | None):
        """Pin the effective ratio regardless of constant/schedule (None clears)."""
        self._forced_ratio = None if ratio is None else float(ratio)

    def mix_ratio_at(self, train_step: int) -> float:
        """Effective human fraction at this training step."""
        if self._forced_ratio is not None:
            return self._forced_ratio
        if not self.mix_schedule:
            return self.mix_ratio
        return value_at(train_step, self.mix_schedule)

    # -- ingestion / sizing (self-play side) -----------------------------------

    def add(self, traj: Trajectory, rebuild: bool = True):
        self.main.add(traj, rebuild=rebuild)

    def size_transitions(self) -> int:
        return self.main.size_transitions()

    def size_trajectories(self) -> int:
        return self.main.size_trajectories()

    def human_transitions(self) -> int:
        return self.human.size_transitions()

    def __len__(self):
        return self.main.size_transitions()

    # -- sampling ---------------------------------------------------------------

    def sample(self, batch_size: int, train_step: int):
        ratio = self.mix_ratio_at(train_step)
        n_human = min(batch_size, int(np.ceil(batch_size * ratio)))
        if self.main.size_transitions() == 0:
            n_human = batch_size  # pretrain: main buffer still empty
        if self.human.size_transitions() == 0:
            n_human = 0

        parts = []
        if n_human > 0:
            parts.append(("human", self.human.sample(n_human, train_step)))
        if batch_size - n_human > 0:
            parts.append(("main", self.main.sample(batch_size - n_human, train_step)))

        if len(parts) == 1:
            source, batch = parts[0]
            batch["sample_locations"] = [
                (source, tid, t) for tid, t in batch["sample_locations"]
            ]
            return batch

        out = {}
        for key in parts[0][1]:
            if key == "sample_locations":
                out[key] = [
                    (source, tid, t)
                    for source, batch in parts
                    for tid, t in batch["sample_locations"]
                ]
            else:
                out[key] = np.concatenate([batch[key] for _, batch in parts], axis=0)
        return out

    def update_priorities(self, sample_locations, new_priorities: np.ndarray):
        by_source: Dict[str, Tuple[List[Tuple[int, int]], List[float]]] = {}
        for (source, tid, t), p in zip(sample_locations, new_priorities):
            locs, ps = by_source.setdefault(source, ([], []))
            locs.append((tid, t))
            ps.append(p)
        for source, (locs, ps) in by_source.items():
            buf = self.human if source == "human" else self.main
            buf.update_priorities(locs, np.asarray(ps))
