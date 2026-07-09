"""Invariants for TrajectoryBuffer that the sampling-speed refactor must keep.

The flat priority cache is an optimization; these tests pin the invariants it
must satisfy regardless of how it is maintained (full rebuild vs incremental):

  * the flat arrays equal a from-scratch reconstruction in id/position order;
  * sampling probabilities stay proportional to priority**alpha;
  * sample() is deterministic under a fixed numpy seed (RNG consumption is not
    allowed to change).
"""
import numpy as np

from src.muzero.buffer import Trajectory, TrajectoryBuffer


def _make_traj(T, A=4, base=1.0):
    return Trajectory(
        obs_stacks=np.zeros((T, 4, 8, 8), dtype=np.uint8),
        actions=np.zeros(T, dtype=np.int64),
        rewards=np.zeros(T, dtype=np.float32),
        policies=np.ones((T, A), dtype=np.float32) / A,
        root_values=np.zeros(T, dtype=np.float32),
        returns=np.zeros(T, dtype=np.float32),
        priorities=np.full(T, base, dtype=np.float32),
    )


def _reference_flat(buf):
    """Recompute the flat cache from scratch in deque order (the ground truth)."""
    prios, tids, poss = [], [], []
    off_by_id = {}
    off = 0
    for tid in buf._ids:
        traj = buf.trajectories[tid]
        L = traj.length
        off_by_id[tid] = off
        prios.append(np.asarray(traj.priorities, dtype=np.float64))
        tids.append(np.full(L, tid, dtype=np.int64))
        poss.append(np.arange(L, dtype=np.int64))
        off += L
    return (
        np.concatenate(prios),
        np.concatenate(tids),
        np.concatenate(poss),
        off_by_id,
    )


def _assert_flat_consistent(buf):
    fp, ft, fpos, off = _reference_flat(buf)
    np.testing.assert_array_equal(buf._flat_priorities, fp)
    np.testing.assert_array_equal(buf._flat_traj_ids, ft)
    np.testing.assert_array_equal(buf._flat_positions, fpos)
    assert buf._offset_by_id == off
    assert buf._max_priority == max(float(fp.max()), buf.eps)


def test_flat_cache_consistent_through_add_evict_update():
    buf = TrajectoryBuffer(capacity_transitions=60, unroll_K=2, num_actions=4)
    buf.add(_make_traj(20, base=1.0))
    buf.add(_make_traj(20, base=2.0))
    _assert_flat_consistent(buf)

    # Trigger eviction of the oldest trajectory.
    buf.add(_make_traj(20, base=3.0))
    buf.add(_make_traj(20, base=4.0))
    _assert_flat_consistent(buf)

    # In-place priority edits must be reflected in the flat cache.
    batch = buf.sample(batch_size=8, train_step=10)
    buf.update_priorities(batch["sample_locations"], np.full(8, 0.5, dtype=np.float32))
    _assert_flat_consistent(buf)


def test_sampling_is_proportional_to_priority():
    buf = TrajectoryBuffer(capacity_transitions=10_000, unroll_K=1, num_actions=4)
    # Set the 3:1 ratio explicitly via the public priority-update path.
    buf.add(_make_traj(100))
    buf.add(_make_traj(100))
    tid0, tid1 = buf._ids[0], buf._ids[1]
    locs = [(tid0, t) for t in range(100)] + [(tid1, t) for t in range(100)]
    prios = np.array([3.0] * 100 + [1.0] * 100, dtype=np.float32)
    buf.update_priorities(locs, prios)
    np.random.seed(123)
    locs = buf.sample(batch_size=20_000, train_step=0)["sample_locations"]
    tid0 = buf._ids[0]
    frac0 = np.mean([tid == tid0 for tid, _ in locs])
    assert abs(frac0 - 0.75) < 0.02


def test_sample_is_deterministic_under_fixed_seed():
    buf = TrajectoryBuffer(capacity_transitions=10_000, unroll_K=2, num_actions=4)
    for b in (1.0, 2.0, 0.5):
        buf.add(_make_traj(50, base=b))

    np.random.seed(42)
    a = buf.sample(batch_size=16, train_step=5)
    np.random.seed(42)
    b = buf.sample(batch_size=16, train_step=5)

    assert a["sample_locations"] == b["sample_locations"]
    np.testing.assert_array_equal(a["obs"], b["obs"])
    np.testing.assert_array_equal(a["is_weights"], b["is_weights"])
