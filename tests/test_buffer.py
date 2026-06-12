import numpy as np

from src.muzero.buffer import Trajectory, TrajectoryBuffer


def _make_traj(T=10, A=4, priority=1.0):
    return Trajectory(
        obs_stacks=np.zeros((T, 4, 8, 8), dtype=np.uint8),
        actions=np.zeros(T, dtype=np.int64),
        rewards=np.zeros(T, dtype=np.float32),
        policies=np.ones((T, A), dtype=np.float32) / A,
        root_values=np.zeros(T, dtype=np.float32),
        returns=np.zeros(T, dtype=np.float32),
        priorities=np.full(T, priority, dtype=np.float32),
    )


def test_eviction_respects_capacity():
    buf = TrajectoryBuffer(capacity_transitions=25, unroll_K=2, num_actions=4)
    for _ in range(5):
        buf.add(_make_traj(T=10))
    assert buf.size_transitions() <= 25 + 10  # at most one traj beyond capacity


def test_sample_returns_expected_shapes():
    buf = TrajectoryBuffer(capacity_transitions=500, unroll_K=3, num_actions=4)
    for _ in range(5):
        buf.add(_make_traj(T=8))
    batch = buf.sample(batch_size=7, train_step=0)
    assert batch["obs"].shape == (7, 4, 8, 8)
    assert batch["actions"].shape == (7, 3)
    assert batch["rewards"].shape == (7, 3)
    assert batch["policies"].shape == (7, 4, 4)
    assert batch["returns"].shape == (7, 4)
    assert batch["next_obs"].shape == (7, 3, 4, 8, 8)
    assert batch["next_obs"].dtype == np.uint8
    assert batch["next_obs_mask"].shape == (7, 3)
    assert batch["is_weights"].shape == (7,)


def test_priority_update_after_eviction_is_safe():
    buf = TrajectoryBuffer(capacity_transitions=50, unroll_K=2, num_actions=4)
    for _ in range(3):
        buf.add(_make_traj(T=20))
    batch = buf.sample(batch_size=4, train_step=0)
    # Force-evict the oldest trajectory
    for _ in range(5):
        buf.add(_make_traj(T=20))
    # Must not raise even though some sampled locations refer to evicted ids.
    buf.update_priorities(batch["sample_locations"], np.ones(4, dtype=np.float32))
