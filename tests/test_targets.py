import numpy as np

from src.muzero.buffer import Trajectory
from src.muzero.returns import compute_n_step_returns
from src.muzero.targets import build_reanalyze_targets, build_targets


def _make_traj(T=5, C=4, H=8, W=8, A=4, terminal=True):
    obs = np.random.randint(0, 255, size=(T, C, H, W), dtype=np.uint8)
    actions = np.arange(T, dtype=np.int64) % A
    rewards = np.arange(T, dtype=np.float32) * 0.1
    policies = np.ones((T, A), dtype=np.float32) / A
    root_v = np.arange(T, dtype=np.float32) * 0.2
    returns = np.arange(T, dtype=np.float32) * 0.3
    priorities = np.ones(T, dtype=np.float32)
    return Trajectory(
        obs_stacks=obs,
        actions=actions,
        rewards=rewards,
        policies=policies,
        root_values=root_v,
        returns=returns,
        priorities=priorities,
        terminal=terminal,
    )


def test_build_targets_within_bounds():
    traj = _make_traj(T=10)
    obs, actions, rewards, policies, returns, next_obs, next_obs_mask = build_targets(
        traj, t=2, K=5, num_actions=4
    )
    assert obs.shape == traj.obs_stacks.shape[1:]
    assert actions.shape == (5,)
    assert rewards.shape == (5,)
    assert policies.shape == (6, 4)
    assert returns.shape == (6,)
    assert np.allclose(actions, traj.actions[2:7])
    assert np.allclose(rewards, traj.rewards[2:7])
    assert np.allclose(returns, traj.returns[2:8])
    # Consistency targets: obs at t+1..t+K, all real -> mask of ones.
    assert next_obs.shape == (5,) + traj.obs_stacks.shape[1:]
    assert next_obs.dtype == traj.obs_stacks.dtype
    assert np.array_equal(next_obs, traj.obs_stacks[3:8])
    assert np.all(next_obs_mask == 1.0)


def test_build_targets_pads_past_end():
    traj = _make_traj(T=4)
    obs, actions, rewards, policies, returns, next_obs, next_obs_mask = build_targets(
        traj, t=3, K=5, num_actions=4
    )
    # Only step 3 is real; rest is padded.
    assert rewards[0] == traj.rewards[3]
    assert np.all(rewards[1:] == 0.0)
    assert returns[0] == traj.returns[3]
    assert np.all(returns[1:] == 0.0)
    assert np.allclose(policies[1:].sum(axis=-1), 1.0)
    # t=3 is the last step: every next-obs (t+1..) is past the end.
    assert np.all(next_obs_mask == 0.0)
    # Padded entries repeat the final real observation.
    assert np.array_equal(next_obs[0], traj.obs_stacks[3])


def test_build_targets_partial_next_obs_mask():
    traj = _make_traj(T=6)
    _, _, _, _, _, next_obs, next_obs_mask = build_targets(traj, t=2, K=5, num_actions=4)
    # t+k for k=1..5 -> steps 3,4,5 real; 6,7 padded.
    assert np.array_equal(next_obs_mask, np.array([1, 1, 1, 0, 0], dtype=np.float32))
    assert np.array_equal(next_obs[0], traj.obs_stacks[3])
    assert np.array_equal(next_obs[2], traj.obs_stacks[5])
    assert np.array_equal(next_obs[3], traj.obs_stacks[5])


def test_reanalyze_targets_match_nstep_returns_terminal():
    """reward_window + factor * V(bootstrap obs) must reproduce
    compute_n_step_returns when V(.) is taken to be the stored root value at
    the bootstrap index — i.e. reanalyze changes *where the value comes from*,
    never the n-step bootstrap semantics."""
    T, n, K, disc = 12, 3, 4, 0.9
    traj = _make_traj(T=T, terminal=True)
    expected = compute_n_step_returns(
        traj.rewards, traj.root_values, n_step=n, discount=disc, terminal=True
    )
    for t in (0, 5, 9):
        v_obs, v_fac, r_win = build_reanalyze_targets(traj, t, K, n, disc)
        for k in range(K + 1):
            s = t + k
            if s >= T:
                assert v_fac[k] == 0.0 and r_win[k] == 0.0
                continue
            if s + n < T:
                boot = traj.root_values[s + n]
                assert np.array_equal(v_obs[k], traj.obs_stacks[s + n])
            else:
                boot = 0.0
                assert v_fac[k] == 0.0  # terminal: no bootstrap past the end
            np.testing.assert_allclose(
                r_win[k] + v_fac[k] * boot, expected[s], rtol=1e-5
            )


def test_reanalyze_targets_match_nstep_returns_truncated():
    T, n, K, disc = 10, 4, 3, 0.95
    traj = _make_traj(T=T, terminal=False)
    expected = compute_n_step_returns(
        traj.rewards, traj.root_values, n_step=n, discount=disc, terminal=False
    )
    for t in (0, 4, 8):
        v_obs, v_fac, r_win = build_reanalyze_targets(traj, t, K, n, disc)
        for k in range(K + 1):
            s = t + k
            if s >= T:
                assert v_fac[k] == 0.0 and r_win[k] == 0.0
                continue
            # Truncated: bootstrap from the last observed step in the horizon.
            idx = s + min(n, T - 1 - s)
            assert np.array_equal(v_obs[k], traj.obs_stacks[idx])
            np.testing.assert_allclose(
                r_win[k] + v_fac[k] * traj.root_values[idx], expected[s], rtol=1e-5
            )
