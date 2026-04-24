import numpy as np

from src.muzero.returns import compute_monte_carlo_returns, compute_n_step_returns


def test_monte_carlo_matches_geometric_sum():
    rewards = [1.0, 1.0, 1.0, 1.0]
    g = 0.9
    mc = compute_monte_carlo_returns(rewards, discount=g)
    expected = np.array([
        1 + g + g**2 + g**3,
        1 + g + g**2,
        1 + g,
        1,
    ], dtype=np.float32)
    assert np.allclose(mc, expected, atol=1e-5)


def test_n_step_bootstrap_pads_past_end():
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.6, 0.7]
    g = 1.0
    # n_step > T -> everything past end contributes zero reward and zero bootstrap.
    out = compute_n_step_returns(rewards, values, n_step=10, discount=g)
    assert out.shape == (3,)
    # t=0: 1 + 2 + 3 + 0... = 6
    assert np.isclose(out[0], 6.0, atol=1e-5)


def test_n_step_bootstrap_uses_values():
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
    values = [2.0, 2.0, 2.0, 2.0, 2.0]
    out = compute_n_step_returns(rewards, values, n_step=2, discount=1.0)
    # t=0: r0 + r1 + V2 = 1 + 1 + 2 = 4
    assert np.isclose(out[0], 4.0, atol=1e-5)
    # t=3: r3 + r4 + V5=0 = 1 + 1 + 0 = 2
    assert np.isclose(out[3], 2.0, atol=1e-5)
