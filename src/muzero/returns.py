"""n-step TD return computation. Ported from Muzero-Hanoi/utils.py."""
import numpy as np


def compute_n_step_returns(rewards, root_values, n_step, discount, terminal=True):
    """Bootstrapped n-step returns.

        z_t = sum_{i=0}^{n-1} gamma^i * r_{t+i} + gamma^n * V(s_{t+n})

    where V(s_{t+n}) is the MCTS root value at the bootstrap step.

    Args:
        rewards: 1D iterable of length T.
        root_values: 1D iterable of length T (MCTS root Q or value at each step).
        n_step: bootstrap horizon.
        discount: gamma.
        terminal: if True, the trajectory ended on a true terminal (rewards
            past T are zero and bootstrap value is zero). If False (truncated
            mid-episode), past-end rewards are zero but the bootstrap value is
            taken from the last available root_value so we don't underestimate
            late returns.
    Returns:
        np.ndarray of shape (T,) with TD targets.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    root_values = np.asarray(root_values, dtype=np.float32)
    T = rewards.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    padded_r = np.zeros(T + n_step, dtype=np.float32)
    padded_r[:T] = rewards
    padded_v = np.empty(T + n_step, dtype=np.float32)
    padded_v[:T] = root_values
    padded_v[T:] = 0.0 if terminal else float(root_values[-1])

    discounts = np.power(discount, np.arange(n_step, dtype=np.float64)).astype(np.float32)
    # Rolling (T, n_step) window of rewards[t..t+n_step-1]; sliding_window_view
    # is a zero-copy view so the (T, n_step) matmul is the only real work.
    windows = np.lib.stride_tricks.sliding_window_view(padded_r, window_shape=n_step)[:T]
    out = windows @ discounts
    out += (discount ** n_step) * padded_v[n_step:T + n_step]
    return out.astype(np.float32)


def compute_monte_carlo_returns(rewards, discount):
    """Full-episode Monte Carlo returns."""
    T = len(rewards)
    out = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + discount * running
        out[t] = running
    return out
