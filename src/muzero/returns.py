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

    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if terminal:
            # True terminal: rewards past T-1 are 0 and the bootstrap value is
            # 0 once the n-step window runs off the end of the episode.
            horizon = min(n_step, T - t)
            acc = 0.0
            g = 1.0
            for i in range(horizon):
                acc += g * rewards[t + i]
                g *= discount
            if t + n_step < T:
                acc += (discount ** n_step) * root_values[t + n_step]
            out[t] = acc
        else:
            # Truncated mid-episode: bootstrap off the last *observed* root
            # value at its correct discount horizon, rather than zero-padding
            # the reward gap and bootstrapping a far-future value.
            k = min(n_step, T - 1 - t)
            acc = 0.0
            g = 1.0
            for i in range(k):
                acc += g * rewards[t + i]
                g *= discount
            acc += (discount ** k) * root_values[t + k]
            out[t] = acc
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
