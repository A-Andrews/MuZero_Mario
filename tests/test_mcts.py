import numpy as np
import torch

from src.muzero.mcts import MCTS
from src.muzero.node import Node
from src.muzero.utils_mcts import MinMaxStats


class _StubNet:
    """Mock MuZero net that returns constant policy/value/reward for MCTS unit tests."""

    num_actions = 4

    def __init__(self, policy=None, value=0.0, reward=0.1):
        self.policy = policy if policy is not None else np.ones(4) / 4
        self.value = value
        self.reward = reward

    def eval(self):
        return self

    def initial_inference(self, obs):
        h = torch.zeros(1, 4, 2, 2)
        policy_logits = torch.log(torch.tensor(self.policy, dtype=torch.float32) + 1e-9).unsqueeze(0)
        v = torch.tensor([self.value], dtype=torch.float32)
        return h, policy_logits, v

    def recurrent_inference(self, h, a):
        h_next = h + 0.01
        policy_logits = torch.log(torch.tensor(self.policy, dtype=torch.float32) + 1e-9).unsqueeze(0)
        r = torch.tensor([self.reward], dtype=torch.float32)
        v = torch.tensor([self.value], dtype=torch.float32)
        return h_next, r, policy_logits, v


def test_mcts_runs_and_returns_valid_policy():
    mcts = MCTS(discount=0.99, num_simulations=16, root_dirichlet_alpha=0.0, device="cpu")
    net = _StubNet(policy=np.array([0.1, 0.7, 0.1, 0.1]))
    obs = np.random.rand(4, 8, 8).astype(np.float32)
    action, pi, q = mcts.run(obs, net, temperature=1.0, deterministic=False)
    assert 0 <= action < 4
    assert pi.shape == (4,)
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)
    assert pi.min() >= 0


def test_mcts_deterministic_argmax():
    mcts = MCTS(discount=0.99, num_simulations=32, root_dirichlet_alpha=0.0, device="cpu")
    # Heavy prior on action 2 -> most visits at action 2.
    net = _StubNet(policy=np.array([0.01, 0.01, 0.97, 0.01]))
    obs = np.random.rand(4, 8, 8).astype(np.float32)
    action, _, _ = mcts.run(obs, net, temperature=0.0, deterministic=True)
    assert action == 2


def test_minmax_stats_normalises():
    stats = MinMaxStats()
    stats.update(1.0)
    stats.update(5.0)
    assert stats.normalize(3.0) == 0.5
