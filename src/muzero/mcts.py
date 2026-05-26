"""MCTS search for MuZero (adapted from Muzero-Hanoi/MCTS/mcts.py).

Changes vs reference:
- `network.initial_inference(obs)` is expected to return (h_state_tensor,
  policy_logits_tensor, value_scalar_tensor) — we softmax the logits and numpy-ify
  here.
- `network.recurrent_inference(h_state_tensor, action_idx_tensor)` returns
  (h_next_tensor, reward_scalar_tensor, policy_logits_tensor, value_scalar_tensor).
- hidden state stays as a CPU/GPU torch.Tensor on the device the network lives
  on (no numpy<->tensor ping-pong).
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.muzero.node import Node
from src.muzero.utils_mcts import MinMaxStats


class MCTS:
    def __init__(
        self,
        discount,
        num_simulations,
        root_dirichlet_alpha=0.25,
        root_exploration_eps=0.25,
        pb_c_base=19652,
        pb_c_init=1.25,
        device="cpu",
    ):
        self.discount = discount
        self.num_simulations = num_simulations
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_eps = root_exploration_eps
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.device = device

    @torch.inference_mode()
    def run(self, obs_np, network, temperature=1.0, deterministic=False):
        """Run MCTS from the current environment observation.

        Args:
            obs_np: np.ndarray of shape (C, H, W) in [0, 1].
            network: a MuZeroNet on `self.device` in eval mode.
            temperature: softmax temperature on visit counts.
            deterministic: if True, bypass Dirichlet noise and argmax visit counts.
        Returns:
            (action, pi_prob, root_Q) — action int, numpy visit-count distribution,
            root mean action-value.
        """
        min_max_stats = MinMaxStats()

        obs = torch.from_numpy(obs_np).to(self.device, dtype=torch.float32).unsqueeze(0)
        h_state, policy_logits, value = network.initial_inference(obs)
        prior = F.softmax(policy_logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        root_value = float(value.squeeze(0).detach().cpu().item())

        if not deterministic and self.root_dirichlet_alpha > 0.0 and self.root_exploration_eps > 0.0:
            prior = _add_dirichlet_noise(prior, self.root_exploration_eps, self.root_dirichlet_alpha)

        root = Node(prior=0.0)
        root.expand(prior, h_state, reward=0.0)
        # Prime the backup path so root contributes a visit and min_max_stats is
        # seeded with a finite range.
        root.backup(root_value, self, min_max_stats)

        for _ in range(self.num_simulations):
            node = root
            while node.is_expanded:
                node = node.best_child(self, min_max_stats)

            parent_h = node.parent.h_state
            action_tensor = torch.tensor([node.move], dtype=torch.long, device=self.device)
            h_next, reward, policy_logits, value = network.recurrent_inference(parent_h, action_tensor)
            child_prior = F.softmax(policy_logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
            reward_val = float(reward.squeeze(0).detach().cpu().item())
            value_val = float(value.squeeze(0).detach().cpu().item())

            node.expand(child_prior, h_next, reward_val)
            node.backup(value_val, self, min_max_stats)

        visits = root.child_N.astype(np.float64)
        pi_prob = _visits_to_policy(visits, temperature)

        if deterministic:
            action_idx = int(np.argmax(visits))
        else:
            action_idx = int(np.random.choice(np.arange(len(pi_prob)), p=pi_prob))

        action = root.children[action_idx].move
        return action, pi_prob.astype(np.float32), float(root.Q)


def _add_dirichlet_noise(prob, eps, alpha):
    alphas = np.ones_like(prob) * alpha
    noise = np.random.dirichlet(alphas).astype(np.float32)
    return (1.0 - eps) * prob + eps * noise


def _visits_to_policy(visits, temperature):
    if temperature <= 0.0:
        out = np.zeros_like(visits)
        out[np.argmax(visits)] = 1.0
        return out
    powered = np.power(visits, 1.0 / temperature)
    total = powered.sum()
    if total <= 0:
        return np.ones_like(powered) / len(powered)
    return powered / total
