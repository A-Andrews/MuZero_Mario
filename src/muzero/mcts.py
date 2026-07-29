"""MCTS search for MuZero (adapted from Muzero-Hanoi/MCTS/mcts.py).

Changes vs reference:
- `network.initial_inference(obs)` is expected to return (h_state_tensor,
  policy_logits_tensor, value_scalar_tensor) — we softmax the logits and numpy-ify
  here.
- `network.recurrent_inference(h_batch, action_idx_tensor)` is batched over m
  leaves: h_batch is (m, C, H, W), actions (m,), and it returns
  (h_next (m, C, H, W), reward (m,), policy_logits (m, A), value (m,)).
- hidden state stays as a CPU/GPU torch.Tensor on the device the network lives
  on (no numpy<->tensor ping-pong); through the inference-server adapter it is
  an opaque numpy array instead.
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
        leaf_batch=1,
    ):
        self.discount = discount
        self.num_simulations = num_simulations
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_eps = root_exploration_eps
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.device = device
        # Leaves selected (with virtual visits) and evaluated per network
        # round trip. 1 reproduces fully-sequential search; >1 amortises the
        # inference-server round-trip latency over several simulations.
        self.leaf_batch = max(1, int(leaf_batch))

    @torch.inference_mode()
    def run(self, obs_np, network, temperature=1.0, deterministic=False, stats_out=None):
        """Run MCTS from the current environment observation.

        Args:
            obs_np: np.ndarray of shape (C, H, W) in [0, 1].
            network: a MuZeroNet on `self.device` in eval mode.
            temperature: softmax temperature on visit counts.
            deterministic: if True, bypass Dirichlet noise and argmax visit counts.
            stats_out: optional dict; if given, filled with search diagnostics —
                `prior_entropy` (nats, policy head at the root **before** any
                Dirichlet noise, so it measures the head itself and not the
                exploration on top of it), `visit_entropy` (nats, of the raw
                visit distribution) and `visit_max_frac` (largest single-action
                visit share). Compare entropies against ln(num_actions), which
                is `uniform_entropy` in the same dict.
        Returns:
            (action, pi_prob, root_Q) — action int, numpy visit-count
            distribution (raw, un-tempered — this is the policy training
            target), root mean action-value.
        """
        min_max_stats = MinMaxStats()

        obs = torch.from_numpy(obs_np).to(self.device, dtype=torch.float32).unsqueeze(0)
        h_state, policy_logits, value = network.initial_inference(obs)
        prior = F.softmax(policy_logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        root_value = float(value.squeeze(0).detach().cpu().item())

        # Measured pre-noise on purpose: the open question is whether the policy
        # head is discriminative on its own, and Dirichlet noise would inflate
        # this straight back towards uniform and hide exactly that.
        if stats_out is not None:
            stats_out["prior_entropy"] = _entropy(prior)
            stats_out["uniform_entropy"] = float(np.log(len(prior)))

        if not deterministic and self.root_dirichlet_alpha > 0.0 and self.root_exploration_eps > 0.0:
            prior = _add_dirichlet_noise(prior, self.root_exploration_eps, self.root_dirichlet_alpha)

        root = Node(prior=0.0)
        root.expand(prior, h_state, reward=0.0)
        # Prime the backup path so root contributes a visit and min_max_stats is
        # seeded with a finite range.
        root.backup(root_value, self, min_max_stats)

        sims_done = 0
        while sims_done < self.num_simulations:
            m = min(self.leaf_batch, self.num_simulations - sims_done)

            # Select m leaves; after each selection, pin the path with a
            # virtual visit (N only, no value) so the next selection in this
            # round is steered elsewhere. The same unexpanded leaf can still
            # be reached twice — it is then evaluated once and backed up per
            # selection.
            selections = []
            for _ in range(m):
                node = root
                while node.is_expanded:
                    node = node.best_child(self, min_max_stats)
                selections.append(node)
                cur = node
                while cur is not None:
                    cur.N += 1
                    cur = cur.parent

            unique = []
            batch_index = {}
            for leaf in selections:
                if id(leaf) not in batch_index:
                    batch_index[id(leaf)] = len(unique)
                    unique.append(leaf)

            # One batched recurrent inference for the whole round. Hidden
            # states are opaque here: torch tensors when the network is local,
            # numpy arrays through the inference-server adapter.
            parent_h = [leaf.parent.h_state for leaf in unique]
            if isinstance(parent_h[0], torch.Tensor):
                h_batch = torch.cat(parent_h, dim=0)
            else:
                h_batch = np.concatenate(parent_h, axis=0)
            action_tensor = torch.tensor(
                [leaf.move for leaf in unique], dtype=torch.long, device=self.device
            )
            h_next, reward, policy_logits, value = network.recurrent_inference(h_batch, action_tensor)
            child_prior = F.softmax(policy_logits, dim=-1).detach().cpu().numpy().astype(np.float32)
            reward_np = reward.detach().cpu().numpy()
            value_np = value.detach().cpu().numpy()

            # Roll the virtual visits back, then expand and back up for real.
            for leaf in selections:
                cur = leaf
                while cur is not None:
                    cur.N -= 1
                    cur = cur.parent
            for j, leaf in enumerate(unique):
                leaf.expand(child_prior[j], h_next[j:j + 1], float(reward_np[j]))
            for leaf in selections:
                j = batch_index[id(leaf)]
                leaf.backup(float(value_np[j]), self, min_max_stats)
            sims_done += m

        visits = root.child_N.astype(np.float64)
        # Policy target: the RAW normalised visit distribution (MuZero paper).
        # Temperature is applied only to action *selection* below — feeding the
        # tempered distribution back as the training target over-sharpens the
        # policy head once the schedule drops below 1.0.
        total = visits.sum()
        if total > 0:
            pi_target = visits / total
        else:
            pi_target = np.ones_like(visits) / len(visits)

        if stats_out is not None:
            stats_out["visit_entropy"] = _entropy(pi_target)
            stats_out["visit_max_frac"] = float(pi_target.max())

        if deterministic:
            action_idx = int(np.argmax(visits))
        else:
            pi_select = _visits_to_policy(visits, temperature)
            action_idx = int(np.random.choice(np.arange(len(pi_select)), p=pi_select))

        action = root.children[action_idx].move
        return action, pi_target.astype(np.float32), float(root.Q)


def _entropy(prob):
    """Shannon entropy in nats, with 0·log0 = 0."""
    p = np.asarray(prob, dtype=np.float64)
    nz = p > 0.0
    return float(-(p[nz] * np.log(p[nz])).sum())


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
