"""MCTS Node (ported from Muzero-Hanoi). h_state is stored as a torch.Tensor of
shape (1, C, H, W); kept on whatever device the MCTS was run on (typically CPU
in workers).
"""
import math

import numpy as np


class Node:
    def __init__(self, prior=0.0, move=None, parent=None):
        self.prior = float(prior)
        self.move = move
        self.parent = parent
        self.is_expanded = False
        self.N = 0
        self.W = 0.0
        self.rwd = 0.0
        self.h_state = None  # torch.Tensor (1, C, H, W)
        self.children = []

    def expand(self, prior, h_state, reward):
        if self.is_expanded:
            raise RuntimeError("Node has already been expanded")
        self.h_state = h_state
        self.rwd = float(reward)
        for action in range(prior.shape[0]):
            self.children.append(Node(prior=prior[action], move=action, parent=self))
        self.is_expanded = True

    def backup(self, value, config, min_max_stats):
        current = self
        value = float(value)
        while current is not None:
            current.W += value
            current.N += 1
            min_max_stats.update(current.rwd + config.discount * current.Q)
            value = current.rwd + config.discount * value
            current = current.parent

    def best_child(self, config, min_max_stats):
        if not self.is_expanded:
            raise ValueError("Expand leaf node first.")
        ucb = self.child_Q(config, min_max_stats) + self.child_U(config)
        a_idx = np.random.choice(np.where(ucb == ucb.max())[0])
        return self.children[a_idx]

    def child_Q(self, config, min_max_stats):
        # Unvisited children fall back to this node's own (normalised) mean
        # value rather than 0 — the bottom of the normalised range. With
        # Mario's mostly-positive shaped rewards a 0 default makes every
        # fresh action look worst-case after a few visits, so exploration
        # would rest entirely on the prior term (EfficientZero uses the same
        # parent-value fallback).
        fallback = min_max_stats.normalize(self.Q)
        Q = np.full(len(self.children), fallback, dtype=np.float32)
        for i, child in enumerate(self.children):
            if child.N > 0:
                Q[i] = min_max_stats.normalize(child.rwd + config.discount * child.Q)
        return Q

    def child_U(self, config):
        U = np.zeros(len(self.children), dtype=np.float32)
        base = math.log((self.N + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
        sqrtN = math.sqrt(max(self.N, 1))
        for i, child in enumerate(self.children):
            U[i] = child.prior * base * sqrtN / (child.N + 1)
        return U

    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

    @property
    def child_N(self):
        return np.array([c.N for c in self.children], dtype=np.int64)
