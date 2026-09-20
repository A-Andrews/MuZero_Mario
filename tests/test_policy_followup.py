"""Scientific accounting and gradient-isolation contracts for follow-up probes."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from scripts.eval_controller_diagnostic import select_plan, summarize_records
from scripts.diagnose_policy_gradients import choose_roots, gradient_pair_stats, policy_losses
from scripts.plot_action_distributions import action_counts, distribution_summary
from tests.test_controller_diagnostic import manifest


def confirmation():
    data = manifest()
    data.update(split="confirmation", episodes=100, development_trials=data["trials"],
                trials=data["confirmation_trials"])
    data["levels"] = {k: v for k, v in data["levels"].items() if k in ("Level1-1", "Level6-1")}
    data["conditions"] = [c for c in data["conditions"] if c["name"] in ("greedy", "sampled")]
    return data


def test_confirmation_uses_all_reserved_pairs_without_development_or_smoke():
    data = confirmation()
    plan = select_plan(data)
    assert len(plan) == 200
    assert len({t["env_seed"] for _, t in plan}) == 100
    assert summarize_records(plan, [], split="confirmation")["split"] == "confirmation"
    with pytest.raises(ValueError, match="smoke"):
        select_plan(data, smoke=True)
    altered = copy.deepcopy(data)
    altered["trials"] = copy.deepcopy(altered["trials"])
    altered["trials"][0]["env_seed"] = 42
    with pytest.raises(ValueError, match="unchanged reserved"):
        select_plan(altered)
    overlap = copy.deepcopy(data)
    overlap["development_trials"][0]["env_seed"] = overlap["trials"][0]["env_seed"]
    with pytest.raises(ValueError, match="overlap"):
        select_plan(overlap)


def test_action_aggregation_weights_units_not_long_episodes():
    first, second = action_counts(np.zeros(1000, dtype=int)), action_counts(np.ones(10, dtype=int))
    report = distribution_summary([first, second])
    np.testing.assert_allclose(report["mean"][:2], [.5, .5])
    assert report["pooled_decision_distribution"][0] > .99
    assert report["n_units"] == 2 and report["n_decisions"] == 1010
    with pytest.raises(ValueError):
        action_counts(np.array([12]))


def test_roots_cover_time_and_exclude_incomplete_unrolls():
    steps = np.r_[np.arange(0, 800, 20), np.arange(800, 1000)]
    roots = choose_roots(steps, 1000, 5, 16)
    assert len(roots) == 16
    assert steps[roots[-1]] == 994
    assert sum(steps[roots] < 800) >= 12
    assert len(choose_roots(np.arange(3), 3, 5, 10)) == 0


def test_gradient_metrics_distinguish_size_and_opposition():
    result = gradient_pair_stats([torch.tensor([1., 0.])], [torch.tensor([-2., 0.])])
    assert result["cosine"] == -1
    assert result["recurrent_to_root_norm_ratio"] == 2
    assert result["combined_cosine_with_root"] == -1
    absent = gradient_pair_stats([torch.zeros(2)], [torch.ones(2)])
    assert absent["cosine"] is None and absent["root_norm"] == 0


class TinyUnroll(nn.Module):
    def __init__(self):
        super().__init__()
        self.rep = nn.Linear(2, 3)
        self.dyn = nn.Linear(3, 3)
        self.head = nn.Linear(3, 2)

    def initial_step(self, obs):
        h = self.rep(obs)
        return h, self.head(h), None

    def recurrent_step(self, h, action):
        h = torch.tanh(self.dyn(h) + action[:, None]*.1)
        return h, None, self.head(h), None


def test_hidden_hook_changes_upstream_gradients_but_not_direct_head_gradients():
    torch.manual_seed(31)
    net = TinyUnroll()
    obs = torch.randn(4, 2)
    actions = torch.zeros((4, 5), dtype=torch.long)
    targets = torch.tensor([.8, .2]).expand(4, 6, 2)
    results = []
    for scale in (.5, 1.):
        root, recurrent, individual = policy_losses(net, obs, actions, targets, scale)
        torch.testing.assert_close(recurrent, sum(individual)/5)
        head, dyn = torch.autograd.grad(root + recurrent, (net.head.weight, net.dyn.weight))
        results.append((head, dyn))
    torch.testing.assert_close(results[0][0], results[1][0])
    assert not torch.allclose(results[0][1], results[1][1])
