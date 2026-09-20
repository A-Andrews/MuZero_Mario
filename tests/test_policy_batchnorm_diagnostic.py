"""Behavioral contracts for the isolated BatchNorm policy probe."""
import numpy as np
import pytest
import torch
from torch import nn

from scripts.diagnose_policy_batchnorm import evaluate_mode, load_banks, metric_vectors, probe_condition, summarize


class _Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(8, 3)

    def forward(self, hidden):
        logits = self.fc(torch.relu(self.bn(hidden)).flatten(1))
        return logits, logits[:, :1]


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.representation = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
        self.prediction = _Prediction()

    def initial_step(self, obs):
        hidden = self.representation(obs)
        logits, value = self.prediction(hidden)
        return hidden, logits, value


def _case():
    torch.manual_seed(3)
    net = _TinyNet().eval()
    # Make the evaluation/batch-stat distinction explicit and reproducible.
    net.prediction.bn.running_mean.fill_(0.7)
    obs = np.random.default_rng(5).integers(0, 256, (12, 1, 2, 2), dtype=np.uint8)
    return net, obs


def test_source_and_copy_do_not_change_and_only_selected_bn_uses_batch_stats():
    net, obs = _case()
    net.representation[0].train()  # Mixed source flags must survive untouched.
    before = {key: value.clone() for key, value in net.state_dict().items()}
    flags = {key: module.training for key, module in net.named_modules()}
    eval_priors, _ = evaluate_mode(net, obs, "cpu", "eval", 4)
    batch_priors, checks = evaluate_mode(net, obs, "cpu", "prediction_batch_stats", 4)
    assert checks["batch_stat_modules"] == ["prediction.bn"]
    assert checks["parameters_and_buffers_unchanged"]
    assert checks["original_flags_unchanged"]
    assert not np.allclose(eval_priors, batch_priors)
    assert all(torch.equal(value, before[key]) for key, value in net.state_dict().items())
    assert flags == {key: module.training for key, module in net.named_modules()}
    assert net.prediction.bn.track_running_stats
    _, all_checks = evaluate_mode(net, obs, "cpu", "all_batch_stats", 4)
    assert set(all_checks["batch_stat_modules"]) == {"representation.1", "prediction.bn"}


def test_eval_regrouping_preserves_state_order_and_batch_arm_is_reproducible():
    net, obs = _case()
    order = np.random.default_rng(10).permutation(len(obs))
    reference, _ = evaluate_mode(net, obs, "cpu", "eval", 4)
    shuffled, _ = evaluate_mode(net, obs, "cpu", "eval", 4, order)
    np.testing.assert_allclose(reference, shuffled, atol=1e-7)
    first, _ = evaluate_mode(net, obs, "cpu", "prediction_batch_stats", 4, order)
    second, _ = evaluate_mode(net, obs, "cpu", "prediction_batch_stats", 4, order)
    original_groups, _ = evaluate_mode(net, obs, "cpu", "prediction_batch_stats", 4)
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, original_groups)


def test_metric_direction_and_episode_uncertainty_use_episodes_not_states():
    reference = np.array([[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]])
    candidate = np.array([[0.25, 0.75], [0.25, 0.75], [0.75, 0.25]])
    vectors = metric_vectors(reference, candidate, reference)
    np.testing.assert_allclose(vectors["tv_from_eval"], [0.5, 0.5, 0])
    np.testing.assert_allclose(vectors["top1_changed_from_eval"], [1, 1, 0])
    assert (vectors["fresh_target_kl_change_from_eval"][:2] > 0).all()
    report = summarize(vectors, np.array([0, 0, 1]))
    assert report["n_episodes"] == 2
    assert report["episode_mean_ci95"]["tv_from_eval"]["mean"] == pytest.approx(0.25)
    assert report["pooled_state_means"]["tv_from_eval"] == pytest.approx(1 / 3)
    single = summarize(vectors, np.array([0, 0, 0]))
    assert single["episode_mean_ci95"]["tv_from_eval"]["bootstrap_ci95"] is None


def _write_bank(path, digest="abc", episode=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    obs = np.zeros((7, 1, 2, 2), dtype=np.uint8)
    np.savez(path, obs=obs, priors=np.full((7, 3), 1 / 3), policies=np.full((7, 3), 1 / 3),
             steps=np.arange(7) * 10, episode_index=episode, checkpoint_sha256=digest)


def test_banks_bound_each_condition_and_reject_wrong_checkpoint(tmp_path):
    for condition in ("greedy", "sampled"):
        for episode in range(3):
            _write_bank(tmp_path / condition / f"episode_{episode:04d}_bank.npz", episode=episode)
    banks = load_banks(tmp_path, "abc", max_episodes=2, states_per_episode=3)
    assert set(banks) == {"greedy", "sampled"}
    for bank in banks.values():
        assert len(bank["obs"]) == 6
        np.testing.assert_array_equal(bank["steps"], [0, 30, 60, 0, 30, 60])
        np.testing.assert_array_equal(bank["episode_index"], [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_banks(tmp_path, "different")


def test_regroupings_do_not_inflate_episode_count():
    net, obs = _case()
    prior, _ = evaluate_mode(net, obs, "cpu", "eval", 4)
    bank = {"obs": obs, "policies": prior, "priors": prior,
            "episode_index": np.repeat([0, 1, 2], 4), "files": []}
    report = probe_condition(net, bank, "cpu", batch_size=4, regroupings=3)
    assert report["frozen_eval"]["n_states"] == 12
    assert report["frozen_eval"]["n_episodes"] == 3
    for arm in report["arms"].values():
        assert arm["mean_metrics_across_regroupings"]["n_episodes"] == 3
        assert len(arm["regroupings"]) == 3
        assert arm["composition_dependence"]["pooled_state_means"]["pairwise_regrouping_tv"] > 0


def test_invalid_distribution_and_order_fail_loudly():
    with pytest.raises(ValueError, match="normalized"):
        metric_vectors(np.array([[0.8, 0.8]]), np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))
    net, obs = _case()
    with pytest.raises(ValueError, match="permutation"):
        evaluate_mode(net, obs, "cpu", "eval", 4, np.zeros(len(obs), dtype=int))
