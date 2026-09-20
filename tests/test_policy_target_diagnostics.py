"""Scientific interpretation and bank-provenance checks; run on compute nodes."""
import json

import numpy as np
import pytest

from scripts.diagnose_policy_targets import (
    balanced_search_indices, choose_episode_files, fit_metrics,
    probabilities, read_bank, repeated_target_metrics, state_information, stratified_indices,
)


def test_flat_target_high_cross_entropy_is_not_bad_fit():
    target = np.full((4, 12), 1 / 12)
    result = fit_metrics(target, target)
    assert result["cross_entropy_mean_nats"] == pytest.approx(np.log(12))
    assert result["kl_target_to_prior_mean_nats"] == pytest.approx(0)
    assert result["target"]["exact_top_tie_fraction"] == 1
    assert result["target"]["top_two_margin_mean"] == 0


def test_noisy_targets_can_have_mismatch_even_when_expected_target_is_fitted():
    result = repeated_target_metrics([[[1, 0], [0, 1]]], [[0.5, 0.5]])
    assert result["mean_target_fit"]["kl_target_to_prior_mean_nats"] == pytest.approx(0)
    assert result["mean_individual_target_kl_nats"] == pytest.approx(np.log(2))
    assert result["target_variation_contribution_nats"] == pytest.approx(np.log(2))


def test_tie_aware_agreement_accepts_any_maximal_target_action():
    target = [[0.4, 0.4, 0.2]]
    prior = [[0.2, 0.7, 0.1]]
    result = fit_metrics(target, prior)
    assert result["argmax_agreement"] == 0
    assert result["tie_aware_argmax_agreement"] == 1
    assert result["kl_target_to_prior_mean_nats"] > 0
    # Close values are not reported as exact visit ties.
    near = fit_metrics([[0.40001, 0.39999, 0.2]], prior)
    assert near["target"]["exact_top_tie_fraction"] == 0
    assert near["tie_aware_argmax_agreement"] == 0


def test_shuffle_detects_state_information_even_with_identical_marginals():
    target = np.array([[0.9, 0.1], [0.1, 0.9]])
    result = state_information(target, target, ["ep", "ep"], repeats=4)
    assert result["within_episode_shuffle_ce_minus_correct_nats"] > 1
    assert result["constant_mean_prior_ce_minus_correct_nats"] > 0
    assert result["within_episode_shuffle_n_states"] == 2
    # A policy independent of the observation gets no shuffle penalty.
    flat = state_information(target, np.full((2, 2), 0.5), ["ep", "ep"])
    assert flat["within_episode_shuffle_ce_minus_correct_nats"] == pytest.approx(0)


def test_shuffle_never_crosses_episodes_and_singletons_are_explicit():
    target = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
    result = state_information(target, target, ["a", "a", "b", "b"])
    assert result["within_episode_shuffle_ce_minus_correct_nats"] == pytest.approx(0)
    singleton = state_information(target[:1], target[:1], ["a"])
    assert singleton["within_episode_shuffle_ce_mean_nats"] is None
    json.dumps(singleton, allow_nan=False)


def test_stratification_spans_prefix_progress_and_dense_tail_without_duplicates():
    steps = np.unique(np.r_[np.arange(0, 1000, 20), np.arange(800, 1000)])
    xs = np.minimum(steps * 3, 2400)
    rows = stratified_indices(steps, xs, 24, seed=7)
    assert len(rows) == len(set(rows)) == 24
    assert steps[rows].min() == 0
    assert steps[rows].max() == 999
    assert (steps[rows] < 400).any()
    assert (steps[rows] >= 800).sum() >= 4
    assert np.all(np.diff(steps[rows]) > 0)
    np.testing.assert_array_equal(rows, stratified_indices(steps, xs, 24, seed=7))


def test_search_subset_balances_conditions_and_episodes():
    episode_ids = np.array([f"{condition}:{episode}" for condition in ("greedy", "noisy") for episode in range(3) for _ in range(8)])
    selected = balanced_search_indices(episode_ids, 6)
    assert len(set(episode_ids[selected])) == 6
    assert len(set(selected)) == 6
    selected_two = balanced_search_indices(episode_ids, 2)
    assert {key.split(":")[0] for key in episode_ids[selected_two]} == {"greedy", "noisy"}
    assert len(balanced_search_indices(episode_ids, 0)) == 0


def _write_bank(path, **overrides):
    n = 12
    fields = {
        "obs": np.zeros((n, 4, 8, 8), dtype=np.uint8),
        "priors": np.full((n, 12), 1 / 12, dtype=np.float32),
        "policies": np.full((n, 12), 1 / 12, dtype=np.float32),
        "actions": np.zeros(n, dtype=np.int64), "steps": np.arange(n),
        "x_before": np.arange(n) * 4, "x_after": np.arange(n) * 4 + 4,
        "episode_index": np.array(3), "condition": np.array("greedy"),
        "completed": np.array(False), "timed_out": np.array(True),
        "checkpoint_sha256": np.array("a" * 64),
        "env_seed": np.array(42), "search_seed": np.array(43),
    }
    fields.update(overrides)
    np.savez_compressed(path, **fields)


def test_bank_rejects_wrong_checkpoint_before_scoring(tmp_path):
    path = tmp_path / "episode_0003_bank.npz"
    _write_bank(path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        read_bank(path, "b" * 64, 6)
    bank = read_bank(path, "a" * 64, 6)
    assert bank["metadata"]["n_available_correlated_states"] == 12
    assert bank["metadata"]["n_selected_correlated_states"] == 6
    assert bank["metadata"]["episode_key"] == "greedy:3"
    assert bank["metadata"]["timed_out"]
    assert bank["obs"].dtype == np.uint8


def test_bank_rejects_ambiguous_observation_scaling_and_mixed_episodes(tmp_path):
    path = tmp_path / "episode_0003_bank.npz"
    _write_bank(path, obs=np.zeros((12, 4, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        read_bank(path, "a" * 64, 6)
    _write_bank(path, episode_index=np.arange(12))
    with pytest.raises(ValueError, match="one episode"):
        read_bank(path, "a" * 64, 6)


def test_episode_cap_is_per_condition_and_spans_available_episodes(tmp_path):
    for condition in ("greedy", "noisy"):
        folder = tmp_path / condition
        folder.mkdir()
        for i in range(10):
            (folder / f"episode_{i:04d}_bank.npz").touch()
    files = choose_episode_files(tmp_path, 2)
    assert len(files) == 4
    assert {path.name for path in files} == {"episode_0000_bank.npz", "episode_0009_bank.npz"}


def test_unnormalized_counts_cannot_silently_become_policy_targets():
    with pytest.raises(ValueError, match="sum to one"):
        probabilities([[8, 7, 3]])
    with pytest.raises(ValueError, match="nonnegative"):
        probabilities([[1.1, -0.1]])
