"""Boundary cases that change the scientific interpretation of target probes."""
import numpy as np
import pytest

from scripts.diagnose_value_targets import target_from_values, target_spec
from src.muzero.returns import compute_n_step_returns


def test_terminal_reward_included_without_bootstrapping_terminal_value():
    rule = target_spec([1, 2, 4], True, 1, 10, .5)
    assert rule['bootstrap_index'] is None
    assert rule['bootstrap_factor'] == 0
    assert target_from_values(rule, {}) == 4


def test_truncation_keeps_last_observed_value_at_correct_discount():
    rule = target_spec([1, 2, 999], False, 1, 10, .5)
    assert rule['bootstrap_index'] == 2
    # The final reward has no saved successor observation. Do not count it
    # and then incorrectly bootstrap the predecessor as though it came after.
    assert target_from_values(rule, {2: 8}) == 6
    last = target_spec([1, 2, 999], False, 2, 10, .5)
    assert last['reward_window'] == 0
    assert last['bootstrap_factor'] == 1
    assert target_from_values(last, {2: 8}) == 8


@pytest.mark.parametrize('terminal', [True, False])
@pytest.mark.parametrize('horizon', [1, 10, 50, 100, 200])
def test_selected_bootstrap_observation_matches_collection_returns(terminal, horizon):
    rewards = np.linspace(-1, 4, 23, dtype=np.float32)
    values = np.linspace(7, 77, 23, dtype=np.float32)
    expected = compute_n_step_returns(rewards, values, horizon, .999, terminal)
    for t in [0, 1, 8, 16, 22]:
        rule = target_spec(rewards, terminal, t, horizon, .999)
        assert target_from_values(rule, dict(enumerate(values))) == pytest.approx(expected[t], rel=1e-5, abs=1e-5)


def test_short_bootstrap_can_hide_a_delayed_success_reward():
    rewards = [0.] * 20 + [100.]
    values = dict(enumerate([2.] * 21))
    short = target_from_values(target_spec(rewards, True, 0, 10, .999), values)
    complete = target_from_values(target_spec(rewards, True, 0, 50, .999), values)
    assert short == pytest.approx(2 * .999**10)
    assert complete == pytest.approx(100 * .999**20)
    assert complete > 40 * short
