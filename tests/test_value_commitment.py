import numpy as np
import pytest

from scripts.diagnose_value_commitment import (check_baseline, decompose_error,
    held_action, return_tails, unique_roots)


def test_value_error_decomposition_separates_three_sources():
    # Predicted return 2+.5*12=8; realized return 1+.5*4=3.
    result = decompose_error([2], [1], 12, 8, 4, .5)
    assert result['total_error'] == 5
    assert result['reward_error'] == 1
    assert result['imagined_state_value_gap'] == 2
    assert result['observed_state_value_residual'] == 2
    assert result['actual_rewards_observed_value'] == 5


def test_return_tails_have_zero_terminal_value_and_preserve_prefix_identity():
    assert np.array_equal(return_tails([1, 2, 4], .5), [3, 4, 4, 0])
    result = decompose_error([1, 2], [1, 2], 4, 4, 4, .5)
    assert result['total_error'] == 0
    assert result['realized_return'] == 3
    with pytest.raises(ValueError, match='same horizon'):
        decompose_error([1], [1, 2], 4, 4, 4, .5)


def test_hold_one_is_identity_and_hold_two_changes_only_second_decision():
    proposals = [4, 0, 2, 1]
    assert [held_action(a, 4, t, 1) for t, a in enumerate(proposals)] == proposals
    assert [held_action(a, 4, t, 2) for t, a in enumerate(proposals)] == [4, 4, 2, 1]


def test_repeated_references_do_not_inflate_roots_or_episode_count():
    case = {'case_id': 1, 'level': 'Level6-1', 'source_condition': 'greedy',
            'episode_index': 93, 'group': 'successful_reference',
            'roots': [{'step': 160, 'x': 1393}, {'step': 160, 'x': 1393}]}
    cases = [case, {**case, 'case_id': 3}, {**case, 'case_id': 5, 'episode_index': 94}]
    roots = unique_roots(cases)
    assert len(roots) == 2
    assert len(roots[0]['aliases']) == 4
    assert roots[0]['reference'] == {'case_id': 1, 'root_index': 0}
    assert roots[1]['root_id'] == 1


def test_baseline_check_rejects_changed_actions_and_rewards():
    row = {key: 0 for key in ('action', 'controller_action', 'temperature',
        'rescue_trigger', 'x_after', 'player_state', 'reward', 'done', 'died', 'completed', 'root_q')}
    row.update(action=4, controller_action=4, reward=2.)
    trace = {key: np.asarray([value]) for key, value in row.items()}
    check_baseline([row], trace)
    with pytest.raises(ValueError, match='action'):
        check_baseline([{**row, 'action': 2}], trace)
    with pytest.raises(ValueError, match='reward'):
        check_baseline([{**row, 'reward': 3.}], trace)
