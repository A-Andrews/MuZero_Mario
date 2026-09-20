import numpy as np
import pytest

from scripts.diagnose_stall_pulses import control_step, replay_to_branch, stall_step, summarize_branch
from scripts.eval_controller_diagnostic import observation_uint8, reset_with_provenance
from tests.test_controller_diagnostic import FakeEnv


def test_stall_selection_needs_history_and_followup_and_excludes_spawn():
    x = np.r_[np.arange(200)*3, np.full(300, 597)]
    assert stall_step(x, np.arange(0, 500, 20)) == 300
    assert stall_step(np.full(500, 30), np.arange(0, 500, 20)) is None
    assert stall_step(np.arange(500)*3, np.arange(0, 500, 20)) is None
    assert stall_step(x, [480]) is None


def test_control_selection_does_not_filter_for_success_and_keeps_prefix():
    assert control_step(np.arange(0, 200, 20), 200, .5) == 100
    assert control_step([0, 190, 195], 200, 1.) == 190
    with pytest.raises(ValueError):
        control_step([1, 2], 8, .5)


def test_crossing_escape_threshold_then_dying_does_not_count_as_escape():
    rows = [{"x_after": 180, "reward": 1., "completed": False, "died": False, "done": False},
            {"x_after": 180, "reward": -1., "completed": False, "died": True, "done": True}]
    result = summarize_branch(rows, 100)
    assert result["max_delta_x"] == 80 and not result["forward_escape"]
    rows[-1].update(died=False, completed=True)
    assert summarize_branch(rows, 100)["forward_escape"]


class ReplayEnv(FakeEnv):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.last_info = info
        return obs, reward, done, info


def test_prefix_replay_verifies_reset_rewards_and_saved_observation():
    env = ReplayEnv(9)
    _, original = reset_with_provenance(env)
    for _ in range(3):
        obs, _, _, _ = env.step(0)
    trace = {"actions": np.zeros(3, dtype=int), "x_after": np.arange(1,4),
             "x_before": np.arange(4), "rewards": np.ones(3),
             "bank_steps": np.array([3]), "obs": observation_uint8(obs)[None]}
    replay_to_branch(ReplayEnv(9), original, trace, 3)
    trace["obs"] = np.zeros_like(trace["obs"])
    with pytest.raises(ValueError, match="observation differs"):
        replay_to_branch(ReplayEnv(9), original, trace, 3)
