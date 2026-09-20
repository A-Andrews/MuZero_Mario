"""Check fixed-attempt accounting and paired trials before allocating rollouts."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.eval_controller_diagnostic import (
    execute_plan, numpy_search_rng, run_episode, select_plan,
)


def manifest():
    data = json.loads((Path(__file__).resolve().parents[1] /
                       "docs/controller_policy_diagnostic_v1.json").read_text())
    for level in data["levels"].values():
        level["checkpoint_sha256"] = "a" * 64
    return data


def test_source_identity_accepts_cluster_symlink_spelling(tmp_path, monkeypatch):
    import scripts.eval_controller_diagnostic as diagnostic
    real = tmp_path / "lustre_source"
    real.mkdir()
    alias = tmp_path / "projects_source"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(diagnostic, "REPO_ROOT", real)
    monkeypatch.setattr(diagnostic, "file_sha", lambda path: "a" * 64)
    monkeypatch.setattr(diagnostic, "__file__", str(alias / "scripts/eval_controller_diagnostic.py"))
    aliased = diagnostic.implementation_sha()
    monkeypatch.setattr(diagnostic, "__file__", str(real / "scripts/eval_controller_diagnostic.py"))
    assert diagnostic.implementation_sha() == aliased


class FakeEnv:
    def __init__(self, seed, terminate=None, completed=False):
        self.rng = np.random.default_rng(seed)
        self.terminate, self.completed = terminate, completed
        self.closed = False

    def reset(self):
        self.offset = int(self.rng.integers(0, 31))
        self.step_number = 0
        self.last_info = {"player_x_posHi": 0, "player_x_posLo": 0,
                          "player_state": 8, "lives": 2}
        return np.zeros((1, 2, 2), np.float32)

    def step(self, action):
        self.step_number += 1
        done = self.terminate == self.step_number
        info = {"player_x_posHi": 0, "player_x_posLo": self.step_number,
                "player_state": 8, "lives": 1 if done and not self.completed else 2,
                "level_complete": done and self.completed}
        return np.full((1, 2, 2), self.step_number / 255, np.float32), 1, done, info

    def close(self):
        self.closed = True


def runner(condition, trial, *, terminate=None, completed=False, max_steps=8):
    env = FakeEnv(trial["env_seed"], terminate, completed)
    def decide(obs):
        return np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0, 1.0
    result = run_episode(env, decide, trial, max_steps=max_steps, bank_stride=3, bank_tail=3)
    assert env.closed
    return result


def test_greedy_keeps_all_thirty_trials_and_start_seeds_are_paired():
    plan = select_plan(manifest())
    assert len(plan) == 150
    for i in range(30):
        cells = [(condition, trial) for condition, trial in plan if trial["episode_index"] == i]
        assert len(cells) == 5
        assert len({trial["env_seed"] for _, trial in cells}) == 1
        assert len({trial["search_seed"] for _, trial in cells}) == 1
    with pytest.raises(ValueError, match="smoke"):
        select_plan(manifest(), episodes=1)
    with pytest.raises(ValueError, match="control"):
        data = manifest()
        del data["levels"]["Level6-1"]
        select_plan(data)


def test_banks_keep_pre_action_alignment_and_tail_without_duplicates():
    trial = {"episode_index": 0, "env_seed": 12, "search_seed": 21}
    result, trace, bank = runner({}, trial)
    assert result["timed_out"] and not result["completed"] and not result["died"]
    assert result["longest_action_run"] == 8
    assert result["visit_top_tie_fraction"] == 1
    np.testing.assert_array_equal(bank["steps"], [0, 3, 5, 6, 7])
    np.testing.assert_array_equal(bank["obs"][:, 0, 0, 0], bank["steps"])
    np.testing.assert_array_equal(bank["x_before"], bank["steps"])
    np.testing.assert_array_equal(bank["x_after"], bank["steps"] + 1)
    assert len(trace["actions"]) == 8


@pytest.mark.parametrize("completed", [False, True])
def test_death_and_completion_terminate_and_count_as_different_outcomes(completed):
    result, _, bank = runner({}, {"episode_index": 0, "env_seed": 3, "search_seed": 9},
                             terminate=3, completed=completed)
    assert result["steps"] == 3
    assert result["completed"] == completed
    assert result["died"] != completed
    assert not result["timed_out"]
    assert len(bank["obs"]) == 3


def test_search_rng_replays_and_does_not_change_caller_stream():
    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)
    with numpy_search_rng(987):
        first = np.random.random(4)
    with numpy_search_rng(987):
        np.testing.assert_array_equal(first, np.random.random(4))
    np.testing.assert_array_equal(expected, np.random.random(4))


def test_all_failures_are_counted_and_resume_preserves_trial_artifacts(tmp_path):
    plan = select_plan(manifest(), smoke=True, episodes=2)
    identity = {"checkpoint_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    summary = execute_plan(tmp_path, identity, plan, runner)
    assert summary["complete"]
    for condition in summary["conditions"]:
        assert condition["finished_trials"] == condition["planned_trials"] == 2
        assert condition["level_completions"] == 0
        assert condition["timeouts"] == 2
    with np.load(tmp_path / "greedy/episode_0000_trace.npz") as data:
        assert data["completed"].shape == (8,)
        assert data["episode_completed"].shape == ()
    def must_not_run(*args):
        raise AssertionError("completed trial was repeated")
    assert execute_plan(tmp_path, identity, plan, must_not_run) == summary
    wrong = {**identity, "checkpoint_sha256": "c" * 64}
    with pytest.raises(ValueError, match="identity differs"):
        execute_plan(tmp_path, wrong, plan, runner)


def test_infrastructure_failure_cannot_be_hidden_in_complete_denominator(tmp_path):
    plan = select_plan(manifest(), smoke=True, episodes=1)
    identity = {"checkpoint_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    def fails(condition, trial):
        raise RuntimeError("simulated broken environment")
    with pytest.raises(RuntimeError):
        execute_plan(tmp_path, identity, plan, fails)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert not summary["complete"]
    assert summary["conditions"][0]["finished_trials"] == 0
    assert summary["conditions"][0]["infrastructure_errors"] == 1
    assert (tmp_path / "infrastructure_errors.jsonl").exists()
    repaired = execute_plan(tmp_path, identity, plan, runner)
    assert repaired["complete"]


def test_mismatched_environment_start_is_rejected(tmp_path):
    plan = select_plan(manifest(), smoke=True, episodes=1)
    identity = {"checkpoint_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    def mismatched(condition, trial):
        result, trace, bank = runner(condition, trial)
        if condition["name"] == "sequential_greedy":
            result["initial_x"] = 99
        return result, trace, bank
    with pytest.raises(ValueError, match="starting conditions differ"):
        execute_plan(tmp_path, identity, plan, mismatched)
