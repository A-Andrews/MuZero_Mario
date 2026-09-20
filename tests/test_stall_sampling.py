import copy

import pytest

from scripts.eval_controller_diagnostic import select_plan
from scripts.stall_sampling_controller import SETTINGS, StallSamplingController
from tests.test_controller_diagnostic import manifest


def test_trigger_waits_for_96_decisions_and_samples_exactly_32():
    controller = StallSamplingController()
    assert all(not controller.observe(2370, 8)["rescue_active"] for _ in range(96))
    first = controller.observe(2370, 8)
    assert first["rescue_trigger"] and first["controller_temperature"] == .25
    assert first["stall_window_span"] == 0
    for _ in range(31):
        row = controller.observe(2370, 8)
        assert row["rescue_active"] and not row["rescue_trigger"]
    assert all(not controller.observe(2370, 8)["rescue_active"] for _ in range(96))
    assert controller.observe(2370, 8)["rescue_trigger"]


def test_progress_and_wide_oscillations_do_not_trigger():
    controller = StallSamplingController()
    for t in range(500):
        assert not controller.observe(3*t, 8)["rescue_active"]
    controller = StallSamplingController()
    for t in range(500):
        assert not controller.observe(300 + (30 if t%2 else 0), 8)["rescue_active"]


def test_exact_span_boundary_and_noncontrol_clear_history():
    controller = StallSamplingController()
    for _ in range(96):
        controller.observe(100, 8)
    assert controller.observe(116, 8)["rescue_trigger"]
    assert not controller.observe(116, 11)["rescue_active"]
    for _ in range(96):
        assert not controller.observe(116, 8)["rescue_active"]
    assert controller.observe(116, 8)["rescue_trigger"]


def test_fresh_episode_has_no_carried_trigger_state():
    first = StallSamplingController()
    for _ in range(97):
        row = first.observe(500, 8)
    assert row["rescue_active"]
    assert not StallSamplingController().observe(500, 8)["rescue_active"]


def gated_manifest():
    data = manifest()
    data["evaluation_profile"] = "stall_gated"
    data["stall_controller"] = SETTINGS.copy()
    data["levels"] = {level: {"checkpoint": "frozen.pt", "checkpoint_sha256": "a"*64}
                      for level in ("Level1-1", "Level6-1", "Level1-3")}
    data["conditions"] = [c for c in data["conditions"] if c["name"] in ("greedy", "sampled")]
    data["conditions"].append({"name": "stall_sampled", "leaf_batch": 4, "eps": 0., "alpha": .25, "temperature": 0.})
    data["previous_confirmation_trials"] = data["confirmation_trials"]
    data["confirmation_trials"] = [{"episode_index": i, "env_seed": 5100001+i, "search_seed": 6100001+i} for i in range(100)]
    return data


def test_gated_protocol_includes_both_controls_all_trials_and_fresh_reserve():
    data = gated_manifest()
    plan = select_plan(data)
    assert len(plan) == 90
    assert [c["name"] for c, _ in plan[:3]] == ["greedy", "sampled", "stall_sampled"]
    bad = copy.deepcopy(data)
    del bad["levels"]["Level1-3"]
    with pytest.raises(ValueError, match="both"):
        select_plan(bad)
    bad = copy.deepcopy(data)
    bad["stall_controller"]["burst_decisions"] = 8
    with pytest.raises(ValueError, match="frozen protocol"):
        select_plan(bad)
    bad = copy.deepcopy(data)
    bad["confirmation_trials"] = bad["previous_confirmation_trials"]
    with pytest.raises(ValueError, match="new seed pairs"):
        select_plan(bad)
    with pytest.raises(ValueError, match="smoke"):
        select_plan(data, conditions=["stall_sampled"])


def test_gated_confirmation_keeps_reserved_trials_and_both_controls():
    data = gated_manifest()
    data.update(split="confirmation", episodes=100,
                development_trials=copy.deepcopy(data["trials"]),
                trials=copy.deepcopy(data["confirmation_trials"]))
    plan = select_plan(data)
    assert len(plan) == 300
    assert [trial for condition, trial in plan if condition["name"] == "stall_sampled"] == data["confirmation_trials"]
    with pytest.raises(ValueError, match="consume confirmation"):
        select_plan(data, smoke=True)
    with pytest.raises(ValueError, match="smoke"):
        select_plan(data, episodes=10)
    bad = copy.deepcopy(data)
    bad["trials"][0]["search_seed"] += 1000
    with pytest.raises(ValueError, match="unchanged reserved"):
        select_plan(bad)
    bad = copy.deepcopy(data)
    bad["previous_confirmation_trials"] = bad["confirmation_trials"]
    with pytest.raises(ValueError, match="new seed pairs"):
        select_plan(bad)
    bad = copy.deepcopy(data)
    del bad["levels"]["Level1-3"]
    with pytest.raises(ValueError, match="both"):
        select_plan(bad)
