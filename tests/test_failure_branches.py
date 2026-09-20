import numpy as np
import pytest

from scripts.diagnose_failure_branches import (death_anchors, match_success,
    monitor_at, prefix_actions, summarize_rows)
from scripts.stall_sampling_controller import StallSamplingController


def source(n=300, onset=220):
    record = {"initial_player_state": 8, "died": True, "completed": False,
              "noop_frames_requested": 4, "episode_index": 0}
    trace = {"player_state": np.r_[np.full(onset, 8), np.full(n-onset, 11)],
             "died": np.arange(n) == n-1, "bank_steps": np.arange(n),
             "actions": np.ones(n, dtype=int), "x_before": np.arange(n)*3}
    return record, trace


def test_anchors_precede_death_onset_not_delayed_life_loss():
    record, trace = source()
    onset, roots = death_anchors(record, trace, [32,16])
    assert onset == 220
    assert [r["step"] for r in roots] == [188,204]
    trace["bank_steps"] = np.arange(0,300,20)
    assert [r["step"] for r in death_anchors(record, trace, [32,16])[1]] == [180,200]


def test_anchor_selection_refuses_missing_or_collapsed_roots():
    record, trace = source(onset=10)
    with pytest.raises(ValueError, match="eligible"):
        death_anchors(record, trace, [32,16])
    record, trace = source()
    trace["bank_steps"] = np.array([0,100,250])
    with pytest.raises(ValueError, match="collapsed"):
        death_anchors(record, trace, [32,16])


def test_offscreen_fall_precedes_delayed_death_animation():
    record, trace = source()
    trace["below_playfield"] = np.arange(300) >= 100
    onset, roots = death_anchors(record, trace, [32,16])
    assert onset == 100
    assert [r["step"] for r in roots] == [68,84]


def test_success_reference_requires_intact_completion_and_matches_x():
    record, trace = source()
    roots = [{"step": 100, "x": 300}, {"step": 110, "x": 330}]
    with pytest.raises(ValueError, match="complete intact"):
        match_success(roots, record, [(record,trace)])
    good = {**record, "completed": True, "died": False, "episode_index": 5}
    chosen, matched = match_success(roots, record, [(good,trace)])
    assert chosen["episode_index"] == 5
    assert [r["step"] for r in matched] == [100,110]
    assert all(r["x_error"] == 0 for r in matched)


def test_reconstructed_monitor_preserves_active_burst_and_rearm_history():
    record, trace = source(n=300,onset=250)
    trace["x_before"] = np.full(300,2370)
    original = StallSamplingController()
    expected = [original.observe(2370, int(8 if t == 0 else trace["player_state"][t-1])) for t in range(300)]
    for t in (0,95,96,100,127,128,200,224,260):
        restored = monitor_at(record,trace,t,True)
        state = 8 if t == 0 else trace["player_state"][t-1]
        assert restored.observe(2370,int(state)) == expected[t]
    assert monitor_at(record,trace,100,False) is None


def test_prefix_actions_are_eight_decisions_and_continue_is_unforced():
    assert prefix_actions("continue", [1]*8) == []
    assert prefix_actions("right_jump", [1]*8) == [2]*8
    assert prefix_actions("right_run_jump", [1]*8) == [4]*8
    assert prefix_actions("release", [1]*8) == [0]*8
    assert prefix_actions("logged_prefix", range(8)) == list(range(8))
    with pytest.raises(ValueError, match="eight"):
        prefix_actions("logged_prefix", range(7))


def test_local_survival_does_not_imply_completion_and_original_cap_is_respected():
    rows = [{"completed":False,"died":False,"x_after":100+i,"reward":1.,"done":False} for i in range(10)]
    rows[-1].update(done=True,died=True)
    result = summarize_rows(rows,100,.9,5,20)
    assert not result["local"]["died"] and result["died"]
    assert not result["completed"] and not result["timed_out"]
    rows[-1].update(done=False,died=False)
    assert summarize_rows(rows,100,.9,5,10)["timed_out"]
