"""Completion-gated root-Dirichlet anneal (2026-08-18).

The gate exists because the T6 Level1-3/Level4-3 specialists annealed
exploration 0.25 -> 0.05 without ever having completed the level once, which
closed the only route out of the pit-edge local optimum. With the gate on the
schedule's clock does not start until the run's first completion.

These exercise `value_at_gated` directly — the same function
`src/selfplay/worker.py` calls — rather than re-implementing its arithmetic.
"""
import json

from src.muzero.schedules import validate_schedule, value_at, value_at_gated

SCHED = validate_schedule([[0, 0.25], [500_000, 0.05]])


def test_ungated_path_is_unchanged():
    """Gate off must reproduce the historical schedule exactly."""
    for step in (0, 100_000, 250_000, 450_000, 500_000, 900_000):
        assert value_at(step, SCHED) == value_at(step, SCHED)
    assert value_at(450_000, SCHED) == 0.07
    assert value_at(900_000, SCHED) == 0.05


def test_locked_gate_holds_start_eps_forever():
    # The Level1-3 case: 760k RL train steps, never a completion.
    for step in (0, 250_000, 500_000, 760_000, 10**9):
        assert value_at_gated(step, SCHED, None) == 0.25


def test_gate_starts_clock_at_first_completion():
    origin = 200_000
    assert value_at_gated(origin, SCHED, origin) == 0.25
    assert value_at_gated(origin + 250_000, SCHED, origin) == 0.15
    assert value_at_gated(origin + 500_000, SCHED, origin) == 0.05
    # Ungated, those same absolute steps are much further down the schedule.
    assert value_at(origin + 250_000, SCHED) == 0.07
    assert value_at(origin + 500_000, SCHED) == 0.05


def test_gate_is_a_delay_not_a_stretch():
    """Once unlocked the anneal runs at its configured rate, just shifted."""
    origin = 300_000
    for d in (0, 50_000, 100_000, 400_000, 500_000, 800_000):
        assert value_at_gated(origin + d, SCHED, origin) == value_at(d, SCHED)


def test_completion_at_step_zero_matches_ungated():
    """An immediate completion degenerates to the old behaviour."""
    for step in (0, 250_000, 450_000, 900_000):
        assert value_at_gated(step, SCHED, 0) == value_at(step, SCHED)


def test_worker_maps_negative_shared_origin_to_locked():
    """The shared mp.Value uses -1 as 'not yet'; the worker maps it to None."""
    for raw in (-1, -999):
        origin = None if raw < 0 else raw
        assert value_at_gated(760_000, SCHED, origin) == 0.25


def test_sidecar_roundtrip(tmp_path):
    """first_completion.json is what carries the origin across chain legs."""
    p = tmp_path / "first_completion.json"
    p.write_text(json.dumps({"rl_train_step": 12_345, "level": "Level1-3"}))
    assert int(json.loads(p.read_text())["rl_train_step"]) == 12_345


def test_coordinator_worker_arity_matches():
    """The coordinator passes the shared origin positionally — keep them in sync.

    A mismatch here does not fail until workers are spawned inside a real run,
    which on this cluster means burning a SLURM allocation to find out.
    """
    import inspect
    import re
    from pathlib import Path

    from src.selfplay.worker import selfplay_worker

    src = Path("src/selfplay/coordinator.py").read_text()
    m = re.search(r"target=selfplay_worker,\s*args=\((.*?)\n\s*\),", src, re.S)
    assert m, "could not find the worker spawn args in coordinator.py"
    n_args = len([
        ln for ln in m.group(1).splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ])

    params = list(inspect.signature(selfplay_worker).parameters.values())
    required = [p for p in params if p.default is inspect.Parameter.empty]
    assert len(required) <= n_args <= len(params), (
        f"coordinator passes {n_args} positional args but selfplay_worker takes "
        f"{len(required)} required / {len(params)} total"
    )
    assert "anneal_origin_val" in inspect.signature(selfplay_worker).parameters


# --- learner side: first-completion detection and the sidecar ----------------
#
# These call the real MuzeroLearner methods against a stub `self`. Constructing
# a full learner needs a net, optimizer, buffer and coordinator; the logic under
# test touches only the handful of attributes set below, and binding the real
# unbound method keeps the test honest about what the method actually does.


class _FakeCoord:
    def __init__(self):
        self.origin = -1

    def set_anneal_origin(self, step):
        self.origin = int(step)


def _stub_learner(tmp_path, training_step=250_000, rl_offset=50_000, gate=True):
    from src.muzero.muzero import MuzeroLearner

    class _Stub:
        pass

    s = _Stub()
    s.training_step = training_step
    s._rl_offset = rl_offset
    s.env_step = 4_000_000
    s.ckpt_dir = tmp_path
    s.coord = _FakeCoord()
    s._first_completion_step = -1
    s._eps_gate_on_completion = gate
    s._record_first_completion = MuzeroLearner._record_first_completion.__get__(s)
    s._push_anneal_origin = MuzeroLearner._push_anneal_origin.__get__(s)
    return s


def test_first_completion_uses_rl_phase_step_not_raw_step(tmp_path):
    """The origin must be on the same axis the workers read (pretrain removed)."""
    s = _stub_learner(tmp_path, training_step=250_000, rl_offset=50_000)
    s._record_first_completion("Level1-3")
    assert s._first_completion_step == 200_000
    assert s.coord.origin == 200_000


def test_first_completion_writes_sidecar(tmp_path):
    s = _stub_learner(tmp_path)
    s._record_first_completion("Level1-3")
    meta = json.loads((tmp_path / "first_completion.json").read_text())
    assert meta["rl_train_step"] == 200_000
    assert meta["level"] == "Level1-3"
    assert meta["env_step"] == 4_000_000


def test_sidecar_written_even_when_gate_is_off(tmp_path):
    """It doubles as provenance for when a run escaped."""
    s = _stub_learner(tmp_path, gate=False)
    s._record_first_completion("Level3-3")
    assert (tmp_path / "first_completion.json").exists()
    assert s.coord.origin == 200_000


def test_completion_during_pretrain_clamps_to_zero(tmp_path):
    """training_step below the pretrain offset must not yield a negative origin."""
    s = _stub_learner(tmp_path, training_step=10_000, rl_offset=50_000)
    s._record_first_completion("Level1-3")
    assert s._first_completion_step == 0
    assert s.coord.origin == 0


def test_push_origin_is_a_noop_before_first_completion(tmp_path):
    """A resumed leg with no sidecar must leave the gate armed."""
    s = _stub_learner(tmp_path)
    s._push_anneal_origin()
    assert s.coord.origin == -1


def test_sidecar_write_failure_does_not_raise(tmp_path):
    """A full/quota-exceeded filesystem must not kill the run (2026-08-12)."""
    s = _stub_learner(tmp_path)
    s.ckpt_dir = tmp_path / "does" / "not" / "exist"
    s._record_first_completion("Level1-3")
    # In-memory origin still holds for this leg even though the write failed.
    assert s._first_completion_step == 200_000
    assert s.coord.origin == 200_000


def test_coordinator_exposes_set_anneal_origin():
    """_push_anneal_origin degrades to a warning if this is ever renamed."""
    from src.selfplay.coordinator import SelfPlayCoordinator

    assert hasattr(SelfPlayCoordinator, "set_anneal_origin")
    assert hasattr(SelfPlayCoordinator, "get_anneal_origin")
