"""Cadence + separation tests for the checkpoint / replay split.

`MuzeroLearner.__init__` is heavy (builds networks, optimizer, coordinator),
so these tests construct a bare instance via ``object.__new__`` and set only
the attributes the methods under test touch. That keeps the tests fast and
GPU/env free while still exercising the real method bodies.
"""
import threading

import pytest
import torch

from src.muzero import muzero as muzero_mod
from src.muzero.muzero import MuzeroLearner


# ---------------------------------------------------------------------------
# _resolve_replay_every: the fallback rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "replay_cfg, save_every, expected",
    [
        (1000, 40, 1000),   # explicit positive value is used as-is
        (0, 40, 40),        # unset / zero falls back to save_every
        (-5, 40, 40),       # negative also falls back
        (1, 5000, 1),       # tiny positive value is honored (not clamped)
    ],
)
def test_resolve_replay_every(replay_cfg, save_every, expected):
    assert MuzeroLearner._resolve_replay_every(replay_cfg, save_every) == expected


def test_resolve_replay_every_coerces_to_int():
    # Hydra/yaml can hand us a float; the cadence comparison needs an int.
    out = MuzeroLearner._resolve_replay_every(2500.0, 40)
    assert out == 2500 and isinstance(out, int)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Net:
    def state_dict(self):
        return {"w": torch.zeros(2)}


class _Opt:
    def state_dict(self):
        return {"step": 0}


def _bare_learner():
    return object.__new__(MuzeroLearner)


# ---------------------------------------------------------------------------
# _save_checkpoint: writes ckpt, updates symlink, rotates, and does NOT replay
# ---------------------------------------------------------------------------
def test_save_checkpoint_writes_and_does_not_launch_replay(tmp_path, monkeypatch):
    saved = {}
    rotated = {}

    def fake_save_checkpoint(*, path, **kwargs):
        path.write_bytes(b"ckpt")
        saved["path"] = path
        saved["kwargs"] = kwargs

    def fake_rotate(ckpt_dir, keep=10):
        rotated["args"] = (ckpt_dir, keep)

    monkeypatch.setattr(muzero_mod, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(muzero_mod, "rotate_checkpoints", fake_rotate)

    lr = _bare_learner()
    lr.ckpt_dir = tmp_path
    lr.training_step = 7
    lr.env_step = 123
    lr.cfg = {}
    lr.net = _Net()
    lr.opt = _Opt()
    lr.scheduler = None

    lr._save_checkpoint()

    ckpt = tmp_path / "step_7.pt"
    latest = tmp_path / "latest.pt"
    assert saved["path"] == ckpt and ckpt.exists()
    assert latest.is_symlink() and latest.resolve() == ckpt.resolve()
    assert rotated["args"] == (tmp_path, 10)
    # The split's whole point: saving a checkpoint must not spawn a replay.
    assert not hasattr(lr, "_replay_thread")


def test_save_checkpoint_replaces_existing_latest_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(
        muzero_mod,
        "save_checkpoint",
        lambda *, path, **kw: path.write_bytes(b"x"),
    )
    monkeypatch.setattr(muzero_mod, "rotate_checkpoints", lambda *a, **k: None)

    lr = _bare_learner()
    lr.ckpt_dir = tmp_path
    lr.env_step = 0
    lr.cfg = {}
    lr.net = _Net()
    lr.opt = _Opt()
    lr.scheduler = None

    lr.training_step = 1
    lr._save_checkpoint()
    lr.training_step = 2
    lr._save_checkpoint()

    latest = tmp_path / "latest.pt"
    assert latest.is_symlink()
    assert latest.resolve() == (tmp_path / "step_2.pt").resolve()


# ---------------------------------------------------------------------------
# _maybe_launch_replay: gating
# ---------------------------------------------------------------------------
class _FakeThread:
    instances = []

    def __init__(self, target=None, args=(), name=None, daemon=None):
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.started = False
        _FakeThread.instances.append(self)

    def start(self):
        self.started = True  # deliberately do NOT run target (no GPU/env)

    def is_alive(self):
        return self.started


class _FakeThreadingModule:
    Thread = _FakeThread


@pytest.fixture(autouse=True)
def _reset_fake_threads():
    _FakeThread.instances = []
    yield
    _FakeThread.instances = []


def test_replay_skipped_when_videos_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(muzero_mod, "threading", _FakeThreadingModule)
    lr = _bare_learner()
    lr.cfg = {"wandb": {"log_videos_every_ckpt": False}}
    lr._replay_thread = None
    lr.video_dir = tmp_path
    lr.training_step = 10

    lr._maybe_launch_replay()

    assert _FakeThread.instances == []
    assert lr._replay_thread is None
    assert list(tmp_path.iterdir()) == []  # no video dir created


def test_replay_skipped_when_previous_still_running(tmp_path, monkeypatch):
    monkeypatch.setattr(muzero_mod, "threading", _FakeThreadingModule)
    lr = _bare_learner()
    lr.cfg = {"wandb": {"log_videos_every_ckpt": True}}
    lr.video_dir = tmp_path
    lr.training_step = 10

    prev = _FakeThread(name="prev")
    prev.start()  # makes is_alive() True
    _FakeThread.instances = []  # ignore the prev thread for the assertion below
    lr._replay_thread = prev

    lr._maybe_launch_replay()

    assert _FakeThread.instances == []  # no new thread spawned
    assert lr._replay_thread is prev   # unchanged


def test_replay_launches_when_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(muzero_mod, "threading", _FakeThreadingModule)
    lr = _bare_learner()
    lr.cfg = {"wandb": {"log_videos_every_ckpt": True}}
    lr.video_dir = tmp_path
    lr.training_step = 42
    lr._replay_thread = None
    lr.net = _Net()
    sentinel = object()
    lr._run_replay = lambda *a, **k: sentinel  # stub; never actually invoked

    lr._maybe_launch_replay()

    assert len(_FakeThread.instances) == 1
    th = _FakeThread.instances[0]
    assert th.started is True
    assert th.target == lr._run_replay
    assert th.daemon is True
    assert th.name == "replay-step-42"
    # weight snapshot is passed positionally as the first arg
    cpu_sd, video_dir = th.args
    assert set(cpu_sd.keys()) == {"w"}
    assert video_dir == tmp_path / "step_42"
    assert (tmp_path / "step_42").is_dir()
    assert lr._replay_thread is th


def test_replay_videos_default_true_when_key_absent(tmp_path, monkeypatch):
    # `.get("log_videos_every_ckpt", True)` — absent key must not skip replay.
    monkeypatch.setattr(muzero_mod, "threading", _FakeThreadingModule)
    lr = _bare_learner()
    lr.cfg = {"wandb": {}}
    lr.video_dir = tmp_path
    lr.training_step = 1
    lr._replay_thread = None
    lr.net = _Net()
    lr._run_replay = lambda *a, **k: None

    lr._maybe_launch_replay()

    assert len(_FakeThread.instances) == 1
