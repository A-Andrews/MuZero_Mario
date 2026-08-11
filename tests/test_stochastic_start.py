"""Stochastic starts (env.noop_max / env.skip_to_control).

These run against a stub emulator rather than the ROM so the suite stays
emulator-free. The stub reproduces the two behaviours that actually broke the
first implementation:

  - `reset()` returns an **empty** info dict, so `player_state` can only be
    read by stepping (the skip has to be a do-while, not a while);
  - every level opens with a scripted intro that ignores input, so a delay
    spent inside it is absorbed and buys no variation at all.

Measured against the real emulator: the intro is 123 frames on Level1-1 and
117 on Level1-2, with player_state going 0 -> 7 -> 8 ("player in control").
"""
import numpy as np
import pytest

from src.env.env import CustomWrapper

INTRO_FRAMES = 123


class FakeRetroEnv:
    """Minimal stand-in for a stable-retro env with a scripted intro."""

    def __init__(self, intro_frames=INTRO_FRAMES, expose_player_state=True):
        self.intro_frames = intro_frames
        self.expose_player_state = expose_player_state
        self.frames = 0
        self.steps_taken = 0

    def _frame(self):
        # Distinct per frame so observation-level changes are detectable.
        return np.full((240, 256, 3), self.frames % 256, dtype=np.uint8)

    def _info(self):
        info = {
            "time": 400 - self.frames // 24,
            "player_x_posHi": 0,
            "player_x_posLo": 40,
            "score": 0,
            "lives": 2,
        }
        if self.expose_player_state:
            info["player_state"] = 8 if self.frames >= self.intro_frames else 0
        return info

    def reset(self):
        self.frames = 0
        self.steps_taken = 0
        return self._frame(), {}  # empty info, exactly like stable-retro

    def step(self, _action):
        self.frames += 1
        self.steps_taken += 1
        return self._frame(), 0.0, False, False, self._info()

    def close(self):
        pass


def make(noop_max=0, skip_to_control=False, seed=0, **kw):
    fake = FakeRetroEnv(**kw)
    env = CustomWrapper(
        fake, n_frame=4, downsample=4, pad_to=96, seed=seed,
        noop_max=noop_max, skip_to_control=skip_to_control,
    )
    return env, fake


def test_defaults_take_no_extra_frames():
    """Knobs off must reproduce the old deterministic reset exactly."""
    env, fake = make()
    env.reset()
    assert fake.steps_taken == 0


def test_skip_to_control_stops_at_control_onset():
    env, fake = make(skip_to_control=True)
    env.reset()
    assert fake.steps_taken == INTRO_FRAMES
    assert fake._info()["player_state"] == 8


def test_skip_is_a_do_while_despite_empty_reset_info():
    """Regression: an empty reset info once made the skip a silent no-op."""
    env, fake = make(skip_to_control=True)
    env.reset()
    assert fake.steps_taken > 0


@pytest.mark.parametrize("seed", range(8))
def test_noop_delay_lands_in_range(seed):
    env, fake = make(noop_max=30, skip_to_control=True, seed=seed)
    env.reset()
    delay = fake.steps_taken - INTRO_FRAMES
    assert 0 <= delay <= 30


def test_noop_delay_actually_varies_across_seeds():
    delays = set()
    for seed in range(24):
        env, fake = make(noop_max=30, skip_to_control=True, seed=seed)
        env.reset()
        delays.add(fake.steps_taken - INTRO_FRAMES)
    assert len(delays) > 1


def test_counters_adopted_so_first_step_is_not_charged_for_the_delay():
    """Without adopting post-delay counters the agent eats the timer ticks."""
    env, fake = make(noop_max=30, skip_to_control=True, seed=1)
    env.reset()
    assert env.last_time == fake._info()["time"]
    assert env.last_x == 40
    assert env.curr_lives == 2


def test_missing_player_state_does_not_burn_the_cap():
    """A data.json without player_state should degrade, not spin."""
    env, fake = make(skip_to_control=True, expose_player_state=False)
    env.reset()
    assert fake.steps_taken == 1
    assert fake.steps_taken < CustomWrapper._SKIP_TO_CONTROL_MAX_FRAMES


def test_noop_without_skip_stays_inside_the_intro():
    """Documents why skip_to_control is mandatory, not decorative."""
    env, fake = make(noop_max=30, skip_to_control=False, seed=3)
    env.reset()
    assert fake.steps_taken <= 30 < INTRO_FRAMES
