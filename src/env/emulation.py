import retro
import numpy as np


def add_unused_buttons(actions):
    """Pad the 6-button COMPLEX_MOVEMENT vector with unused NES buttons."""
    return [actions[0], 0, 0, 0] + actions[1:]


def emulator_step(emulator, actions):
    """Step the emulator and handle auto-reset on done. Returns (obs, done, info)."""
    obs, _rew, term, trunc, info = emulator.step(add_unused_buttons(actions))
    done = term or trunc
    if done:
        emulator.reset()
    return obs, done, info


def make_emulator(level, int_path=None):
    """Instantiate a bare retro emulator for a given level state."""
    if int_path is not None:
        retro.data.Integrations.add_custom_path(str(int_path))
    return retro.make(
        "SuperMarioBros-Nes",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        state=level,
        render_mode=None,
    )
