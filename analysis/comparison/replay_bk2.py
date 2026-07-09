"""Replay a human .bk2 gameplay recording and report level outcome + duration.

Each CNeuroMod mario `.bk2` is one human play of one level, recorded at the
native 60 Hz (it may span several lives). We replay the exact recorded button
presses through the *same* stable-retro SuperMarioBros-Nes integration the
MuZero agent trains on, and use the in-game flagpole signal for a clear:

    completed  <=>  player_state enters {4, 5} (flagpole slide + walk-to-castle)
                    at any frame of the rep.

This is more robust than the scenario's `stage`-delta `done` (used by the
agent's env): CNeuroMod reps frequently end the moment Mario grabs the flag,
before the level counter ticks over to the next stage, so a pure `stage` test
misses real clears. The flagpole player-states fire exactly at the flag and
never appear in a death rep. A rep may span several lives (the human keeps
playing until game-over or the fixed recording window ends), so we do NOT treat
a life loss as failure — only "did they reach the flag".

Outputs one dict per .bk2: level, completed, n_frames, duration_s, max_x, ...
"""
from __future__ import annotations

import re
from pathlib import Path

import retro

LEVEL_RE = re.compile(r"level-w(\d+)l(\d+)")


def level_from_name(bk2_path: str) -> str | None:
    m = LEVEL_RE.search(Path(bk2_path).name)
    return f"Level{int(m.group(1))}-{int(m.group(2))}" if m else None


def replay_bk2(bk2_path: str, int_path: str) -> dict:
    retro.data.Integrations.add_custom_path(str(int_path))
    movie = retro.Movie(str(bk2_path))
    movie.step()  # primes the movie; loads the embedded game + start state
    env = retro.make(
        movie.get_game(),
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=None,
    )
    try:
        env.initial_state = movie.get_state()
        env.reset()

        n_buttons = env.num_buttons
        frames = 0
        completed = False
        game_over = False
        max_x = 0
        max_score = 0  # in-game SMB score is cumulative within a rep; take peak
        last_info: dict = {}

        while movie.step():
            keys = [movie.get_key(i, p) for p in range(movie.players) for i in range(n_buttons)]
            obs, reward, terminated, truncated, info = env.step(keys)
            frames += 1
            last_info = info
            if int(info.get("player_state", -1)) in (4, 5):
                completed = True
            x_pos = 256 * int(info.get("player_x_posHi", 0)) + int(info.get("player_x_posLo", 0))
            max_x = max(max_x, x_pos)
            max_score = max(max_score, int(info.get("score", 0)))
            if int(info.get("lives", 0)) == -1:
                game_over = True
                break
            # NB: do not break on `completed` — the flagpole + time bonus is
            # tallied over the ~650 recorded frames *after* the flag is grabbed,
            # so we must replay to the movie's end to capture the real score.

        return {
            "level": level_from_name(bk2_path),
            "completed": bool(completed),
            "game_over": bool(game_over),
            "n_frames": frames,
            "duration_s": frames / 60.0,
            "max_x": max_x,
            "score": max_score,
            "final_lives": int(last_info.get("lives", 0)),
        }
    finally:
        env.close()


if __name__ == "__main__":
    import json
    import sys

    int_path = sys.argv[1] if len(sys.argv) > 1 else "mario.stimuli"
    for bk2 in sys.argv[2:]:
        print(json.dumps({"bk2": Path(bk2).name, **replay_bk2(bk2, int_path)}))
