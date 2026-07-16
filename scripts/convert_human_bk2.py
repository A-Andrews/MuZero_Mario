"""Convert CNeuroMod human .bk2 recordings into MuZero Trajectory .npz files.

Each .bk2 is an exact 60 Hz button log of one human play of one level. We
replay it through the *same* stable-retro integration, preprocessing
(grayscale 84x84, frame_skip max-pool stack, pad to 96) and reward shaping as
`src.env.env.CustomWrapper`, so the resulting (obs, action, reward) tuples are
byte-compatible with what the self-play workers emit — ready to seed the
replay buffer or train a behaviour-cloning loss.

Fidelity rules:
- Emulation is frame-exact: the recorded per-frame keys are fed to the
  emulator untouched. Only the *label* is quantised: each frame_skip window's
  majority button vector is mapped to the nearest of the 12 COMPLEX_MOVEMENT
  actions (Hamming distance, ties -> fewer buttons).
- Training uses done_on_life_loss=true, so a multi-life rep is split into one
  trajectory per life: death ends a segment (terminal), the respawn (after the
  death animation, player_state==8) starts a new one with a freshly seeded
  frame stack — extra start-state diversity, like ppo_study's action-skip
  spawns.
- Completion is the flagpole player_state (4, 5), as in
  analysis/comparison/replay_bk2.py: the rep often ends before the scenario's
  stage counter ticks. Completion adds env.completion_bonus and ends the file
  (the remaining frames are the score tally).
- No MCTS ran, so `policies` is the one-hot human action, `root_values` is set
  equal to the n-step returns (with reanalyze on, the learner recomputes value
  targets anyway) and `priorities` is uniform (1.0).

Usage:
    python scripts/convert_human_bk2.py \
        --mario-root ~/data/mario --out outputs/human_trajectories \
        [--int-path mario.stimuli] [--levels Level1-1,Level1-2] [--jobs 8]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.mario_actions import N_ACTIONS, complex_movement_to_button_presses
from src.env.preprocess import grayscale_resize, stack_max_pooled
from src.muzero.returns import compute_n_step_returns

LEVEL_RE = re.compile(r"level-w(\d+)l(\d+)")
# Same defaults as conf/env/mario.yaml + conf/muzero.yaml.
N_FRAME = 4
FRAME_SKIP = 4
PAD_TO = 96
COMPLETION_BONUS = 100.0
DISCOUNT = 0.997
N_STEP = 10
# The agent's 6-button vector is [run(B), up, down, left, right, jump(A)].
AGENT_BUTTONS = ["B", "UP", "DOWN", "LEFT", "RIGHT", "A"]
# player_state values (RAM 0x000E): 8 = normal control, 4/5 = flagpole slide
# + walk-to-castle, 11 = dying animation.
PS_FLAGPOLE = (4, 5)
PS_NORMAL = 8
RESPAWN_SKIP_CAP = 600  # frames; safety cap while waiting for control back


def level_from_name(bk2_path: str) -> str | None:
    m = LEVEL_RE.search(Path(bk2_path).name)
    return f"Level{int(m.group(1))}-{int(m.group(2))}" if m else None


def _candidate_table() -> np.ndarray:
    return np.stack(
        [complex_movement_to_button_presses(a) for a in range(N_ACTIONS)]
    ).astype(bool)


CANDIDATES = _candidate_table()
CANDIDATE_SIZES = CANDIDATES.sum(axis=1)


def quantize_buttons(pressed: np.ndarray) -> int:
    """Nearest COMPLEX_MOVEMENT action to a human 6-button vector.

    Hamming distance over the 6 buttons; ties broken toward the action with
    fewer buttons (then lower index), so e.g. right+down maps to plain right
    rather than gaining a phantom jump.
    """
    dist = (CANDIDATES ^ pressed[None, :]).sum(axis=1)
    order = np.lexsort((np.arange(N_ACTIONS), CANDIDATE_SIZES, dist))
    return int(order[0])


class _SegmentBuilder:
    """Accumulates one done_on_life_loss-style episode."""

    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []

    def add(self, obs_f32: np.ndarray, action: int, reward: float):
        self.obs.append((obs_f32 * 255.0).clip(0, 255).astype(np.uint8))
        self.actions.append(action)
        self.rewards.append(reward)

    def __len__(self):
        return len(self.actions)

    def finalise(self, level: str, terminal: bool, completed: bool, meta: dict):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        # No MCTS values exist; use n-step returns as the value proxy (see
        # module docstring). Bootstrap values are the recursive returns
        # themselves, so compute MC-style with root_values=0 first, then use
        # those as the bootstrap for the n-step form the buffer expects.
        mc = compute_n_step_returns(
            rewards, np.zeros_like(rewards), n_step=len(rewards), discount=DISCOUNT,
            terminal=terminal,
        )
        returns = compute_n_step_returns(
            rewards, mc, n_step=N_STEP, discount=DISCOUNT, terminal=terminal
        )
        actions = np.asarray(self.actions, dtype=np.int64)
        policies = np.zeros((len(actions), N_ACTIONS), dtype=np.float32)
        policies[np.arange(len(actions)), actions] = 1.0
        return {
            "obs_stacks": np.stack(self.obs, axis=0),
            "actions": actions,
            "rewards": rewards,
            "policies": policies,
            "root_values": returns,
            "returns": returns,
            "priorities": np.ones_like(rewards),
            "level": level,
            "terminal": terminal,
            "completed": completed,
            **{k: str(v) for k, v in meta.items()},
        }


def convert_bk2(bk2_path: str, int_path: str) -> list[dict]:
    """Replay one .bk2; return a list of Trajectory-field dicts (segments)."""
    import cv2
    import retro  # local imports: safe under multiprocessing spawn

    cv2.setNumThreads(0)  # one env per worker process; don't oversubscribe

    retro.data.Integrations.add_custom_path(str(Path(int_path).resolve()))
    movie = retro.Movie(str(bk2_path))
    movie.step()
    env = retro.make(
        movie.get_game(),
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=None,
    )
    level = level_from_name(bk2_path)
    meta = {"bk2": Path(bk2_path).name}
    segments: list[dict] = []
    try:
        env.initial_state = movie.get_state()
        obs, _ = env.reset()
        button_idx = np.array([env.buttons.index(b) for b in AGENT_BUTTONS])
        n_buttons = env.num_buttons

        frame_stack: deque = deque(maxlen=N_FRAME * FRAME_SKIP)

        def reseed(rgb):
            gray = grayscale_resize(rgb)
            frame_stack.clear()
            for _ in range(N_FRAME * FRAME_SKIP):
                frame_stack.append(gray)

        def stack_obs():
            return stack_max_pooled(frame_stack, N_FRAME, FRAME_SKIP, pad_to=PAD_TO)

        seg = _SegmentBuilder()
        curr_lives, curr_score, last_x, last_time = None, 0.0, 0, None
        movie_alive = True

        # Human reps open on the title card / spawn-in (player_state 0/7,
        # ~120 frames) while the agent's .state files start in gameplay —
        # skip to the first controllable frame so segment starts match.
        rgb = obs
        skipped = 0
        while skipped < RESPAWN_SKIP_CAP:
            if not movie.step():
                movie_alive = False
                break
            keys = [movie.get_key(i, 0) for i in range(n_buttons)]
            rgb, _r, _term, _trunc, info = env.step(keys)
            skipped += 1
            if int(info.get("player_state", -1)) == PS_NORMAL:
                break
        reseed(rgb)

        while movie_alive:
            s_t = stack_obs()
            window_keys = np.zeros((0, 6), dtype=bool)
            total_reward = 0.0
            seg_end = None  # None | "death" | "flag" | "gameover"

            for _ in range(FRAME_SKIP):
                if not movie.step():
                    movie_alive = False
                    break
                keys = [movie.get_key(i, 0) for i in range(n_buttons)]
                rgb, _r, _term, _trunc, info = env.step(keys)
                frame_stack.append(grayscale_resize(rgb))
                window_keys = np.vstack(
                    [window_keys, np.asarray(keys, dtype=bool)[button_idx]]
                )

                if curr_lives is None:  # first frame: init shaping state
                    curr_lives = int(info["lives"])
                    curr_score = float(info["score"])
                    last_x = 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])

                # -- reward shaping, mirroring CustomWrapper.step ------------
                reward = 0.0
                if last_time is not None:
                    reward += min(info["time"] - last_time, 0)
                last_time = info["time"]
                x_pos = 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
                diff_x = x_pos - last_x
                reward += diff_x if -5 <= diff_x <= 5 else 0
                last_x = x_pos
                died = int(info["lives"]) < curr_lives
                if died:
                    reward -= 15
                    curr_lives = int(info["lives"])
                    seg_end = "gameover" if int(info["lives"]) < 0 else "death"
                reward = max(min(reward, 15), -15)
                reward += min((float(info["score"]) - curr_score) / 4.0, 50)
                curr_score = float(info["score"])
                if int(info.get("player_state", -1)) in PS_FLAGPOLE and seg_end is None:
                    reward += COMPLETION_BONUS
                    seg_end = "flag"
                if seg_end == "gameover":
                    reward -= 50
                total_reward += reward
                if seg_end is not None:
                    break

            if len(window_keys) == 0:  # movie ended exactly on a window edge
                break

            majority = window_keys.sum(axis=0) * 2 >= len(window_keys)
            seg.add(s_t, quantize_buttons(majority), total_reward / 10.0)

            if seg_end in ("death", "flag", "gameover"):
                if len(seg):
                    segments.append(
                        seg.finalise(level, terminal=True,
                                     completed=(seg_end == "flag"), meta=meta)
                    )
                seg = _SegmentBuilder()
                if seg_end in ("flag", "gameover"):
                    break  # rest of the rep is score tally / menu
                # Respawn: replay through the death animation until control
                # returns, then start a fresh segment (fresh frame stack).
                skipped = 0
                while movie_alive and skipped < RESPAWN_SKIP_CAP:
                    if not movie.step():
                        movie_alive = False
                        break
                    keys = [movie.get_key(i, 0) for i in range(n_buttons)]
                    rgb, _r, _term, _trunc, info = env.step(keys)
                    skipped += 1
                    if int(info.get("player_state", -1)) == PS_NORMAL:
                        break
                if not movie_alive:
                    break
                reseed(rgb)
                last_x = 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
                last_time = info["time"]
                curr_score = float(info["score"])

        # Movie ran out without death/flag: truncated segment.
        if len(seg):
            segments.append(
                seg.finalise(level, terminal=False, completed=False, meta=meta)
            )
    finally:
        env.close()
    return segments


def _convert_and_save(args_tuple):
    bk2_path, int_path, out_dir = args_tuple
    try:
        segments = convert_bk2(bk2_path, int_path)
    except Exception as e:  # one bad rep must not kill the batch
        return {"bk2": Path(bk2_path).name, "error": repr(e)}
    stem = Path(bk2_path).stem
    saved = []
    for k, seg in enumerate(segments):
        path = Path(out_dir) / f"{stem}_seg{k}.npz"
        np.savez_compressed(path, **seg)
        saved.append(
            {"file": path.name, "steps": int(seg["actions"].shape[0]),
             "completed": bool(seg["completed"]), "terminal": bool(seg["terminal"])}
        )
    return {"bk2": Path(bk2_path).name, "segments": saved}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mario-root", required=True,
                    help="datalad clone of courtois-neuromod/mario")
    ap.add_argument("--int-path", default="mario.stimuli")
    ap.add_argument("--out", default="outputs/human_trajectories")
    ap.add_argument("--levels", default=None,
                    help="comma-separated Level<w>-<l> filter (default: all)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="convert at most N reps (for smoke tests)")
    args = ap.parse_args()

    # Annexed-but-not-downloaded files are dangling symlinks; drop them.
    bk2s = sorted(
        p for p in Path(args.mario_root).glob("sub-*/ses-*/gamelogs/*.bk2")
        if p.resolve().exists()
    )
    if args.levels:
        wanted = set(args.levels.split(","))
        bk2s = [p for p in bk2s if level_from_name(str(p)) in wanted]
    if args.limit:
        bk2s = bk2s[: args.limit]
    if not bk2s:
        sys.exit("no .bk2 files found (downloaded + matching filter)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"converting {len(bk2s)} reps -> {out_dir}  (jobs={args.jobs})")

    work = [(str(p), args.int_path, str(out_dir)) for p in bk2s]
    results = []
    with Pool(args.jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(_convert_and_save, work), 1):
            results.append(res)
            if i % 100 == 0 or i == len(work):
                print(f"  {i}/{len(work)}")

    errors = [r for r in results if "error" in r]
    seg_counts = [len(r.get("segments", [])) for r in results if "segments" in r]
    steps = sum(s["steps"] for r in results for s in r.get("segments", []))
    completed = sum(s["completed"] for r in results for s in r.get("segments", []))
    print(f"reps: {len(results)}  segments: {sum(seg_counts)}  "
          f"agent-steps: {steps}  completions: {completed}  errors: {len(errors)}")
    with open(out_dir / "conversion_report.json", "w") as f:
        json.dump(results, f, indent=1)
    if errors:
        print("first errors:", errors[:3])


if __name__ == "__main__":
    main()
