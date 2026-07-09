"""Retro + mario.stimuli environment wrapper (ported from ppo_study/src/ppo/env.py).

Changes from the PPO version:
- `player_actions` is optional; when None the env spawns at the level start (no
  random initial action skip).
- `reset` and `step` return a stacked `(n_frame, H, W)` numpy array instead of a
  list of frames, optionally padded to `pad_to` to match the MuZero-Atari
  downsample tower (96x96 -> 6x6 with 4 stride-2 stages).
- `MultipleEnvironments` keeps the `mp.Pipe` pattern but also accepts a
  `("raw_obs", None)` request (used by replay_eval) and a
  `("close", None)` shutdown.
"""
import retro
import numpy as np
import torch.multiprocessing as mp
from collections import deque

from src.env.emulation import add_unused_buttons
from src.env.mario_actions import complex_movement_to_button_presses
from src.env.preprocess import grayscale_resize, stack_max_pooled


def create_train_env(
    level,
    int_path,
    player_actions=None,
    n_frame=4,
    downsample=4,
    pad_to=None,
    seed=2024,
    done_on_life_loss=True,
    completion_bonus=100.0,
):
    world = level[5]  # "Level1-1" -> "1"
    stage = level[7]
    retro.data.Integrations.add_custom_path(str(int_path))
    env = retro.make(
        "SuperMarioBros-Nes",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        state=f"Level{world}-{stage}",
        render_mode=None,
    )
    return CustomWrapper(
        env,
        player_actions=player_actions,
        n_frame=n_frame,
        downsample=downsample,
        pad_to=pad_to,
        seed=seed,
        done_on_life_loss=done_on_life_loss,
        completion_bonus=completion_bonus,
    )


class CustomWrapper:
    """Preprocess retro frames + shape reward for Mario.

    Reward shaping (base scheme identical to ppo_study):
        - time penalty: negative delta in info["time"]
        - movement reward: delta x clipped to [-5, 5]
        - per-step clip [-15, 15]
        - death: -15 per life lost, -50 on game over
        - score: +min(delta_score/4, 50)
        - level completion (stage counter advanced): +completion_bonus
        - final: sum / 10

    When ``done_on_life_loss`` is True the episode terminates on the *first*
    life lost instead of playing all three lives to game-over. This makes
    death a true terminal (bootstrap value 0) and keeps credit assignment
    crisp; the retro scenario's own done (stage advance / game over) still
    applies.
    """

    def __init__(
        self,
        env,
        player_actions=None,
        n_frame=4,
        downsample=4,
        pad_to=None,
        seed=2024,
        done_on_life_loss=True,
        completion_bonus=100.0,
    ):
        self.env = env
        self.n_frame = n_frame
        self.downsample = downsample
        self.pad_to = pad_to
        self.player_actions = player_actions
        self.done_on_life_loss = bool(done_on_life_loss)
        self.completion_bonus = float(completion_bonus)
        # Grayscale 84x84 frames are cached here; we only preprocess each raw
        # frame once (on push) rather than re-grayscaling the whole stack on
        # every _obs() call.
        self.frame_stack = deque([], maxlen=n_frame * downsample)
        self.latest_rgb = None  # most recent raw RGB frame for video capture
        self.curr_score = 0.0
        self.curr_lives = 2
        self.last_x = 0
        self.last_time = None
        self.last_info = {}
        self.rng = np.random.default_rng(seed=seed)

    # -- observation helper -------------------------------------------------

    def _push_frame(self, raw_rgb):
        self.latest_rgb = raw_rgb
        self.frame_stack.append(grayscale_resize(raw_rgb))

    def _obs(self):
        return stack_max_pooled(
            self.frame_stack, self.n_frame, self.downsample, pad_to=self.pad_to
        )

    def latest_raw_rgb(self):
        """Return the most recently observed raw RGB frame (for replay videos)."""
        return np.asarray(self.latest_rgb, dtype=np.uint8)

    # -- gym API ------------------------------------------------------------

    def step(self, action):
        action = complex_movement_to_button_presses(action)
        action = add_unused_buttons(list(action))
        total_reward = 0.0
        info = {}
        done = False
        for _ in range(self.downsample):
            obs, _reward, term, trunc, info = self.env.step(action)
            self._push_frame(obs)
            scenario_done = term or trunc
            done = scenario_done
            reward = 0.0
            if self.last_time is not None:
                reward += min(info["time"] - self.last_time, 0)
            self.last_time = info["time"]
            x_pos = 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
            diff_x = x_pos - self.last_x
            reward += diff_x if -5 <= diff_x <= 5 else 0
            self.last_x = x_pos
            died = info["lives"] < self.curr_lives
            if died:
                reward -= 15
                self.curr_lives = info["lives"]
                if self.done_on_life_loss:
                    done = True
            reward = max(min(reward, 15), -15)
            reward += min((info["score"] - self.curr_score) / 4.0, 50)
            self.curr_score = info["score"]
            if done:
                # The retro scenario fires done on stage-counter advance
                # (level complete) or game over; with lives intact and no
                # death this tick, the only scenario-done cause is completion.
                completed = scenario_done and not died and info["lives"] >= 0
                if completed:
                    reward += self.completion_bonus
                if info["lives"] < 0:
                    reward -= 50
                info["level_complete"] = completed
                total_reward += reward
                break
            total_reward += reward
        self.last_info = info
        obs = self._obs()
        return obs, total_reward / 10.0, done, info

    def reset(self):
        self.curr_lives = 2
        self.curr_score = 0
        self.last_x = 0
        self.last_time = None
        obs, _ = self.env.reset()
        # Cache the grayscaled initial frame once and duplicate it across the
        # stack, rather than running grayscale/resize n_frame*downsample times.
        self.latest_rgb = obs
        gray = grayscale_resize(obs)
        self.frame_stack.clear()
        for _ in range(self.n_frame * self.downsample):
            self.frame_stack.append(gray)
        if self.player_actions is not None and len(self.player_actions) > 0:
            n_actions = self.rng.integers(int(0.8 * len(self.player_actions)))
            for i in range(n_actions):
                obs, _, _term, _trunc, info = self.env.step(self.player_actions[i])
                self._push_frame(obs)
                self.last_info = info
        return self._obs()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# MultipleEnvironments: one process per level with two-way pipe.
# -----------------------------------------------------------------------------


def _env_worker(
    conn,
    level,
    int_path,
    player_actions,
    n_frame,
    downsample,
    pad_to,
    seed,
):
    """Child-process loop: owns one CustomWrapper and serves requests over a pipe."""
    env = create_train_env(
        level=level,
        int_path=int_path,
        player_actions=player_actions,
        n_frame=n_frame,
        downsample=downsample,
        pad_to=pad_to,
        seed=seed,
    )
    try:
        while True:
            request, payload = conn.recv()
            if request == "step":
                conn.send(env.step(payload))
            elif request == "reset":
                conn.send(env.reset())
            elif request == "raw_obs":
                conn.send(env.latest_raw_rgb())
            elif request == "close":
                conn.send(None)
                break
            else:
                conn.send(("err", f"unknown request {request}"))
    finally:
        env.close()
        conn.close()


class MultipleEnvironments:
    """Spawn one process per level; communicate over `mp.Pipe`.

    Supports messages: ("reset", None), ("step", action), ("raw_obs", None),
    ("close", None).
    """

    def __init__(
        self,
        levels,
        int_path,
        player_actions_per_level=None,
        n_frame=4,
        downsample=4,
        pad_to=None,
        seed=2024,
    ):
        self.levels = list(levels)
        self.num_envs = len(self.levels)
        self.num_actions = 12
        if player_actions_per_level is None:
            player_actions_per_level = {lvl: None for lvl in self.levels}
        ctx = mp.get_context("spawn")
        self.parent_conns = []
        self.processes = []
        for i, level in enumerate(self.levels):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_env_worker,
                args=(
                    child_conn,
                    level,
                    int_path,
                    player_actions_per_level.get(level),
                    n_frame,
                    downsample,
                    pad_to,
                    seed + i,
                ),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.processes.append(p)

    def reset(self, idx):
        self.parent_conns[idx].send(("reset", None))
        return self.parent_conns[idx].recv()

    def step(self, idx, action):
        self.parent_conns[idx].send(("step", int(action)))
        return self.parent_conns[idx].recv()

    def raw_obs(self, idx):
        self.parent_conns[idx].send(("raw_obs", None))
        return self.parent_conns[idx].recv()

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
                conn.recv()
            except Exception:
                pass
            conn.close()
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
