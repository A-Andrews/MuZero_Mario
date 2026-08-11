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
    noop_max=0,
    skip_to_control=False,
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
        noop_max=noop_max,
        skip_to_control=skip_to_control,
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

    ``noop_max`` > 0 enables **stochastic starts**: each reset burns a uniform
    random 0..noop_max emulator frames on the NOOP action before handing back
    the first observation. NES Mario is otherwise fully deterministic, which
    made every eval cell a single trajectory rather than a sample (see T1 in
    BACKLOG.md) and never pressured the policy to be robust. Mario stands
    still during the delay, so this costs only those frames of the level
    timer, but it shifts his phase relative to the frame-counter-driven enemy
    animation — which is exactly the variation the Level1-2 Koopa cluster
    needs. 0 disables it and reproduces the old deterministic behaviour.

    ``skip_to_control`` is not optional decoration — without it ``noop_max``
    does **nothing**. Every level opens with a scripted intro that ignores
    input (measured: 123 frames on Level1-1, 117 on Level1-2, ``player_state``
    0 -> 7 -> 8), so a 0..30 frame delay lands entirely inside that window and
    is absorbed: the state at control onset is bit-identical whatever the
    draw. With this on, reset advances to ``player_state == 8`` ("player in
    control") first and randomises from there.

    Two consequences worth knowing. Those intro frames no longer appear in
    trajectories, so every level loses ~30 agent steps of uncontrollable
    title card per episode (~31 on Level1-1 at frame_skip 4) — which also
    feeds the autocurriculum's inverse-length weighting. And the variation
    the delay buys is largely *unobservable* to the agent: Mario stands still,
    so the frames barely change, but the emulator's frame counter advances and
    that is what drives enemy animation phase later in the level. That is the
    point — it turns a greedy rollout into a distribution instead of a point.
    """

    # Safety cap on the skip-to-control scan (~10s of emulation). No Mario
    # start state comes close; this only stops a pathological state file from
    # spinning reset forever.
    _SKIP_TO_CONTROL_MAX_FRAMES = 600

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
        noop_max=0,
        skip_to_control=False,
    ):
        self.env = env
        self.n_frame = n_frame
        self.downsample = downsample
        self.pad_to = pad_to
        self.player_actions = player_actions
        self.done_on_life_loss = bool(done_on_life_loss)
        self.completion_bonus = float(completion_bonus)
        self.noop_max = max(0, int(noop_max))
        self.skip_to_control = bool(skip_to_control)
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
        obs, info = self.env.reset()
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
        elif self.skip_to_control or self.noop_max > 0:
            self._stochastic_start(info)
        return self._obs()

    def _stochastic_start(self, reset_info):
        """Advance past any scripted intro, then burn a random NOOP delay.

        See the class docstring for why the two halves are separate. Both
        loops break on a terminal — standing still at a level start cannot
        kill Mario, but a state file that spawns him mid-fall would otherwise
        step a finished episode.
        """
        noop = add_unused_buttons(list(complex_movement_to_button_presses(0)))
        info = reset_info if isinstance(reset_info, dict) else {}
        done = False

        if self.skip_to_control:
            # `env.reset()` hands back an *empty* info dict, so the state can
            # only be read by stepping — this must be a do-while, not a while.
            for _ in range(self._SKIP_TO_CONTROL_MAX_FRAMES):
                obs, _reward, term, trunc, info = self.env.step(noop)
                self._push_frame(obs)
                if term or trunc:
                    done = True
                    break
                if "player_state" not in info:
                    break  # variable unavailable — don't burn the whole cap
                if int(info["player_state"]) == 8:
                    break

        if self.noop_max > 0 and not done:
            for _ in range(int(self.rng.integers(0, self.noop_max + 1))):
                obs, _reward, term, trunc, info = self.env.step(noop)
                self._push_frame(obs)
                if term or trunc:
                    break

        # Adopt the post-delay counters. Without this the first real step
        # charges the agent for the timer ticks that elapsed while it had no
        # control — a spurious negative reward proportional to the delay.
        if info:
            self.last_time = info["time"]
            self.last_x = 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
            self.curr_score = info["score"]
            self.curr_lives = info["lives"]
            self.last_info = info

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
