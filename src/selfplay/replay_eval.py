"""Record a greedy-MCTS replay of one level at checkpoint time.

The replay env is built fresh in-process (NOT via MultipleEnvironments — those
are busy running self-play). We grab the raw 240x256 RGB frame from the
underlying retro env after each step so the resulting mp4 is viewable in wandb
(the preprocessed 84x84 is too small/ugly).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.env.env import create_train_env
from src.muzero.mcts import MCTS


def run_replay_rollout(
    level: str,
    int_path,
    network,
    device,
    num_simulations: int,
    discount: float,
    pb_c_base: float,
    pb_c_init: float,
    n_frame_stack: int,
    frame_skip: int,
    pad_to: int,
    max_steps: int = 5000,
    seed: int = 2024,
    bk2_path: Optional[Path] = None,
    leaf_batch: int = 1,
    temperature: float = 0.0,
    root_dirichlet_alpha: float = 0.0,
    root_exploration_eps: float = 0.0,
    np_seed: Optional[int] = None,
    info_out: Optional[dict] = None,
    noop_max: int = 0,
    skip_to_control: bool = False,
    completion_bonus: float = 100.0,
) -> Tuple[List[np.ndarray], float, int, bool]:
    """Return (raw_rgb_frames, total_return, num_env_steps, level_completed).

    If `bk2_path` is provided, the underlying retro emulator records a `.bk2`
    movie of the rollout. The recording starts after `env.reset()` so the
    initial state is captured; playback (`python -m retro.scripts.playback_movie
    <file>.bk2`) replays the exact button-press sequence at native 60 Hz.

    Search stochasticity is controlled by two *independent* knobs, because the
    self-play-vs-greedy completion gap turned out to hinge on the noise one:
    `root_exploration_eps`/`root_dirichlet_alpha` perturb the root prior (both
    must be > 0 to take effect), and `temperature` shapes action selection
    (<= 0 means argmax over visit counts). Defaults reproduce the original
    fully-greedy checkpoint replay: no noise, argmax.

    `np_seed` seeds numpy so a noisy rollout is reproducible; `info_out`, if
    given, is filled with extra per-rollout diagnostics (`final_x`, `timed_out`).

    `completion_bonus` must be passed the run's `env.completion_bonus` for the
    returned total to be on the same reward scale as `selfplay/episode_return`;
    it defaults to the env's own 100.0 default, which is what every call site
    used before — so runs training at a different bonus (the curriculum recipe
    uses 200) had `replay/<level>_return` silently 10 units short on a
    completing rollout.

    `noop_max`/`skip_to_control` are the env's stochastic-start knobs and
    default to **off** here even when training has them on, so a greedy eval
    stays a single reproducible trajectory and stays comparable to the
    deterministic-env diag baselines. Turn them on to measure a policy under
    the same start distribution it was trained on.
    """
    if np_seed is not None:
        np.random.seed(int(np_seed))
    env = create_train_env(
        level=level,
        int_path=int_path,
        player_actions=None,
        n_frame=n_frame_stack,
        downsample=frame_skip,
        pad_to=pad_to,
        seed=seed,
        noop_max=noop_max,
        skip_to_control=skip_to_control,
        completion_bonus=completion_bonus,
    )
    recording = False
    rec_env = None
    mcts = MCTS(
        discount=discount,
        num_simulations=num_simulations,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_eps=root_exploration_eps,
        pb_c_base=pb_c_base,
        pb_c_init=pb_c_init,
        device=device,
        leaf_batch=leaf_batch,
    )
    network.eval()

    frames: List[np.ndarray] = []
    obs = env.reset()
    # record_movie snapshots the emulator's current state as the bk2 starting
    # point, so it must come after reset() and before the first step().
    if bk2_path is not None:
        rec_env = getattr(env, "env", None)
        if rec_env is not None and hasattr(rec_env, "record_movie"):
            try:
                bk2_path = Path(bk2_path)
                bk2_path.parent.mkdir(parents=True, exist_ok=True)
                rec_env.record_movie(str(bk2_path))
                recording = True
            except Exception as e:
                print(f"[replay] bk2 recording unavailable: {e}")
        else:
            print("[replay] bk2 recording unsupported by env wrapper; skipping")
    # Seed the video with the first raw RGB frame so single-step deaths don't
    # produce an empty video.
    frames.append(env.latest_raw_rgb().copy())

    total_return = 0.0
    step = 0
    done = False
    completed = False
    final_x = 0
    try:
        while not done and step < max_steps:
            with torch.no_grad():
                # deterministic=False so the noise and temperature knobs above
                # are what decide stochasticity: eps/alpha of 0 disable the
                # root noise, temperature <= 0 falls back to argmax on visits.
                action, _, _ = mcts.run(obs, network, temperature=temperature, deterministic=False)
            obs, reward, done, info = env.step(action)
            total_return += float(reward)
            frames.append(env.latest_raw_rgb().copy())
            step += 1
            if "player_x_posHi" in info:
                final_x = 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
            if done:
                completed = bool(info.get("level_complete", False))
        if info_out is not None:
            info_out["final_x"] = final_x
            info_out["timed_out"] = bool(not done and step >= max_steps)
    finally:
        if recording and rec_env is not None:
            try:
                rec_env.stop_record()
            except Exception:
                pass
        env.close()
    return frames, total_return, step, completed
