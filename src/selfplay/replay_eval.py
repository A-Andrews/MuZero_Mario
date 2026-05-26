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
) -> Tuple[List[np.ndarray], float, int]:
    """Return (raw_rgb_frames, total_return, num_env_steps).

    If `bk2_path` is provided, the underlying retro emulator records a `.bk2`
    movie of the rollout. The recording starts after `env.reset()` so the
    initial state is captured; playback (`python -m retro.scripts.playback_movie
    <file>.bk2`) replays the exact button-press sequence at native 60 Hz.
    """
    env = create_train_env(
        level=level,
        int_path=int_path,
        player_actions=None,
        n_frame=n_frame_stack,
        downsample=frame_skip,
        pad_to=pad_to,
        seed=seed,
    )
    recording = False
    rec_env = None
    mcts = MCTS(
        discount=discount,
        num_simulations=num_simulations,
        root_dirichlet_alpha=0.0,
        root_exploration_eps=0.0,
        pb_c_base=pb_c_base,
        pb_c_init=pb_c_init,
        device=device,
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
    try:
        while not done and step < max_steps:
            with torch.no_grad():
                action, _, _ = mcts.run(obs, network, temperature=0.0, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_return += float(reward)
            frames.append(env.latest_raw_rgb().copy())
            step += 1
    finally:
        if recording and rec_env is not None:
            try:
                rec_env.stop_record()
            except Exception:
                pass
        env.close()
    return frames, total_return, step
