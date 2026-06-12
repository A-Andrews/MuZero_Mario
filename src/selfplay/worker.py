"""Self-play worker process.

Each worker plays Mario via gym-retro and runs MCTS through the central
inference server. Per-episode loop:

    1. (autocurriculum) sample the next level from shared weights, swap env
       if needed
    2. reset env
    3. for each step: run MCTS, take action, accumulate trajectory data
    4. on episode end, push a ``Trajectory`` + a status dict to the learner

Between episodes the worker re-reads the level-sampling weights array; weights
are pushed by the learner asynchronously, so a worker only ever sees a
consistent snapshot, never a partial update.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.env.env import create_train_env
from src.muzero.buffer import Trajectory
from src.muzero.mcts import MCTS
from src.muzero.returns import compute_n_step_returns
from src.muzero.temperature import temperature_for_step
from src.selfplay.inference_server import RemoteNetwork


def _sample_level(
    levels: Sequence[str],
    weights_array,
    rng: np.random.Generator,
) -> str:
    """Sample a level from the shared mp.Array of weights.

    Snapshot under the array's lock so the worker never sees a half-written
    update. Falls back to uniform if the array is degenerate (all zero / NaN).
    """
    with weights_array.get_lock():
        w = np.array(weights_array[:], dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    s = w.sum()
    if s <= 0.0:
        return levels[int(rng.integers(len(levels)))]
    p = w / s
    return levels[int(rng.choice(len(levels), p=p))]


def selfplay_worker(
    worker_id: int,
    initial_level: str,
    levels: List[str],
    level_weights,           # shared mp.Array(c_double, num_levels)
    autocurriculum_enabled: bool,
    cfg_pkl: bytes,
    request_queue,           # shared mp.Queue to the inference server
    reply_conn,              # per-worker mp.Pipe reader for server replies
    traj_queue,              # send completed Trajectory
    status_queue,            # send episode-level scalar logs (dict)
    train_step_val,          # shared int (Value) controlled by learner
):
    """Entrypoint for a self-play child process.

    ``cfg_pkl`` is a pickled dict holding the subset of config the worker needs
    (config objects are not always picklable through Hydra's proxy types).
    """
    cfg = pickle.loads(cfg_pkl)
    torch.set_num_threads(max(1, int(cfg["worker"]["torch_threads"])))
    seed = cfg["seed"] + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cpu")

    def _make_env(lv: str, env_seed: int):
        return create_train_env(
            level=lv,
            int_path=cfg["env"]["int_path"],
            player_actions=None,
            n_frame=cfg["env"]["n_frame_stack"],
            downsample=cfg["env"]["frame_skip"],
            pad_to=cfg["model"]["input_spatial"] if cfg["env"]["pad_to_input_spatial"] else None,
            seed=env_seed,
            done_on_life_loss=bool(cfg["env"].get("done_on_life_loss", True)),
            completion_bonus=float(cfg["env"].get("completion_bonus", 100.0)),
        )

    current_level = initial_level
    env = _make_env(current_level, seed)

    net = RemoteNetwork(
        worker_id=worker_id,
        request_queue=request_queue,
        reply_conn=reply_conn,
        wire_dtype=str(cfg.get("inference_server", {}).get("wire_dtype", "float32")),
    )

    mcts = MCTS(
        discount=cfg["muzero"]["discount"],
        num_simulations=cfg["mcts"]["num_simulations"],
        root_dirichlet_alpha=cfg["mcts"]["dirichlet_alpha"],
        root_exploration_eps=cfg["mcts"]["root_exploration_eps"],
        pb_c_base=cfg["mcts"]["pb_c_base"],
        pb_c_init=cfg["mcts"]["pb_c_init"],
        device=device,
    )

    temperature_schedule = list(cfg["selfplay"]["temperature_schedule"])

    obs = env.reset()
    ep_obs, ep_actions, ep_rewards, ep_policies, ep_root_q = [], [], [], [], []
    ep_steps = 0
    ep_return = 0.0

    max_traj_len = int(cfg["selfplay"]["max_trajectory_length"])

    while True:
        step = int(train_step_val.value)
        temperature = temperature_for_step(step, temperature_schedule)

        action, pi_prob, root_q = mcts.run(
            obs, net, temperature=temperature, deterministic=False
        )

        next_obs, reward, done, info = env.step(action)

        # Store obs as uint8 [0, 255] to save memory (float obs lives in [0,1]).
        ep_obs.append((obs * 255.0).clip(0, 255).astype(np.uint8))
        ep_actions.append(int(action))
        ep_rewards.append(float(reward))
        ep_policies.append(pi_prob.astype(np.float32))
        ep_root_q.append(float(root_q))
        ep_steps += 1
        ep_return += float(reward)

        obs = next_obs

        if done or ep_steps >= max_traj_len:
            traj = _finalise_trajectory(
                level=current_level,
                obs_list=ep_obs,
                actions=ep_actions,
                rewards=ep_rewards,
                policies=ep_policies,
                root_q=ep_root_q,
                discount=cfg["muzero"]["discount"],
                n_step=cfg["muzero"]["n_step"],
                terminal=bool(done),
            )
            traj_queue.put(traj)
            try:
                status_queue.put(
                    {
                        "worker_id": worker_id,
                        "level": current_level,
                        "episode_return": ep_return,
                        "episode_length": ep_steps,
                        "final_x_pos": int(
                            256 * int(info["player_x_posHi"])
                            + int(info["player_x_posLo"])
                        ),
                        "mcts_root_q_mean": float(np.mean(ep_root_q)) if ep_root_q else 0.0,
                        "completed": bool(info.get("level_complete", False)),
                        "train_step": step,
                    }
                )
            except Exception:
                pass

            ep_obs, ep_actions, ep_rewards, ep_policies, ep_root_q = [], [], [], [], []
            ep_steps = 0
            ep_return = 0.0

            # Pick the next level under the current curriculum, then either
            # keep the existing env or close+rebuild for the new level. Retro
            # envs are tied to a single state at make-time so we cannot just
            # reset across levels.
            if autocurriculum_enabled:
                next_level = _sample_level(levels, level_weights, rng)
            else:
                next_level = current_level
            if next_level != current_level:
                try:
                    env.close()
                except Exception:
                    pass
                current_level = next_level
                env = _make_env(current_level, seed)
            obs = env.reset()


def _finalise_trajectory(
    level, obs_list, actions, rewards, policies, root_q, discount, n_step, terminal
) -> Trajectory:
    obs_arr = np.stack(obs_list, axis=0)  # (T, C, H, W) uint8
    actions_arr = np.asarray(actions, dtype=np.int64)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    policies_arr = np.stack(policies, axis=0).astype(np.float32)
    root_q_arr = np.asarray(root_q, dtype=np.float32)
    returns = compute_n_step_returns(
        rewards_arr, root_q_arr, n_step=n_step, discount=discount, terminal=terminal
    )
    priorities = np.abs(returns - root_q_arr).astype(np.float32) + 1e-3
    return Trajectory(
        obs_stacks=obs_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        policies=policies_arr,
        root_values=root_q_arr,
        returns=returns,
        priorities=priorities,
        level=level,
    )
