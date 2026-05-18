"""Self-play worker process.

Each worker owns one retro env (for one level) and a CPU copy of the MuZero
network. It runs an endless loop of:

    1. reset env (or continue from current state if not done)
    2. run MCTS with temperature set by the learner
    3. step env with the sampled action
    4. when episode ends, compute returns + priorities, push a `Trajectory` to
       the `traj_queue`

Between episodes the worker drains the `weights_queue` and loads the latest
network `state_dict()` from the learner.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.env.env import create_train_env
from src.muzero.buffer import Trajectory
from src.muzero.mcts import MCTS
from src.muzero.returns import compute_n_step_returns
from src.muzero.temperature import temperature_for_step
from src.selfplay.inference_server import RemoteNetwork


def selfplay_worker(
    worker_id: int,
    level: str,
    cfg_pkl: bytes,
    request_queue,    # shared mp.Queue to the inference server
    reply_conn,       # per-worker mp.Pipe reader for server replies
    traj_queue,       # send completed Trajectory
    status_queue,     # send episode-level scalar logs (dict)
    train_step_val,   # shared int (Value) controlled by learner
):
    """Entrypoint for a self-play child process.

    `cfg_pkl` is a pickled dict holding the subset of config the worker needs
    (config objects are not always picklable through Hydra's proxy types).
    """
    cfg = pickle.loads(cfg_pkl)
    torch.set_num_threads(max(1, int(cfg["worker"]["torch_threads"])))
    seed = cfg["seed"] + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")

    env = create_train_env(
        level=level,
        int_path=cfg["env"]["int_path"],
        player_actions=None,
        n_frame=cfg["env"]["n_frame_stack"],
        downsample=cfg["env"]["frame_skip"],
        pad_to=cfg["model"]["input_spatial"] if cfg["env"]["pad_to_input_spatial"] else None,
        seed=seed,
    )

    net = RemoteNetwork(
        worker_id=worker_id,
        request_queue=request_queue,
        reply_conn=reply_conn,
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
                level=level,
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
                        "level": level,
                        "episode_return": ep_return,
                        "episode_length": ep_steps,
                        "final_x_pos": int(
                            256 * int(info["player_x_posHi"])
                            + int(info["player_x_posLo"])
                        ),
                        "mcts_root_q_mean": float(np.mean(ep_root_q)) if ep_root_q else 0.0,
                        "train_step": step,
                    }
                )
            except Exception:
                pass

            ep_obs, ep_actions, ep_rewards, ep_policies, ep_root_q = [], [], [], [], []
            ep_steps = 0
            ep_return = 0.0
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
