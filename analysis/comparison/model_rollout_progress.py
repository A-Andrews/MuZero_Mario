"""Greedy-MCTS rollout of the current-best checkpoint, tracking max-x progress.

For each level, runs one deterministic (temperature 0) MCTS rollout through the
same env the agent trains on and records how far it gets (max world-x reached),
whether it completed the level, episode return and length. This is the model
counterpart to the human .bk2 replays — same emulator, same x coordinate, same
completion criterion — so the two can share a "fraction of level reached" axis.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.checkpoint import load_checkpoint
from src.env.env import create_train_env
from src.muzero.mcts import MCTS
from src.muzero.networks import MuZeroNet


def build_net(model_cfg):
    return MuZeroNet(
        input_channels=model_cfg["input_channels"],
        input_spatial=model_cfg["input_spatial"],
        hidden_channels=model_cfg["hidden_channels"],
        hidden_spatial=model_cfg["hidden_spatial"],
        num_actions=model_cfg["num_actions"],
        value_support=tuple(model_cfg["value_support"]),
        reward_support=tuple(model_cfg["reward_support"]),
        rep_blocks=tuple(model_cfg["rep_blocks"]),
        dyn_blocks=model_cfg["dyn_blocks"],
        pred_blocks=model_cfg["pred_blocks"],
    )


def rollout(level, cfg, net, device, max_steps, seed=2024):
    pad_to = int(cfg["model"]["input_spatial"]) if cfg["env"]["pad_to_input_spatial"] else None
    env = create_train_env(
        level=level,
        int_path=cfg["env"]["int_path"],
        player_actions=None,
        n_frame=int(cfg["env"]["n_frame_stack"]),
        downsample=int(cfg["env"]["frame_skip"]),
        pad_to=pad_to,
        seed=seed,
    )
    mcts = MCTS(
        discount=float(cfg["muzero"]["discount"]),
        num_simulations=int(cfg["mcts"]["num_simulations"]),
        root_dirichlet_alpha=0.0,
        root_exploration_eps=0.0,
        pb_c_base=float(cfg["mcts"]["pb_c_base"]),
        pb_c_init=float(cfg["mcts"]["pb_c_init"]),
        device=device,
    )
    obs = env.reset()
    max_x, max_score, total_return, step, done, completed = 0, 0, 0.0, 0, False, False
    try:
        while not done and step < max_steps:
            with torch.no_grad():
                action, _, _ = mcts.run(obs, net, temperature=0.0, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_return += float(reward)
            x = 256 * int(info.get("player_x_posHi", 0)) + int(info.get("player_x_posLo", 0))
            max_x = max(max_x, x)
            max_score = max(max_score, int(info.get("score", 0)))
            step += 1
            if done:
                completed = bool(info.get("level_complete", False))
    finally:
        env.close()
    return {"completed": completed, "max_x": max_x, "score": max_score,
            "n_steps": step, "return": total_return}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(_REPO / "analysis/comparison/model_progress.json"))
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--max-steps", type=int, default=3000)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = str(Path(args.ckpt).resolve())  # pin the concrete step_*.pt
    state = load_checkpoint(ckpt_path, map_location=device)
    cfg = state["cfg_snapshot"]
    net = build_net(cfg["model"]).to(device)
    net.load_state_dict(state["online"], strict=True)
    net.eval()

    # Prefer the step stored in the checkpoint; fall back to the step_<N>.pt
    # filename (older checkpoints don't carry a "training_step" field).
    step = state.get("training_step")
    if step is None:
        m = re.search(r"step_(\d+)\.pt$", ckpt_path)
        step = int(m.group(1)) if m else None

    levels = args.levels or cfg["env"]["levels"]
    out = {"checkpoint": ckpt_path,
           "step": step,
           "env_step": state.get("env_step"),
           "levels": {}}
    for lv in levels:
        try:
            rec = rollout(lv, cfg, net, device, args.max_steps)
        except Exception as e:  # noqa: BLE001
            rec = {"error": str(e)[:200]}
        out["levels"][lv] = rec
        print(f"{lv}: {rec}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
