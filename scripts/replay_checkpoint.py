"""Offline: load a saved checkpoint and render one mp4 per level. No wandb."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.checkpoint import load_checkpoint
from src.logs.video import save_video
from src.muzero.networks import MuZeroNet
from src.selfplay.replay_eval import run_replay_rollout


def _build_net(model_cfg):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Path to step_N.pt")
    ap.add_argument("--out", default="replays", help="Output dir for mp4s")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--levels", nargs="*", default=None, help="Subset of levels (default: cfg.env.levels)")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument(
        "--num-simulations", type=int, default=None,
        help="Override MCTS simulations per move (default: checkpoint's cfg value)",
    )
    ap.add_argument(
        "--seed", type=int, default=2024,
        help="Env seed — vary to probe robustness of the greedy rollout",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    state = load_checkpoint(args.ckpt, map_location=device)
    cfg = state["cfg_snapshot"]

    net = _build_net(cfg["model"]).to(device)
    net.load_state_dict(state["online"], strict=True)
    net.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = args.levels or cfg["env"]["levels"]
    pad_to = int(cfg["model"]["input_spatial"]) if cfg["env"]["pad_to_input_spatial"] else None
    frame_skip = int(cfg["env"]["frame_skip"])
    video_fps = max(1, 60 // frame_skip)

    for level in levels:
        try:
            frames, ret, n, completed = run_replay_rollout(
                level=level,
                int_path=cfg["env"]["int_path"],
                network=net,
                device=device,
                num_simulations=(
                    args.num_simulations
                    if args.num_simulations is not None
                    else int(cfg["mcts"]["num_simulations"])
                ),
                discount=float(cfg["muzero"]["discount"]),
                pb_c_base=float(cfg["mcts"]["pb_c_base"]),
                pb_c_init=float(cfg["mcts"]["pb_c_init"]),
                n_frame_stack=int(cfg["env"]["n_frame_stack"]),
                frame_skip=frame_skip,
                pad_to=pad_to,
                max_steps=args.max_steps,
                seed=args.seed,
                bk2_path=out_dir / f"{level}.bk2",
            )
            path = out_dir / f"{level}.mp4"
            save_video(path, frames, fps=video_fps)
            print(
                f"{level}: return={ret:.2f}  steps={n}  "
                f"completed={completed}  -> {path} (+ {level}.bk2)"
            )
        except Exception as e:
            print(f"{level}: FAILED — {e}")


if __name__ == "__main__":
    main()
