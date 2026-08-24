"""Offline: load a saved checkpoint and render one mp4 per level. No wandb.

With ``--compare-human`` each rollout is also scored against a human playing
the same level (same references the in-run `human_compare:` block logs), so a
checkpoint from a finished run can be compared without relaunching training.
"""
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
from src.muzero.human_baseline import compute_level_stats, load_action_eval_sets
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


def _print_human_comparison(level, completed, n_steps, ret, stats, eval_set, net, device, bonus):
    """One-line agent-vs-human verdict. `bonus` re-bases the human return onto
    this run's env.completion_bonus (the converter's was 100)."""
    if stats is None:
        print(f"    vs human: no human data for {level} (never played)")
        return
    if not stats.has_completions:
        print(
            f"    vs human: {stats.n_reps} reps, 0 completed — "
            f"no length/return reference"
        )
        return
    hr = stats.return_at_bonus(bonus)
    verdict = "INCOMPLETE"
    if completed:
        if n_steps <= stats.length_p10:
            verdict = "beats a good (p10) human"
        elif n_steps <= stats.length_median:
            verdict = "beats the median human"
        else:
            verdict = "completes, slower than median"
    print(
        f"    vs human: human completes {stats.completion_rate:.0%} of "
        f"{stats.n_reps} attempts ({stats.no_death_rate:.0%} without dying); "
        f"median {stats.length_median:.0f} steps / {hr:.1f} return "
        f"(p10 {stats.length_p10:.0f} steps)"
    )
    line = f"    -> {verdict}; return_ratio={ret / hr:.2f}" if hr else f"    -> {verdict}"
    if completed:
        line += f" length_ratio={n_steps / stats.length_median:.2f} (lower is better)"
    print(line)

    if eval_set is not None and len(eval_set):
        import torch.nn.functional as F
        obs = torch.from_numpy(eval_set.obs)
        acts = torch.from_numpy(eval_set.actions)
        correct, ce = 0, 0.0
        with torch.no_grad():
            for i in range(0, len(eval_set), 256):
                o = obs[i:i + 256].to(device).float().div_(255.0)
                a = acts[i:i + 256].to(device)
                _, logits, _ = net.initial_step(o)
                lp = F.log_softmax(logits.float(), dim=-1)
                ce += float(-lp.gather(1, a.unsqueeze(1)).sum().item())
                correct += int((lp.argmax(dim=-1) == a).sum().item())
        print(
            f"    -> action agreement with human presses: "
            f"{correct / len(eval_set):.1%} (CE {ce / len(eval_set):.3f}, n={len(eval_set)})"
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
    ap.add_argument(
        "--compare-human", action="store_true",
        help="Score each rollout against the human corpus for the same level",
    )
    ap.add_argument(
        "--human-dir", default="outputs/human_trajectories",
        help="Converted human corpus (for --compare-human)",
    )
    ap.add_argument(
        "--human-action-eval", type=int, default=0,
        help="Held-out human states per level to also score policy-head action "
             "agreement on (0 = performance comparison only). NOTE: with no "
             "record of which files this checkpoint's run trained on, this draws "
             "from the whole corpus — only meaningful for a non-imitation run.",
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

    human_stats, human_eval = {}, {}
    if args.compare_human:
        try:
            human_stats = compute_level_stats(args.human_dir, levels)
            human_eval = load_action_eval_sets(
                args.human_dir, levels,
                max_transitions_per_level=int(args.human_action_eval),
            )
        except Exception as e:
            print(f"[compare] human corpus unavailable: {e}")

    for level in levels:
        try:
            info = {}
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
                info_out=info,
            )
            path = out_dir / f"{level}.mp4"
            save_video(path, frames, fps=video_fps)
            print(
                f"{level}: return={ret:.2f}  steps={n}  "
                f"completed={completed}  -> {path} (+ {level}.bk2)"
            )
            if args.compare_human:
                _print_human_comparison(
                    level, completed, n, ret, human_stats.get(level),
                    human_eval.get(level), net, device,
                    float(cfg["env"].get("completion_bonus", 0.0)),
                )
        except Exception as e:
            print(f"{level}: FAILED — {e}")


if __name__ == "__main__":
    main()
