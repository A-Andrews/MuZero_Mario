"""Sweep eval-time search knobs against a checkpoint and report completion rates.

Motivation (2026-07): level1-1-diag-v1 and level1-2-diag-v1 both reached
0.59-0.84 `selfplay/completion_rate_100ep` while every greedy `replay/<level>_completed`
stayed at 0. By the end of those runs the self-play temperature schedule was
already at 0.1 (visits^10 — within a few percent of argmax), so the *only*
substantive difference between the two measurements was root Dirichlet noise
(eps=0.25, alpha=0.25 in self-play; hardcoded to 0 in replay_eval).

This script separates the knobs — noise, temperature, UCB constant, simulation
count — and runs N episodes per cell so the numbers have error bars. The env is
fully deterministic, so a noise-free/argmax cell is a single trajectory no
matter how many episodes you ask for: it is run once and reused, and the
per-cell `deterministic` flag in the output records that.

Usage:
  python scripts/eval_sweep.py <ckpt> --levels Level1-2 --episodes 20 \
      --eps 0 0.1 0.25 --temperature 0 0.1 0.25 --out outputs/eval_sweep/1-2

Results land in `<out>/results.json` (per-episode) and `<out>/summary.csv`
(per-cell), plus one mp4 of the first episode of each cell unless --no-video.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
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


def _wilson_interval(k, n, z=1.96):
    """95% Wilson score interval — sane at k=0 and k=n, unlike the normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Path to a checkpoint (best.pt / latest.pt / step_N.pt)")
    ap.add_argument("--out", default="outputs/eval_sweep", help="Output dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--levels", nargs="*", default=None, help="Default: cfg.env.levels")
    ap.add_argument("--episodes", type=int, default=20, help="Episodes per stochastic cell")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2024, help="Base seed; episode i uses seed+i")
    # Sweep axes. Each is a list; the grid is their product.
    ap.add_argument("--eps", type=float, nargs="*", default=[0.0, 0.1, 0.25],
                    help="Root Dirichlet exploration fractions")
    ap.add_argument("--alpha", type=float, nargs="*", default=[0.25],
                    help="Root Dirichlet alphas (only used where eps > 0)")
    ap.add_argument("--temperature", type=float, nargs="*", default=[0.0, 0.1, 0.25],
                    help="Visit-count selection temperatures (0 = argmax)")
    ap.add_argument("--pb-c-init", type=float, nargs="*", default=None,
                    help="UCB exploration constants (default: checkpoint's cfg value)")
    ap.add_argument("--num-simulations", type=int, nargs="*", default=None,
                    help="MCTS simulation counts (default: checkpoint's cfg value)")
    ap.add_argument("--no-video", action="store_true", help="Skip mp4 writing (faster)")
    args = ap.parse_args()

    device = torch.device(args.device)
    state = load_checkpoint(args.ckpt, map_location=device)
    cfg = state["cfg_snapshot"]

    net = _build_net(cfg["model"]).to(device)
    net.load_state_dict(state["online"], strict=True)
    net.eval()

    levels = args.levels or cfg["env"]["levels"]
    pb_c_inits = args.pb_c_init or [float(cfg["mcts"]["pb_c_init"])]
    sim_counts = args.num_simulations or [int(cfg["mcts"]["num_simulations"])]
    pad_to = int(cfg["model"]["input_spatial"]) if cfg["env"]["pad_to_input_spatial"] else None
    frame_skip = int(cfg["env"]["frame_skip"])
    video_fps = max(1, 60 // frame_skip)
    leaf_batch = int(cfg["mcts"].get("leaf_batch", 1))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(levels, sim_counts, pb_c_inits, args.eps, args.alpha, args.temperature))
    # alpha only matters when eps > 0 — collapse the duplicate cells it creates.
    seen = set()
    cells = []
    for level, sims, pb_c, eps, alpha, temp in grid:
        key = (level, sims, pb_c, eps, alpha if eps > 0 else None, temp)
        if key in seen:
            continue
        seen.add(key)
        cells.append((level, sims, pb_c, eps, alpha, temp))

    print(f"[sweep] checkpoint: {args.ckpt}")
    print(f"[sweep] {len(cells)} cells x up to {args.episodes} episodes")
    print(f"[sweep] levels={levels} sims={sim_counts} pb_c_init={pb_c_inits}")

    episodes_out = []
    summary_rows = []
    t_start = time.time()

    for ci, (level, sims, pb_c, eps, alpha, temp) in enumerate(cells):
        # No root noise and argmax selection => the search is a pure function of
        # the (deterministic) emulator state. Repeats would be bit-identical.
        deterministic = (eps <= 0.0 or alpha <= 0.0) and temp <= 0.0
        n_ep = 1 if deterministic else args.episodes
        tag = f"{level}_sims{sims}_pbc{pb_c:g}_eps{eps:g}_a{alpha:g}_t{temp:g}"
        completions = 0
        returns, steps, xs = [], [], []

        for ep in range(n_ep):
            info_out = {}
            want_video = (not args.no_video) and ep == 0
            try:
                frames, ret, n, completed = run_replay_rollout(
                    level=level,
                    int_path=cfg["env"]["int_path"],
                    network=net,
                    device=device,
                    num_simulations=sims,
                    discount=float(cfg["muzero"]["discount"]),
                    pb_c_base=float(cfg["mcts"]["pb_c_base"]),
                    pb_c_init=pb_c,
                    n_frame_stack=int(cfg["env"]["n_frame_stack"]),
                    frame_skip=frame_skip,
                    pad_to=pad_to,
                    max_steps=args.max_steps,
                    seed=args.seed + ep,
                    leaf_batch=leaf_batch,
                    temperature=temp,
                    root_dirichlet_alpha=alpha,
                    root_exploration_eps=eps,
                    np_seed=args.seed + ep,
                    info_out=info_out,
                )
            except Exception as e:
                print(f"[sweep] {tag} ep{ep}: FAILED — {e}")
                continue

            completions += int(completed)
            returns.append(ret)
            steps.append(n)
            xs.append(info_out.get("final_x", 0))
            episodes_out.append({
                "level": level, "num_simulations": sims, "pb_c_init": pb_c,
                "eps": eps, "alpha": alpha, "temperature": temp,
                "episode": ep, "seed": args.seed + ep,
                "completed": bool(completed), "return": ret, "steps": n,
                "final_x": info_out.get("final_x", 0),
                "timed_out": info_out.get("timed_out", False),
            })
            if want_video:
                save_video(out_dir / f"{tag}.mp4", frames, fps=video_fps)

        n_done = len(returns)
        rate = completions / n_done if n_done else 0.0
        lo, hi = _wilson_interval(completions, n_done)
        row = {
            "level": level, "num_simulations": sims, "pb_c_init": pb_c,
            "eps": eps, "alpha": alpha, "temperature": temp,
            "deterministic": deterministic, "episodes": n_done,
            "completions": completions, "completion_rate": round(rate, 4),
            "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            "mean_return": round(sum(returns) / n_done, 2) if n_done else 0.0,
            "mean_steps": round(sum(steps) / n_done, 1) if n_done else 0.0,
            "mean_final_x": round(sum(xs) / n_done, 1) if n_done else 0.0,
            "max_final_x": max(xs) if xs else 0,
        }
        summary_rows.append(row)
        print(
            f"[sweep] {ci+1}/{len(cells)} {tag}: "
            f"completion {completions}/{n_done} = {rate:.2f} [{lo:.2f},{hi:.2f}]  "
            f"return {row['mean_return']}  x {row['mean_final_x']}/{row['max_final_x']}  "
            f"({time.time()-t_start:.0f}s elapsed)"
        )
        # Written incrementally so a wall-time kill still leaves usable results.
        (out_dir / "results.json").write_text(json.dumps({
            "checkpoint": str(args.ckpt), "args": vars(args),
            "episodes": episodes_out, "summary": summary_rows,
        }, indent=1))
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print(f"[sweep] done in {time.time()-t_start:.0f}s -> {out_dir}")
    best = max(summary_rows, key=lambda r: r["completion_rate"], default=None)
    if best:
        print(f"[sweep] best cell: {best}")


if __name__ == "__main__":
    main()
