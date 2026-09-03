"""How sharp is the root visit distribution, and what makes it flat?

Motivation: across all 12 T6 specialists `selfplay/mcts_visit_max_frac` sits at
0.23-0.27 for the *whole* run — the most-visited root action never takes much
more than a quarter of the 32 simulations. That distribution is the policy
training target, so a permanently flat search trains a permanently flat policy
head, and greedy eval then argmaxes a near-tie.

Two mechanical suspects, both measurable on a fixed checkpoint with no
training and no env:
  * `mcts.leaf_batch` — leaves per inference round trip. Virtual visits pin
    each selected path so the next selection in the same round is steered
    elsewhere, which at 32 sims / 8 rounds forces visits across many root
    children by construction.
  * `mcts.num_simulations` — 32 over a 12-action set is few.

This sweeps both against the same bank of states and reports the visit
distribution's entropy / max share, its distance from the prior (is search
adding anything?), and whether the argmax agrees with a high-sim, sequential
reference search.

Usage:
    python scripts/diag_search_sharpness.py <ckpt> --level Level1-1 \
        --leaf-batches 1 2 4 --sims 32 64 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.checkpoint import load_checkpoint
from src.muzero.mcts import MCTS
from src.muzero.networks import MuZeroNet


def _build_net(m):
    return MuZeroNet(
        input_channels=m["input_channels"], input_spatial=m["input_spatial"],
        hidden_channels=m["hidden_channels"], hidden_spatial=m["hidden_spatial"],
        num_actions=m["num_actions"], value_support=tuple(m["value_support"]),
        reward_support=tuple(m["reward_support"]), rep_blocks=tuple(m["rep_blocks"]),
        dyn_blocks=m["dyn_blocks"], pred_blocks=m["pred_blocks"],
    )


def _entropy(p, axis=-1):
    p = np.asarray(p, dtype=np.float64)
    return -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)).sum(axis=axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--level", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--states", type=int, default=64, help="states from the human corpus")
    ap.add_argument("--human-dir", default="outputs/human_trajectories")
    ap.add_argument("--leaf-batches", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--sims", type=int, nargs="+", default=[32, 64, 200])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    state = load_checkpoint(args.ckpt, map_location=device)
    cfg = state["cfg_snapshot"]
    net = _build_net(cfg["model"]).to(device)
    net.load_state_dict(state["online"], strict=True)
    net.eval()
    level = args.level or cfg["env"]["levels"][0]

    from src.muzero.human_baseline import load_action_eval_sets
    sets = load_action_eval_sets(args.human_dir, [level], max_transitions_per_level=args.states)
    es = sets.get(level)
    if es is None or not len(es):
        print(f"no human states for {level}; pass a level humans played")
        return
    obs = np.asarray(es.obs, dtype=np.float32) / 255.0
    print(f"checkpoint {args.ckpt}\n  level={level} states={len(obs)} "
          f"train_step={state.get('training_step')}")

    with torch.no_grad():
        o = torch.from_numpy(obs).to(device)
        _, logits, _ = net.initial_inference(o)
        prior = F.softmax(logits.float(), dim=-1).cpu().numpy()
    print(f"  policy prior: entropy {_entropy(prior).mean():.3f} nats "
          f"(uniform {np.log(prior.shape[1]):.3f}), max share {prior.max(1).mean():.3f}, "
          f"{len(set(prior.argmax(1).tolist()))} distinct argmax over {len(obs)} states")

    def search_bank(sims, lb):
        np.random.seed(args.seed)
        mcts = MCTS(
            discount=float(cfg["muzero"]["discount"]), num_simulations=sims,
            root_dirichlet_alpha=0.0, root_exploration_eps=0.0,
            pb_c_base=float(cfg["mcts"]["pb_c_base"]),
            pb_c_init=float(cfg["mcts"]["pb_c_init"]),
            device=device, leaf_batch=lb,
        )
        return np.stack([mcts.run(obs[i], net, temperature=0.0, deterministic=True)[1]
                         for i in range(len(obs))])

    # Reference = the sharpest configuration available: fully sequential search
    # at the largest sim count. Every other cell is scored against its argmax,
    # so "!=ref" reads as "moves this checkpoint would not have made with a
    # search that was allowed to concentrate".
    ref_sims, ref_lb = max(args.sims), min(args.leaf_batches)
    ref_argmax = search_bank(ref_sims, ref_lb).argmax(1)
    print(f"\n  reference search: sims={ref_sims} leaf_batch={ref_lb}")

    rows = []
    print(f"\n{'leaf_batch':>10s} {'sims':>5s} {'visit_H':>8s} {'max_frac':>9s} "
          f"{'TV(prior)':>10s} {'!=prior':>8s} {'!=ref':>7s} {'n_visited':>9s}")
    for sims in args.sims:
        for lb in args.leaf_batches:
            pis = search_bank(sims, lb)
            tv = 0.5 * np.abs(pis - prior).sum(1)
            argmax = pis.argmax(1)
            row = {
                "leaf_batch": lb, "sims": sims,
                "visit_entropy": float(_entropy(pis).mean()),
                "visit_max_frac": float(pis.max(1).mean()),
                "tv_vs_prior": float(tv.mean()),
                "argmax_differs_from_prior": float(np.mean(argmax != prior.argmax(1))),
                "argmax_differs_from_ref": float(np.mean(argmax != ref_argmax)),
                "actions_visited_mean": float((pis > 0).sum(1).mean()),
                "n_distinct_argmax": int(len(set(argmax.tolist()))),
            }
            rows.append(row)
            print(f"{lb:>10d} {sims:>5d} {row['visit_entropy']:>8.3f} "
                  f"{row['visit_max_frac']:>9.3f} {row['tv_vs_prior']:>10.3f} "
                  f"{row['argmax_differs_from_prior']:>7.1%} "
                  f"{row['argmax_differs_from_ref']:>6.1%} "
                  f"{row['actions_visited_mean']:>9.2f}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"checkpoint": args.ckpt, "level": level,
             "training_step": int(state.get("training_step", -1)),
             "prior_entropy": float(_entropy(prior).mean()),
             "prior_max_frac": float(prior.max(1).mean()), "rows": rows}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
