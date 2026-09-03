"""Diagnose "the policy only ever plays one action".

Three measurements, because the complaint conflates things that have very
different causes:

1. **Static probe** — the policy head evaluated on a fixed bank of *off-policy*
   states (human corpus for the level, else states sampled along the rollout).
   A head that is collapsed *as a function* has near-zero entropy and the same
   argmax on every state; a head that is merely confident has low entropy but
   an argmax that moves with the state. `argmax_switch_rate` (how often the
   argmax changes between consecutive human states) separates the two.

2. **Rollout probe** — one episode stepped with the same MCTS the evaluator
   uses, recording per step: the pre-noise prior, the raw visit distribution,
   and the action actually taken. Answers whether search is doing anything on
   top of the prior (`visit_vs_prior_tv`) and whether the *taken* action is
   varied (run-length encoding of the action sequence).

3. **Stuck-window probe** — the same numbers restricted to the tail of a
   rollout that timed out, since a policy can be fine for 1500 steps and then
   sit on one action against a wall forever. Reported separately so the
   healthy prefix does not average the pathology away.

Usage:
    python scripts/diag_policy_collapse.py <ckpt> --level Level1-1 [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.checkpoint import load_checkpoint
from src.env.env import create_train_env
from src.muzero.mcts import MCTS
from src.muzero.networks import MuZeroNet

ACTION_NAMES = [
    "NOOP", "right", "right+A", "right+B", "right+A+B", "A",
    "left", "left+A", "left+B", "left+A+B", "down", "up",
]


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


def _entropy(p, axis=-1):
    p = np.asarray(p, dtype=np.float64)
    return -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)).sum(axis=axis)


def _hist(actions, n_actions, top=4):
    c = Counter(int(a) for a in actions)
    total = max(1, len(actions))
    rows = sorted(c.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(
        f"{ACTION_NAMES[a] if a < len(ACTION_NAMES) else a}={n / total:.1%}" for a, n in rows
    )


def _rle(actions, max_runs=12):
    runs, prev, n = [], None, 0
    for a in actions:
        if a == prev:
            n += 1
        else:
            if prev is not None:
                runs.append((int(prev), n))
            prev, n = a, 1
    if prev is not None:
        runs.append((int(prev), n))
    head = runs[:max_runs]
    txt = " ".join(f"{ACTION_NAMES[a] if a < len(ACTION_NAMES) else a}x{n}" for a, n in head)
    if len(runs) > max_runs:
        txt += f" ... (+{len(runs) - max_runs} more runs)"
    return runs, txt


def _priors_for_obs(net, obs_u8, device, batch=128, scale_255=True):
    """(N, A) prior probabilities and (N,) value scalars for a bank of obs."""
    priors, values = [], []
    with torch.no_grad():
        for i in range(0, len(obs_u8), batch):
            o = torch.from_numpy(np.asarray(obs_u8[i:i + batch])).to(device).float()
            if scale_255:
                o = o.div_(255.0)
            h, logits, value = net.initial_inference(o)
            priors.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
            values.append(value.float().cpu().numpy().reshape(-1))
    return np.concatenate(priors, 0), np.concatenate(values, 0)


def _summarise_prior_bank(priors, values, n_actions, label):
    ent = _entropy(priors)
    argmax = priors.argmax(axis=1)
    switch = float(np.mean(argmax[1:] != argmax[:-1])) if len(argmax) > 1 else 0.0
    top_share = Counter(argmax.tolist()).most_common(1)[0]
    out = {
        "label": label,
        "n_states": int(len(priors)),
        "prior_entropy_mean": float(ent.mean()),
        "prior_entropy_p10": float(np.percentile(ent, 10)),
        "prior_entropy_p90": float(np.percentile(ent, 90)),
        "uniform_entropy": float(np.log(n_actions)),
        "prior_max_prob_mean": float(priors.max(axis=1).mean()),
        "n_distinct_argmax": int(len(set(argmax.tolist()))),
        "modal_argmax": ACTION_NAMES[top_share[0]] if top_share[0] < len(ACTION_NAMES) else top_share[0],
        "modal_argmax_share": top_share[1] / len(argmax),
        "argmax_switch_rate": switch,
        "per_action_prob_std": priors.std(axis=0).max().item(),
        "value_mean": float(values.mean()),
        "value_std": float(values.std()),
        "argmax_hist": _hist(argmax, n_actions),
    }
    return out


def _print_bank(b):
    print(f"  [{b['label']}] n={b['n_states']} states")
    print(
        f"    prior entropy  {b['prior_entropy_mean']:.3f} nats "
        f"(p10 {b['prior_entropy_p10']:.3f} / p90 {b['prior_entropy_p90']:.3f}) "
        f"vs uniform {b['uniform_entropy']:.3f}"
    )
    print(f"    max prob mean  {b['prior_max_prob_mean']:.3f}")
    print(
        f"    argmax: {b['n_distinct_argmax']} distinct, modal {b['modal_argmax']} "
        f"{b['modal_argmax_share']:.1%}, switches between consecutive states "
        f"{b['argmax_switch_rate']:.1%}"
    )
    print(f"    argmax hist: {b['argmax_hist']}")
    print(
        f"    largest per-action prob std across states {b['per_action_prob_std']:.4f}"
        f"   |  value {b['value_mean']:.2f} +/- {b['value_std']:.2f}"
    )


def _window_stats(prior, visits, actions, n_actions, label):
    prior, visits = np.asarray(prior), np.asarray(visits)
    tv = 0.5 * np.abs(prior - visits).sum(axis=1)
    return {
        "label": label,
        "n_steps": int(len(actions)),
        "prior_entropy_mean": float(_entropy(prior).mean()),
        "visit_entropy_mean": float(_entropy(visits).mean()),
        "uniform_entropy": float(np.log(n_actions)),
        "prior_max_frac_mean": float(prior.max(axis=1).mean()),
        "visit_max_frac_mean": float(visits.max(axis=1).mean()),
        "visit_vs_prior_tv_mean": float(tv.mean()),
        "search_flips_argmax": float(np.mean(prior.argmax(1) != visits.argmax(1))),
        "n_distinct_actions_taken": int(len(set(int(a) for a in actions))),
        "modal_action_share": (
            Counter(int(a) for a in actions).most_common(1)[0][1] / max(1, len(actions))
        ),
        "action_hist": _hist(actions, n_actions),
        "prior_argmax_hist": _hist(prior.argmax(1), n_actions),
        "visit_argmax_hist": _hist(visits.argmax(1), n_actions),
    }


def _print_window(w):
    print(f"  [{w['label']}] {w['n_steps']} steps")
    print(
        f"    entropy   prior {w['prior_entropy_mean']:.3f} | visits "
        f"{w['visit_entropy_mean']:.3f}  (uniform {w['uniform_entropy']:.3f})"
    )
    print(
        f"    max-frac  prior {w['prior_max_frac_mean']:.3f} | visits "
        f"{w['visit_max_frac_mean']:.3f}"
    )
    print(
        f"    search vs prior: TV {w['visit_vs_prior_tv_mean']:.3f}, "
        f"flips argmax {w['search_flips_argmax']:.1%}"
    )
    print(
        f"    actions taken: {w['n_distinct_actions_taken']} distinct, "
        f"modal {w['modal_action_share']:.1%} -> {w['action_hist']}"
    )
    print(f"    prior argmax: {w['prior_argmax_hist']}")
    print(f"    visit argmax: {w['visit_argmax_hist']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--level", default=None, help="default: first of the checkpoint's env.levels")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--num-simulations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--np-seed", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0, help="<=0 = argmax on visits")
    ap.add_argument("--eps", type=float, default=0.0, help="root Dirichlet eps (0 = greedy eval)")
    ap.add_argument("--dirichlet-alpha", type=float, default=0.0)
    ap.add_argument(
        "--stochastic-start", action="store_true",
        help="use the run's own noop_max/skip_to_control instead of a deterministic start",
    )
    ap.add_argument("--human-dir", default="outputs/human_trajectories")
    ap.add_argument("--human-states", type=int, default=512)
    ap.add_argument("--tail-window", type=int, default=200)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    state = load_checkpoint(args.ckpt, map_location=device)
    cfg = state["cfg_snapshot"]
    net = _build_net(cfg["model"]).to(device)
    net.load_state_dict(state["online"], strict=True)
    net.eval()

    level = args.level or cfg["env"]["levels"][0]
    n_actions = int(cfg["model"]["num_actions"])
    sims = args.num_simulations or int(cfg["mcts"]["num_simulations"])
    print(f"checkpoint : {args.ckpt}")
    print(f"  train_step={state.get('training_step')} env_step={state.get('env_step')}")
    print(f"  level={level}  sims={sims}  eps={args.eps}  temperature={args.temperature}")

    report = {
        "checkpoint": str(args.ckpt),
        "level": level,
        "training_step": int(state.get("training_step", -1)),
        "env_step": int(state.get("env_step", -1)),
        "num_simulations": sims,
        "eps": args.eps,
        "temperature": args.temperature,
        "banks": [],
        "windows": [],
    }

    # ---- 1. static probe on off-policy human states -----------------------
    print("\n[1] policy head on off-policy human states")
    try:
        from src.muzero.human_baseline import load_action_eval_sets
        sets = load_action_eval_sets(
            args.human_dir, [level], max_transitions_per_level=int(args.human_states)
        )
        es = sets.get(level)
        if es is None or not len(es):
            print("    no human states for this level (never played) — skipping")
        else:
            pri, val = _priors_for_obs(net, es.obs, device)
            bank = _summarise_prior_bank(pri, val, n_actions, "human states")
            human_acts = np.asarray(es.actions)
            bank["agreement_with_human"] = float(np.mean(pri.argmax(1) == human_acts))
            bank["human_action_hist"] = _hist(human_acts, n_actions)
            _print_bank(bank)
            print(
                f"    agreement with the human's button press "
                f"{bank['agreement_with_human']:.1%}  (human plays: {bank['human_action_hist']})"
            )
            report["banks"].append(bank)
    except Exception as e:
        print(f"    human probe unavailable: {e}")

    # ---- 2. rollout probe --------------------------------------------------
    print("\n[2] greedy rollout with per-step prior / visits / action")
    if args.np_seed is not None:
        np.random.seed(int(args.np_seed))
    pad_to = int(cfg["model"]["input_spatial"]) if cfg["env"]["pad_to_input_spatial"] else None
    env = create_train_env(
        level=level,
        int_path=cfg["env"]["int_path"],
        player_actions=None,
        n_frame=int(cfg["env"]["n_frame_stack"]),
        downsample=int(cfg["env"]["frame_skip"]),
        pad_to=pad_to,
        seed=args.seed,
        completion_bonus=float(cfg["env"].get("completion_bonus", 100.0)),
        noop_max=int(cfg["env"].get("noop_max", 0)) if args.stochastic_start else 0,
        skip_to_control=bool(cfg["env"].get("skip_to_control", False)) if args.stochastic_start else False,
    )
    mcts = MCTS(
        discount=float(cfg["muzero"]["discount"]),
        num_simulations=sims,
        root_dirichlet_alpha=args.dirichlet_alpha,
        root_exploration_eps=args.eps,
        pb_c_base=float(cfg["mcts"]["pb_c_base"]),
        pb_c_init=float(cfg["mcts"]["pb_c_init"]),
        device=device,
        leaf_batch=int(cfg["mcts"].get("leaf_batch", 1)),
    )

    priors, visit_dists, actions, xs, root_qs = [], [], [], [], []
    obs_bank = []
    obs = env.reset()
    done, completed, step, final_x, total_return = False, False, 0, 0, 0.0
    try:
        while not done and step < args.max_steps:
            with torch.no_grad():
                o = torch.from_numpy(obs).to(device, dtype=torch.float32).unsqueeze(0)
                _, logits, _ = net.initial_inference(o)
                prior = F.softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
            action, pi, root_q = mcts.run(
                obs, net, temperature=args.temperature, deterministic=False
            )
            priors.append(prior)
            visit_dists.append(pi)
            actions.append(action)
            root_qs.append(root_q)
            if step % 10 == 0:
                obs_bank.append(np.asarray(obs))
            obs, reward, done, info = env.step(action)
            total_return += float(reward)
            step += 1
            if "player_x_posHi" in info:
                final_x = 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
            xs.append(final_x)
            if done:
                completed = bool(info.get("level_complete", False))
    finally:
        env.close()

    timed_out = bool(not done and step >= args.max_steps)
    print(
        f"    steps={step} completed={completed} timed_out={timed_out} "
        f"final_x={final_x} return={total_return:.2f} "
        f"root_Q mean={np.mean(root_qs):.2f}"
    )
    report.update(
        {
            "rollout": {
                "steps": step,
                "completed": completed,
                "timed_out": timed_out,
                "final_x": final_x,
                "total_return": total_return,
                "root_q_mean": float(np.mean(root_qs)),
            }
        }
    )
    full = _window_stats(priors, visit_dists, actions, n_actions, "whole rollout")
    _print_window(full)
    report["windows"].append(full)

    runs, rle_txt = _rle(actions)
    print(f"    action run-length (first runs): {rle_txt}")
    print(f"    longest single-action run: {max(n for _, n in runs)} steps")
    report["longest_action_run"] = int(max(n for _, n in runs))
    report["n_action_runs"] = len(runs)

    # ---- 3. stuck-window probe --------------------------------------------
    if step > args.tail_window:
        print(f"\n[3] last {args.tail_window} steps only")
        w = args.tail_window
        tail = _window_stats(
            priors[-w:], visit_dists[-w:], actions[-w:], n_actions, f"last {w} steps"
        )
        _print_window(tail)
        x_tail = xs[-w:]
        print(f"    x range over the window: {min(x_tail)}..{max(x_tail)} (stuck if flat)")
        tail["x_min"], tail["x_max"] = int(min(x_tail)), int(max(x_tail))
        report["windows"].append(tail)

    # on-policy state bank, for comparison with the human one
    if obs_bank:
        print("\n[4] policy head on states visited by this rollout (on-policy)")
        pri, val = _priors_for_obs(net, np.stack(obs_bank), device, scale_255=False)
        bank = _summarise_prior_bank(pri, val, n_actions, "rollout states")
        _print_bank(bank)
        report["banks"].append(bank)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
