"""T8 step 1: turn the T6/T7 specialists into a teacher corpus for distillation.

Rolls each specialist out on its own level and writes `Trajectory` .npz files in
the *same format and naming convention* as `scripts/convert_human_bk2.py`, so
`src/muzero/human_data.py` can load them with **zero loader changes** and the
existing `imitation:` pipeline (pinned buffer + `MixedBuffer` + BC pretrain +
`mix_ratio_schedule`) distills them. A specialist is just another teacher.

Two things the plan in BACKLOG.md got wrong, both corrected here:

* **Filenames must start with `sub-`.** `select_human_files` globs `sub-*.npz`,
  so the proposed `spec-w1l1_...` naming would have been invisible to the
  loader. Files are written as
  ``sub-spec-w1l1_ses-000_task-mario_level-w1l1_rep-000_seg0.npz`` — the level
  tag still drives level filtering and the teacher still shows up as its own
  selectable "subject" (``sub-spec-w1l1``).
* **Rollouts need exploration noise.** The benchmark run (job 6083435) showed
  7 of 12 specialists cannot complete their own level *greedily* from `best.pt`
  — their headline completion rates lean on root-Dirichlet noise. A greedy
  dumper would hand those 7 levels a corpus containing zero successful
  demonstrations. So rollouts run at `--eps` (default 0.25, the training value)
  and, by default, only **completing** episodes are kept.

`policies` stores the **raw MCTS visit distribution**, not a one-hot: that is a
strictly richer target than the human corpus can offer, and it is what the
policy loss expects. `root_values` are real MCTS root Q values, so returns and
priorities are built exactly as `src/selfplay/worker.py` does rather than the
value-free approximation the human converter needs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.checkpoint import load_checkpoint
from src.env.env import create_train_env
from src.muzero.human_data import level_filename_tag
from src.muzero.mcts import MCTS
from src.muzero.returns import compute_n_step_returns
from scripts.eval_human_benchmark import ALL_LEVELS, build_net, git_sha, pick_run


def rollout_episode(level, cfg, net, device, *, eps, alpha, temperature, seed,
                    max_steps, noop_max, skip_to_control):
    """One self-play-style episode. Returns (trajectory dict, completed, final_x)."""
    env_cfg, mcts_cfg = cfg["env"], cfg["mcts"]
    np.random.seed(seed)
    env = create_train_env(
        level=level,
        int_path=env_cfg["int_path"],
        player_actions=None,
        n_frame=int(env_cfg["n_frame_stack"]),
        downsample=int(env_cfg["frame_skip"]),
        pad_to=int(cfg["model"]["input_spatial"]) if env_cfg["pad_to_input_spatial"] else None,
        seed=seed,
        noop_max=noop_max,
        skip_to_control=skip_to_control,
        completion_bonus=float(env_cfg.get("completion_bonus", 100.0)),
    )
    mcts = MCTS(
        discount=float(cfg["muzero"]["discount"]),
        num_simulations=int(mcts_cfg["num_simulations"]),
        root_dirichlet_alpha=alpha,
        root_exploration_eps=eps,
        pb_c_base=float(mcts_cfg["pb_c_base"]),
        pb_c_init=float(mcts_cfg["pb_c_init"]),
        device=device,
        leaf_batch=int(mcts_cfg.get("leaf_batch", 1)),
    )
    obs_l, act_l, rew_l, pol_l, q_l = [], [], [], [], []
    obs = env.reset()
    done, completed, final_x, steps = False, False, 0, 0
    try:
        while not done and steps < max_steps:
            with torch.no_grad():
                action, pi, root_q = mcts.run(
                    obs, net, temperature=temperature, deterministic=False)
            obs_l.append((obs * 255.0).clip(0, 255).astype(np.uint8))
            act_l.append(int(action))
            pol_l.append(np.asarray(pi, dtype=np.float32))
            q_l.append(float(root_q))
            obs, reward, done, info = env.step(action)
            rew_l.append(float(reward))
            steps += 1
            if "player_x_posHi" in info:
                final_x = 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
            if done:
                completed = bool(info.get("level_complete", False))
    finally:
        env.close()

    if not act_l:
        return None, False, final_x
    rewards = np.asarray(rew_l, dtype=np.float32)
    root_q = np.asarray(q_l, dtype=np.float32)
    # `done` distinguishes a real terminal from hitting the step cap: a capped
    # episode must NOT be treated as terminal or its returns lose the bootstrap.
    terminal = bool(done)
    returns = compute_n_step_returns(
        rewards, root_q, n_step=int(cfg["muzero"]["n_step"]),
        discount=float(cfg["muzero"]["discount"]), terminal=terminal)
    traj = dict(
        obs_stacks=np.stack(obs_l, axis=0),
        actions=np.asarray(act_l, dtype=np.int64),
        rewards=rewards,
        policies=np.stack(pol_l, axis=0).astype(np.float32),
        root_values=root_q,
        returns=returns,
        priorities=(np.abs(returns - root_q).astype(np.float32) + 1e-3),
        level=level,
        terminal=terminal,
        completed=completed,
    )
    return traj, completed, final_x


def dump_level(level, args, out_dir, device):
    picked = pick_run(level, Path(args.runs_dir), args.checkpoint)
    if picked is None:
        print(f"[{level}] no run with checkpoints/{args.checkpoint} — skipped", flush=True)
        return None
    rate, run_dir, ckpt_path, meta = picked
    state = load_checkpoint(ckpt_path, map_location=device)
    cfg = state["cfg_snapshot"]
    net = build_net(cfg["model"], device)
    net.load_state_dict(state["online"], strict=True)

    env_cfg = cfg["env"]
    noop_max = int(env_cfg.get("noop_max", 0)) if not args.deterministic_start else 0
    skip = bool(env_cfg.get("skip_to_control", False)) if not args.deterministic_start else False
    alpha = args.alpha if args.alpha is not None else float(cfg["mcts"]["dirichlet_alpha"])
    tag = level_filename_tag(level)
    subject = f"sub-spec-{tag.split('-', 1)[1]}"   # sub-spec-w1l1

    kept, attempts, steps_total, outcomes = 0, 0, 0, Counter()
    t0 = time.time()
    while kept < args.episodes_per_level and attempts < args.max_attempts_per_level:
        seed = args.seed + attempts
        traj, completed, final_x = rollout_episode(
            level, cfg, net, device, eps=args.eps, alpha=alpha,
            temperature=args.temperature, seed=seed, max_steps=args.max_steps,
            noop_max=noop_max, skip_to_control=skip)
        attempts += 1
        outcomes["completed" if completed else "failed"] += 1
        if traj is None or (args.completed_only and not completed):
            continue
        name = (f"{subject}_ses-000_task-mario_{tag}_rep-{kept:03d}_seg0.npz")
        np.savez_compressed(
            out_dir / name, **traj,
            bk2=f"{run_dir.name}@{meta.get('training_step','?')}",
            teacher_run=run_dir.name,
            teacher_checkpoint=str(ckpt_path),
            teacher_training_step=str(meta.get("training_step", "")),
            rollout_seed=str(seed), rollout_eps=str(args.eps),
            rollout_temperature=str(args.temperature))
        kept += 1
        steps_total += int(traj["actions"].shape[0])
        print(f"[{level}] kept {kept}/{args.episodes_per_level} "
              f"(attempt {attempts}, {traj['actions'].shape[0]} steps, x={final_x})", flush=True)

    print(f"[{level}] done: {kept} episodes / {steps_total} steps from {attempts} attempts "
          f"({outcomes['completed']} completed) in {time.time()-t0:.0f}s", flush=True)
    return dict(level=level, subject=subject, teacher_run=run_dir.name,
                teacher_checkpoint=str(ckpt_path),
                teacher_training_step=meta.get("training_step"),
                selfplay_rate_at_best=(rate if rate >= 0 else None),
                episodes_kept=kept, attempts=attempts,
                completions=outcomes["completed"], steps=steps_total,
                eps=args.eps, temperature=args.temperature,
                noop_max=noop_max, skip_to_control=skip)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", nargs="*", default=ALL_LEVELS)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--out", default="outputs/specialist_trajectories")
    ap.add_argument("--report", default="dump_report.json",
                    help="report filename inside --out. Give each job its own when "
                         "fanning out per level, or they race: the .npz files are "
                         "per-level and safe to share a directory, the report is not.")
    ap.add_argument("--episodes-per-level", type=int, default=40,
                    help="target KEPT episodes per level (completions, by default)")
    ap.add_argument("--max-attempts-per-level", type=int, default=200,
                    help="give up after this many rollouts even if short of the target")
    ap.add_argument("--eps", type=float, default=0.25,
                    help="root Dirichlet epsilon; 0 = greedy (which 7/12 specialists "
                         "cannot complete their own level under — see module docstring)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="root Dirichlet alpha (default: the run's own value)")
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--keep-failures", dest="completed_only", action="store_false",
                    help="also write episodes that never reached the flag")
    ap.add_argument("--deterministic-start", action="store_true",
                    help="disable the env's stochastic starts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.set_defaults(completed_only=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dump] device={device} levels={len(args.levels)} target={args.episodes_per_level}"
          f"/level eps={args.eps} temp={args.temperature} completed_only={args.completed_only}",
          flush=True)

    records = [r for r in (dump_level(lv, args, out_dir, device) for lv in args.levels)
               if r is not None]
    report = dict(generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                  git_sha=git_sha(), args=vars(args) | {"device": str(device)},
                  levels=records)
    (out_dir / args.report).write_text(json.dumps(report, indent=1, default=str))

    hdr = f"{'level':10s}{'teacher':22s}{'kept':>6s}{'attempts':>10s}{'steps':>9s}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in records:
        print(f"{r['level']:10s}{r['teacher_run']:22s}{r['episodes_kept']:6d}"
              f"{r['attempts']:10d}{r['steps']:9d}")
    short = [r["level"] for r in records if r["episodes_kept"] < args.episodes_per_level]
    print(f"\ntotal: {sum(r['steps'] for r in records)} steps across {len(records)} levels "
          f"-> {out_dir}")
    if short:
        print(f"SHORT of target on {len(short)} level(s): {', '.join(short)} — raise "
              f"--max-attempts-per-level, --eps, or accept fewer episodes there.")


if __name__ == "__main__":
    main()
