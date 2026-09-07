#!/usr/bin/env python
"""Measure the performance deficit caused by lesioning each MuZero component.

The Mario counterpart of the Towers-of-Hanoi lesion study
(`~/region-specific-planning/Muzero-Hanoi`): re-initialise a targeted part of a
trained network at evaluation time and see how much worse it plays. Conventions
and the condition grid live in `src/muzero/lesion.py`; read its docstring first,
particularly on why "policy" here does not include the shared prediction trunk.

What the deficit is measured on: greedy MCTS run-throughs of the level the
model was trained on, scored by whether it finishes, how far it gets (`final_x`)
and how long it takes. `final_x` is the sensitive measure — completion is
0/1 and most lesions take a model straight to 0, whereas distance degrades
gradually and separates a mild deficit from a total one.

Two things this design gets right that are easy to get wrong:

- **Every rollout starts from a freshly reloaded network.** Lesions are applied
  to a clean copy, never on top of a previous lesion, so conditions cannot
  contaminate each other.
- **A lesion is a random variable.** One re-initialisation of the value head is
  one sample; a different seed gives a different damaged net. `--lesion-seeds`
  repeats each condition with different re-inits, and results should be read
  across seeds, not from any single one.

Usage:
    python scripts/lesion_eval.py --levels Level1-1 --rollouts 3 --lesion-seeds 3
    sbatch scripts/submit_lesion_eval.sh --levels Level3-3 Level6-1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.muzero.lesion import CONDITIONS, apply_lesion  # noqa: E402
from src.muzero.networks import MuZeroNet  # noqa: E402
from src.selfplay.replay_eval import run_replay_rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs" / "runs"


def pick_run(level: str, runs_dir: Path, ckpt_name: str):
    """Best available run for `level`, by recorded peak completion rate.

    Same rule as eval_human_benchmark.pick_run, so the lesion study evaluates
    the same checkpoint the benchmark figure reports.
    """
    tag = level.lower()
    cands = []
    for d in sorted(runs_dir.glob(f"spec*-{tag}")):
        ckpt = d / "checkpoints" / ckpt_name
        if not ckpt.exists():
            continue
        rate = -1.0
        side = d / "checkpoints" / "best.json"
        if side.exists():
            try:
                rate = float(json.loads(side.read_text())["completion_rate_100ep"])
            except (ValueError, KeyError):
                pass
        cands.append((rate, d, ckpt))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0], reverse=True)
    return cands[0]


def build_net(cfg_model, state_dict, device):
    net = MuZeroNet(**cfg_model)
    net.load_state_dict(state_dict, strict=True)
    return net.to(device).eval()


def evaluate_level(level, ckpt_path, conditions, rollouts, lesion_seeds,
                   max_steps, base_seed, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["cfg_snapshot"]
    env_cfg, mcts_cfg = cfg["env"], cfg["mcts"]
    sd = state["online"]

    results = []
    for cond in conditions:
        targets = CONDITIONS[cond]
        # An unlesioned net is deterministic given the rollout seed, so extra
        # lesion seeds would be identical repeats — run it once.
        seeds = [0] if not targets else list(range(lesion_seeds))
        for lseed in seeds:
            # Fresh net per lesion: never stack a lesion on a damaged net.
            net = build_net(cfg["model"], sd, device)
            record = apply_lesion(net, targets, seed=lseed)
            for i in range(rollouts):
                info = {}
                t0 = time.time()
                _frames, ret, steps, completed = run_replay_rollout(
                    level=level,
                    int_path=env_cfg["int_path"],
                    network=net,
                    device=device,
                    num_simulations=int(mcts_cfg["num_simulations"]),
                    discount=float(cfg["muzero"]["discount"]),
                    pb_c_base=float(mcts_cfg["pb_c_base"]),
                    pb_c_init=float(mcts_cfg["pb_c_init"]),
                    n_frame_stack=int(env_cfg["n_frame_stack"]),
                    frame_skip=int(env_cfg["frame_skip"]),
                    pad_to=int(cfg["model"]["input_spatial"])
                    if env_cfg["pad_to_input_spatial"] else None,
                    max_steps=max_steps,
                    seed=base_seed + i,
                    np_seed=base_seed + i,
                    leaf_batch=int(mcts_cfg.get("leaf_batch", 1)),
                    info_out=info,
                    noop_max=int(env_cfg.get("noop_max", 0) or 0),
                    skip_to_control=bool(env_cfg.get("skip_to_control", False)),
                    completion_bonus=float(env_cfg.get("completion_bonus", 100.0)),
                )
                results.append(dict(
                    level=level, condition=cond, lesion_seed=lseed,
                    rollout_seed=base_seed + i, completed=bool(completed),
                    steps=int(steps), ret=float(ret),
                    final_x=int(info.get("final_x", 0)),
                    timed_out=bool(info.get("timed_out", False)),
                    params_reset=record["params_reset"],
                    seconds=round(time.time() - t0, 1),
                ))
                r = results[-1]
                print(f"  [{level}] {cond:<22} lseed={lseed} roll={i} "
                      f"{'FINISHED' if r['completed'] else 'failed  '} "
                      f"x={r['final_x']:<5} steps={r['steps']:<5} ({r['seconds']}s)",
                      flush=True)
    return results


def summarise(results):
    """Per (level, condition): completion rate and median distance reached."""
    out = {}
    for r in results:
        out.setdefault((r["level"], r["condition"]), []).append(r)
    rows = []
    for (level, cond), rs in out.items():
        rows.append(dict(
            level=level, condition=cond, n=len(rs),
            completion_rate=round(sum(x["completed"] for x in rs) / len(rs), 3),
            final_x_median=statistics.median(x["final_x"] for x in rs),
            final_x_mean=round(statistics.fmean(x["final_x"] for x in rs), 1),
            steps_median=statistics.median(x["steps"] for x in rs),
            params_reset=rs[0]["params_reset"],
        ))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    help=f"default: all of {list(CONDITIONS)}")
    ap.add_argument("--rollouts", type=int, default=3, help="rollouts per lesion sample")
    ap.add_argument("--lesion-seeds", type=int, default=3,
                    help="independent re-initialisations per lesioned condition")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="outputs/lesion/lesion_eval.json")
    args = ap.parse_args()

    for c in args.conditions:
        if c not in CONDITIONS:
            ap.error(f"unknown condition {c!r}; known: {list(CONDITIONS)}")

    runs_dir = Path(args.runs_dir)
    all_results, meta = [], []
    for level in args.levels:
        picked = pick_run(level, runs_dir, args.checkpoint)
        if picked is None:
            print(f"[{level}] no run with checkpoints/{args.checkpoint} — skipped")
            continue
        rate, run_dir, ckpt = picked
        print(f"[{level}] {run_dir.name} ({ckpt.name}, recorded rate {rate})", flush=True)
        meta.append(dict(level=level, run=run_dir.name,
                         checkpoint=str(ckpt.relative_to(REPO)), recorded_rate=rate))
        all_results += evaluate_level(
            level, ckpt, args.conditions, args.rollouts, args.lesion_seeds,
            args.max_steps, args.seed, args.device)

    rows = summarise(all_results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        lesion_convention="random re-initialisation at evaluation time only",
        rollouts_per_lesion=args.rollouts, lesion_seeds=args.lesion_seeds,
        max_steps=args.max_steps, base_seed=args.seed, device=args.device,
        models=meta, summary=rows, rollouts=all_results), indent=1) + "\n")
    print(f"\nwrote {out}")

    print(f"\n{'level':<10} {'condition':<22} {'n':>3} {'complete':>9} {'x median':>9} {'params reset':>13}")
    for r in sorted(rows, key=lambda x: (x["level"], -x["final_x_median"])):
        print(f"{r['level']:<10} {r['condition']:<22} {r['n']:>3} "
              f"{r['completion_rate']:>9} {r['final_x_median']:>9} {r['params_reset']:>13,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
