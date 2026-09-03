"""Can the trained model get through a level the way a person does?

For each level this loads that level's **best** checkpoint, plays a handful of
full run-throughs, and plots them against the distribution of *successful human
run-throughs* on the same level. It answers a deliberately narrow question —
"can the model complete this level, and in a human-like number of steps?" — and
is NOT the like-for-like statistical comparison: the agent side is a few
rollouts from a hand-picked checkpoint, the human side is every recorded
attempt. Every figure therefore carries the run name, checkpoint file and
training step it came from, and the JSON sidecar repeats them.

Why `best.pt` and not `latest.pt`: `step_*.pt` files rotate (keep=10) and
several runs end in a late-training collapse, so the final checkpoint
underrepresents what the model learned. `best.pt` is the checkpoint at the
run's peak rolling completion rate and is the only one guaranteed to survive.

Why stochastic starts are ON here: NES Mario is otherwise fully deterministic,
so N greedy rollouts from one checkpoint would be N copies of one trajectory.
`env.noop_max`/`env.skip_to_control` are read from the run's own config so the
rollouts genuinely differ; pass --deterministic for the single reproducible
greedy trajectory instead.

Output (default `images/`):
  human_vs_agent_runthrough.pdf   vector figure
  human_vs_agent_runthrough.json  per-rollout results + provenance
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.checkpoint import load_checkpoint
from src.muzero.human_baseline import compute_level_stats
from src.muzero.networks import MuZeroNet
from src.selfplay.replay_eval import run_replay_rollout

def discover_levels(runs_dir="outputs/runs") -> list:
    """Every level with at least one trained specialist run, world-stage ordered.

    Derived rather than hard-coded so a new fleet (T9 added worlds 5-8) appears
    in the figure without editing this script. Levels whose run never wrote a
    best.pt are still skipped downstream by `pick_run`.
    """
    seen = set()
    for d in Path(runs_dir).glob("spec*-level*"):
        tag = d.name.split("-level", 1)[1]
        if re.fullmatch(r"\d+-\d+", tag):
            seen.add("Level" + tag)
    return sorted(seen, key=lambda L: tuple(int(x) for x in L.removeprefix("Level").split("-")))


ALL_LEVELS = discover_levels()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _repo_rel(p: Path) -> str:
    """Path as written relative to the repo root when it lives inside it.

    `--runs-dir` may be given relative or absolute, and `outputs/` is a symlink
    to /projects on Isambard, so neither `relative_to` alone nor `resolve()`
    alone is enough — try both and fall back to the path as given.
    """
    for cand in (p, p.resolve()):
        try:
            return str(cand.relative_to(_REPO_ROOT))
        except ValueError:
            continue
    return str(p)


def pick_run(level: str, runs_dir: Path, ckpt_name: str):
    """Best available run for `level`, ranked by its recorded peak completion rate.

    Several levels have both a T6 self-play run (`spec-<level>`) and a T7
    imitation rescue (`spec-imit-<level>`); whichever actually got further wins,
    rather than hard-coding which family supersedes.
    """
    tag = level.lower()
    cands = []
    for d in sorted(runs_dir.glob(f"spec*-{tag}")):
        ckpt = d / "checkpoints" / ckpt_name
        if not ckpt.exists():
            continue
        meta = {}
        side = d / "checkpoints" / "best.json"
        if side.exists():
            try:
                meta = json.loads(side.read_text())
            except ValueError:
                pass
        cands.append((float(meta.get("completion_rate_100ep", -1.0)), d, ckpt, meta))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0], reverse=True)
    return cands[0]


def build_net(model_cfg, device):
    net = MuZeroNet(
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
    ).to(device)
    if device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)
    return net.eval()


def evaluate_level(level, runs_dir, ckpt_name, n_rollouts, max_steps, device,
                   deterministic, base_seed):
    picked = pick_run(level, runs_dir, ckpt_name)
    if picked is None:
        print(f"[{level}] no run with checkpoints/{ckpt_name} — skipped", flush=True)
        return None
    rate, run_dir, ckpt_path, meta = picked

    state = load_checkpoint(ckpt_path, map_location=device)
    cfg = state["cfg_snapshot"]
    net = build_net(cfg["model"], device)
    net.load_state_dict(state["online"], strict=True)

    env_cfg = cfg["env"]
    noop_max = 0 if deterministic else int(env_cfg.get("noop_max", 0))
    skip = False if deterministic else bool(env_cfg.get("skip_to_control", False))
    n = 1 if deterministic else n_rollouts  # a noise-free rollout is one trajectory

    rollouts = []
    for i in range(n):
        info, t0 = {}, time.time()
        _frames, ret, steps, completed = run_replay_rollout(
            level=level,
            int_path=env_cfg["int_path"],
            network=net,
            device=device,
            num_simulations=int(cfg["mcts"]["num_simulations"]),
            discount=float(cfg["muzero"]["discount"]),
            pb_c_base=float(cfg["mcts"]["pb_c_base"]),
            pb_c_init=float(cfg["mcts"]["pb_c_init"]),
            n_frame_stack=int(env_cfg["n_frame_stack"]),
            frame_skip=int(env_cfg["frame_skip"]),
            pad_to=int(cfg["model"]["input_spatial"]) if env_cfg["pad_to_input_spatial"] else None,
            max_steps=max_steps,
            seed=base_seed + i,
            np_seed=base_seed + i,
            leaf_batch=int(cfg["mcts"].get("leaf_batch", 1)),
            info_out=info,
            noop_max=noop_max,
            skip_to_control=skip,
            completion_bonus=float(env_cfg.get("completion_bonus", 100.0)),
        )
        rollouts.append(dict(seed=base_seed + i, completed=bool(completed), steps=int(steps),
                             ret=float(ret), final_x=int(info.get("final_x", 0)),
                             timed_out=bool(info.get("timed_out", False))))
        # "died" and "hit the step cap" are different failures — a capped run is
        # looping or stalled, not slow, and the JSON records timed_out either way.
        outcome = "FLAG" if completed else ("capped" if info.get("timed_out") else "died")
        print(f"[{level}] rollout {i+1}/{n}: "
              f"{outcome} steps={steps} x={info.get('final_x')} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del _frames

    done = [r["steps"] for r in rollouts if r["completed"]]
    return dict(
        level=level, run=run_dir.name, checkpoint=_repo_rel(ckpt_path),
        training_step=meta.get("training_step"), env_step=meta.get("env_step"),
        selfplay_rate_at_best=(rate if rate >= 0 else None),
        deterministic=deterministic, noop_max=noop_max, skip_to_control=skip,
        n_rollouts=n, n_completed=len(done),
        steps_median=(float(np.median(done)) if done else None),
        steps_best=(int(min(done)) if done else None),
        rollouts=rollouts,
    )


def make_figure(results, human, out_pdf: Path, meta: dict):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": "#c7ced9", "axes.linewidth": 0.8,
        "xtick.color": "#4a515e", "ytick.color": "#10131a",
        "xtick.direction": "out", "ytick.direction": "out",
    })
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    AGENT, HUMAN, RULE, INK2, INK3 = "#2a78d6", "#eb6834", "#dfe3ea", "#4a515e", "#7b8393"

    rows = [r for r in results if r is not None]

    def rank(r):
        """Never-finished last; otherwise by agent median / human median, so the
        levels where the model most clearly matches a person read first."""
        if r["n_completed"] == 0:
            return (2, 0.0)
        h = human.get(r["level"])
        if h is None or not h.has_completions:
            return (1, 0.0)          # no human reference: between the two groups
        return (0, r["steps_median"] / h.length_median)

    rows.sort(key=rank)

    fig_h = 1.9 + 0.42 * len(rows)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    fig.subplots_adjust(left=0.085, right=0.775, top=1 - 0.85 / fig_h, bottom=1.05 / fig_h)

    ymax = 0
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ymax = max(ymax, y)
        h = human.get(r["level"])
        if h is not None and h.has_completions:
            ax.plot([h.length_p10, h.length_p90], [y, y], color=HUMAN, lw=6,
                    alpha=0.30, solid_capstyle="round", zorder=1)
            ax.plot([h.length_median], [y], marker="|", ms=13, mew=2.0,
                    color=HUMAN, zorder=3)
        done = [x["steps"] for x in r["rollouts"] if x["completed"]]
        if done:
            ax.plot(done, [y] * len(done), "o", ms=6.5, color=AGENT,
                    mec="white", mew=1.0, zorder=4)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["level"].replace("Level", "") for r in reversed(rows)],
                       fontfamily="DejaVu Sans Mono")
    ax.set_ylim(-0.7, ymax + 0.7)
    ax.set_xlabel("agent steps to the flag  (lower is faster)", color=INK2)
    ax.set_xlim(left=0)
    ax.grid(axis="x", color=RULE, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # right-hand gutter: the verdict, and which checkpoint produced it
    xr = ax.get_xlim()[1]
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        h = human.get(r["level"])
        if r["n_completed"] == 0:
            verdict, col = "never finished", HUMAN
        elif h is not None and h.has_completions and r["steps_median"] <= h.length_median:
            verdict, col = "faster than median", AGENT
        elif h is not None and h.has_completions:
            verdict, col = "slower than median", INK2
        else:
            verdict, col = "no human data", INK3
        ax.text(xr * 1.035, y, f"{r['n_completed']}/{r['n_rollouts']}", va="center",
                ha="left", fontsize=8.5, color=col, family="DejaVu Sans Mono")
        ax.text(xr * 1.115, y, verdict, va="center", ha="left", fontsize=8, color=col)
        ax.text(xr * 1.36, y, f"{r['run']} @ {r['training_step'] or '?'}", va="center",
                ha="left", fontsize=6.6, color=INK3, family="DejaVu Sans Mono")
    ax.annotate("finished", xy=(xr * 1.035, ymax + 0.85), fontsize=7, color=INK3,
                annotation_clip=False, family="DejaVu Sans Mono")
    ax.annotate("checkpoint (run @ train step)", xy=(xr * 1.36, ymax + 0.85), fontsize=7,
                color=INK3, annotation_clip=False, family="DejaVu Sans Mono")

    fig.suptitle("Can the model get through the level like a person does?",
                 x=0.085, ha="left", fontsize=13, fontweight="bold", color="#10131a")
    ax.legend(handles=[
        Line2D([], [], color=AGENT, marker="o", ls="", ms=6.5, mec="white",
               label="agent run-through"),
        Line2D([], [], color=HUMAN, lw=6, alpha=0.30, label="human p10–p90"),
        Line2D([], [], color=HUMAN, marker="|", ls="", ms=13, mew=2.0,
               label="human median"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, frameon=False,
        fontsize=8, handletextpad=0.6, columnspacing=1.4)

    fig.text(0.085, 0.008, meta["footer"], fontsize=6.5, color=INK3,
             va="bottom", ha="left", wrap=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", nargs="*", default=ALL_LEVELS)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--checkpoint", default="best.pt",
                    help="checkpoint file inside each run's checkpoints/ (default best.pt)")
    ap.add_argument("--rollouts", type=int, default=5, help="run-throughs per level")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--deterministic", action="store_true",
                    help="one noise-free greedy rollout per level instead of --rollouts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="images")
    ap.add_argument("--human-dir", default="outputs/human_trajectories")
    ap.add_argument("--name", default="human_vs_agent_runthrough")
    args = ap.parse_args()

    device = torch.device(args.device)
    runs_dir = Path(args.runs_dir)
    out_dir = _REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    print(f"[eval] device={device} levels={len(args.levels)} rollouts={args.rollouts} "
          f"checkpoint={args.checkpoint}", flush=True)

    results = [evaluate_level(lv, runs_dir, args.checkpoint, args.rollouts,
                              args.max_steps, device, args.deterministic, args.seed)
               for lv in args.levels]
    ok = [r for r in results if r is not None]
    human = compute_level_stats(args.human_dir, [r["level"] for r in ok])

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sha = git_sha()
    mode = "greedy, deterministic start" if args.deterministic else \
           f"{args.rollouts} rollouts, stochastic starts (seeds {args.seed}–{args.seed+args.rollouts-1})"
    footer = (
        f"Agent: {args.checkpoint} of each level's highest-scoring specialist run; {mode}; "
        f"greedy action selection, no root noise. Human: successful run-throughs from the converted "
        f"CNeuroMod corpus, same frame-skip. Levels humans never played show no orange band. "
        f"Not a like-for-like rate comparison — a few rollouts from a hand-picked checkpoint "
        f"against every recorded human attempt. Generated {stamp} · git {sha}"
    )
    make_figure(ok, human, out_dir / f"{args.name}.pdf", {"footer": footer})

    payload = dict(
        generated=stamp, git_sha=sha, checkpoint_kind=args.checkpoint,
        rollouts_per_level=(1 if args.deterministic else args.rollouts),
        deterministic=args.deterministic, seed=args.seed, max_steps=args.max_steps,
        levels=ok,
        human={lv: dict(completion_rate_first_life=s.no_death_rate,
                        completion_rate_any_life=s.completion_rate,
                        length_p10=s.length_p10, length_median=s.length_median,
                        length_p90=s.length_p90, n_reps=s.n_reps)
               for lv, s in human.items()},
    )
    (out_dir / f"{args.name}.json").write_text(json.dumps(payload, indent=1))

    print(f"\n[eval] wrote {out_dir / (args.name + '.pdf')}")
    print(f"[eval] wrote {out_dir / (args.name + '.json')}")
    hdr = f"{'level':10s}{'run':22s}{'ckpt step':>11s}{'finished':>10s}{'median':>9s}{'human med':>11s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in ok:
        h = human.get(r["level"])
        step = str(r["training_step"]) if r["training_step"] is not None else "?"
        fin = f"{r['n_completed']}/{r['n_rollouts']}"
        med = f"{r['steps_median']:.0f}" if r["steps_median"] is not None else "—"
        hmed = f"{h.length_median:.0f}" if (h is not None and h.has_completions) else "—"
        print(f"{r['level']:10s}{r['run']:22s}{step:>11s}{fin:>10s}{med:>9s}{hmed:>11s}")

if __name__ == "__main__":
    main()
