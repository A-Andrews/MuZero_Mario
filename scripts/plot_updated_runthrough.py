"""Render completed controller evaluations against the original human time bands.

Read-only evaluation analysis, run on Slurm. No new rollouts or seed selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_controller_diagnostic import canonical_sha, file_sha, wilson_interval
from scripts.human_level_scope import human_levels

LEVELS = ["Level1-1", "Level6-1", "Level3-2", "Level8-1"]
COLORS = {"greedy": "#0072B2", "sampled": "#D55E00"}


def arm_summary(records):
    if not records or any(r.get("status", "ok") != "ok" for r in records):
        raise ValueError("All scheduled attempts must be valid results")
    done = [r["steps"] for r in records if r["completed"]]
    return {"n": len(records), "completed": len(done), "rate": len(done)/len(records),
            "wilson95": list(wilson_interval(len(done), len(records))),
            "successful_steps": done, "successful_median": float(np.median(done)) if done else None,
            "timeouts": sum(bool(r.get("timed_out", False)) for r in records), "records": records}


def load_level(experiment, level, split):
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    folder = experiment / split / level
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text())
    if not summary["complete"] or summary["split"] != split:
        raise ValueError(f"Refusing incomplete or wrong-split data: {summary_path}")
    identity = summary["identity"]
    if identity["manifest_sha256"] != canonical_sha(manifest):
        raise ValueError("Manifest identity mismatch")
    if identity["checkpoint_sha256"] != manifest["levels"][level]["checkpoint_sha256"]:
        raise ValueError("Checkpoint identity mismatch")
    result = {"level": level, "split": split, "checkpoint": identity["checkpoint"],
              "checkpoint_sha256": identity["checkpoint_sha256"], "training_step": identity["training_step"],
              "manifest_path": str(manifest_path.resolve()), "manifest_sha256": file_sha(manifest_path),
              "summary_sha256": file_sha(summary_path), "arms": {}}
    for condition in ("greedy", "sampled"):
        records = []
        for trial in manifest["trials"]:
            path = folder / condition / f"episode_{trial['episode_index']:04d}.json"
            record = json.loads(path.read_text())
            if record["identity_sha256"] != canonical_sha(identity):
                raise ValueError("Episode identity mismatch")
            if any(record[k] != v for k, v in trial.items()) or record["condition"] != condition:
                raise ValueError(f"Scheduled trial mismatch: {path}")
            records.append({**record, "result_path": str(path.resolve()), "result_sha256": file_sha(path)})
        arm = arm_summary(records)
        existing = next(c for c in summary["conditions"] if c["condition"] == condition)
        if (arm["n"] != existing["planned_trials"] or arm["completed"] != existing["level_completions"]
                or existing["infrastructure_errors"] or not existing["complete"]):
            raise ValueError("Attempt totals disagree with completed summary")
        result["arms"][condition] = arm
    return result


def build_report(args):
    reference = json.loads(args.reference.read_text())
    legacy = {row["level"]: row for row in reference["levels"]}
    extension = getattr(args, "extension", None)
    extension_levels = (json.loads((extension / "manifest.json").read_text())["evaluation_levels"]
                        if extension is not None else [])
    levels = sorted(legacy, key=lambda s: tuple(map(int, s[5:].split("-")))) if args.all_levels else LEVELS
    available = set(human_levels(getattr(args, "human_dir", "outputs/human_trajectories")))
    excluded = [level for level in levels if level not in available]
    levels = [level for level in levels if level in available]
    rows = []
    for level in levels:
        if level in extension_levels:
            row = load_level(extension, level, "development")
        elif level in LEVELS:
            confirmed = args.confirmation is not None and level in ("Level1-1", "Level6-1")
            row = load_level(args.confirmation if confirmed else args.development,
                             level, "confirmation" if confirmed else "development")
        else:
            old = legacy[level]
            row = {"level": level, "split": "older benchmark", "checkpoint": old["checkpoint"],
                   "training_step": old["training_step"], "run": old["run"],
                   "arms": {"greedy": arm_summary(old["rollouts"])}}
        row["human"] = reference["human"].get(level)
        rows.append(row)
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "script_sha256": file_sha(__file__), "reference_path": str(args.reference.resolve()),
            "reference_sha256": file_sha(args.reference), "reference_generated": reference["generated"],
            "levels": rows, "excluded_without_human_data": excluded, "human_corpus_levels": sorted(available), "notes": [
                "Only levels represented in the converted human corpus are included; a missing successful-time band alone does not exclude a level.",
                "Human bands copied exactly from the original figure: p10/median/p90 of successful life segments, including respawn lives.",
                "Speed panel conditions on completion. Failures count in the model rate panel; zero successes has no speed estimate.",
                "Human and model starts and attempt definitions differ. No human/model rate equivalence or human-level claim.",
                "Each updated arm uses every scheduled trial. Root epsilon 0, leaf batch 4, 50 simulations; greedy T=0, sampled T=.25.",
                "Best specialist checkpoints are fixed, not newly trained. Confirmation/development/older data are labeled per level and never pooled.",
                "Completion intervals are Wilson 95%; the speed panel has one dot per successful scheduled episode, not per correlated frame.",
            ]}


def make_figure(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    rows = report["levels"]
    height = max(6.5, 2.9 + .53*len(rows))
    fig, (speed, rates) = plt.subplots(1, 2, figsize=(14, height), sharey=True,
                                      gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=.17, right=.93, top=1-.95/height, bottom=1.35/height, wspace=.13)
    for i, row in enumerate(rows):
        y = len(rows)-1-i
        h = row["human"]
        if h and h["length_median"] > 0:
            speed.plot([h["length_p10"], h["length_p90"]], [y, y], color="#777777", lw=8, alpha=.28, solid_capstyle="round")
            speed.plot(h["length_median"], y, "|", color="#333333", ms=15, mew=2)
        for condition, offset in (("greedy", .16), ("sampled", -.16)):
            if condition not in row["arms"]:
                continue
            arm = row["arms"][condition]
            yy, color = y+offset, COLORS[condition]
            done = arm["successful_steps"]
            if done:
                jitter = np.linspace(-.045, .045, len(done)) if len(done)>1 else np.zeros(1)
                speed.scatter(done, yy+jitter, s=15, color=color, alpha=.45, linewidths=0, zorder=3)
                speed.plot(arm["successful_median"], yy, "|", color=color, ms=12, mew=2.3, zorder=4)
            else:
                speed.text(.015, yy, "no completions", transform=speed.get_yaxis_transform(), color=color, va="center", fontsize=8)
            lo, hi = arm["wilson95"]
            rates.plot([lo*100, hi*100], [yy, yy], color=color, lw=1.5)
            rates.plot(arm["rate"]*100, yy, "o", color=color, ms=5)
            rates.text(1.04, yy, f"{arm['completed']}/{arm['n']}", transform=rates.get_yaxis_transform(),
                       color=color, va="center", fontsize=9)
        if i < len(rows)-1:
            for ax in (speed, rates):
                ax.axhline(y-.5, color="#dddddd", lw=.5)
    positions = np.arange(len(rows))
    labels = []
    for row in reversed(rows):
        control = " · control" if row["level"] == "Level6-1" else ""
        labels.append(f"{row['level'][5:]}{control}\n{row['split']} · step {row['training_step']:,}")
    speed.set_yticks(positions, labels)
    speed.tick_params(axis="y", length=0, pad=12, labelsize=9)
    rates.tick_params(axis="y", length=0, labelleft=False)
    speed.set_ylim(-.55, len(rows)-.45)
    speed.set_xlim(left=0)
    rates.set_xlim(-3, 103)
    rates.set_xticks([0, 25, 50, 75, 100])
    for ax in (speed, rates):
        ax.grid(axis="x", color="#eeeeee", lw=.8)
        ax.set_axisbelow(True)
    speed.set_xlabel("Decisions to the flag, successful attempts only (lower is faster)")
    rates.set_xlabel("Model completion rate (%) · Wilson 95% interval")
    speed.set_title("Completion time", loc="left", fontsize=12, pad=16)
    rates.set_title("Reliability across all attempts", loc="left", fontsize=12, pad=16)
    fig.suptitle("Human and MuZero level run-throughs · updated controller evaluation", fontsize=16, x=.17, ha="left", y=.985)
    fig.legend(handles=[Line2D([], [], color="#777777", lw=7, alpha=.4, label="Human successful p10–p90; tick = median"),
                        Line2D([], [], color=COLORS["greedy"], marker="o", ls="", label="Greedy MCTS"),
                        Line2D([], [], color=COLORS["sampled"], marker="o", ls="", label="Sampled MCTS (T = 0.25)")],
               loc="upper left", bbox_to_anchor=(.165, 1-.34/height), frameon=False, ncol=3, fontsize=9)
    footer = ("Model: unchanged best.pt specialists; no root noise. Dots show successful attempts; colored ticks show their medians. Counts include every failure.\n"
              "Human: same CNeuroMod successful-life bands as the original figure, including respawn lives; nominal four frames per decision.\n"
              "Rows identify development, confirmation or older benchmark data. These sets are separate. Human/model starts differ; speed alone is not human-level performance.")
    fig.text(.17, .025, footer, ha="left", va="bottom", fontsize=8, color="#555555", linespacing=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(output.with_suffix("."+ext), dpi=180, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--extension", type=Path, help="Completed remaining-level development experiment")
    parser.add_argument("--reference", type=Path, default=Path("images/human_vs_agent_runthrough.json"))
    parser.add_argument("--human-dir", type=Path, default=Path("outputs/human_trajectories"))
    parser.add_argument("--out", type=Path, required=True, help="Output stem, without extension")
    parser.add_argument("--all-levels", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run figure generation on a compute node")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    make_figure(report, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Wrote {args.out}.pdf / .png / .json", flush=True)


if __name__ == "__main__":
    main()
