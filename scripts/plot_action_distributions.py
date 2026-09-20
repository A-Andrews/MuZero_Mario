"""Compare recorded human action labels with actions actually executed by models.

Compute-node analysis; no inference. Human labels are quantized four-frame
windows, not original per-frame button presses. Include every available outcome.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env.mario_actions import complex_movement_to_button_presses

LABELS = ["No input", "Right", "Right + jump", "Right + run", "Right + jump + run",
          "Jump", "Left", "Left + jump", "Left + run", "Left + jump + run", "Down", "Up"]
LEVELS = ["Level1-1", "Level6-1", "Level3-2", "Level8-1"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def action_counts(actions):
    a = np.asarray(actions)
    if a.ndim != 1 or not len(a) or a.dtype.kind not in "iu" or (a < 0).any() or (a >= 12).any():
        raise ValueError("Expected a nonempty vector of integer actions 0..11")
    return np.bincount(a, minlength=12)


def distribution_summary(counts, seed=1511):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 12 or (counts < 0).any() or (counts.sum(1) <= 0).any():
        raise ValueError("Expected nonempty action counts for each sampling unit")
    units = counts / counts.sum(1, keepdims=True)
    draws = np.random.default_rng(seed).integers(len(units), size=(4000, len(units)))
    ci = np.quantile(units[draws].mean(1), [0.025, 0.975], axis=0)
    return {"n_units": len(units), "n_decisions": int(counts.sum()),
            "mean": units.mean(0).tolist(), "unit_distributions": units.tolist(),
            "bootstrap_ci95": ci.tolist(), "pooled_decision_distribution": (counts.sum(0) / counts.sum()).tolist()}


def collect(human_dir, experiment):
    manifest = json.loads((experiment / "manifest.json").read_text())
    report = {"levels": {}, "source_experiment": str(experiment.resolve()),
              "manifest_sha256": sha(experiment / "manifest.json"),
              "human_directory": str(human_dir.resolve()), "action_labels": LABELS,
              "interpretation": [
                  "Descriptive executed-action frequencies on each group's own visited states; not matched-state policy accuracy.",
                  "All human segments and all 30 development model episodes per arm count, including failures and stalls.",
                  "Human: pool decisions within participant, then weight participants equally. Model: weight episodes equally.",
                  "Human labels: majority button state in nominal four-frame windows, nearest supported action; not raw button frequencies.",
                  "Model decisions repeat an action for nominal four frames; terminal windows can be shorter in both groups.",
                  "Intervals resample participants (human) or episodes (model); five participants give limited population precision.",
                  "Different starts, respawn segments and visited states can explain frequency differences. No fMRI alignment claim.",
              ]}
    for level in LEVELS:
        world, stage = level.removeprefix("Level").split("-")
        human_files = sorted(human_dir.glob(f"sub-*_level-w{world}l{stage}_rep-*_seg*.npz"))
        if not human_files:
            raise ValueError(f"No human data for {level}")
        by_subject, files, reps = {}, [], set()
        for path in human_files:
            subject = path.name.split("_", 1)[0]
            with np.load(path, allow_pickle=False) as data:
                if str(data["level"].item()) != level:
                    raise ValueError(f"Human level mismatch: {path}")
                counts = action_counts(data["actions"])
                completed = bool(data["completed"].item())
            by_subject[subject] = by_subject.get(subject, np.zeros(12, np.int64)) + counts
            rep = re.sub(r"_seg\d+\.npz$", "", path.name)
            reps.add(rep)
            files.append({"path": str(path), "sha256": sha(path), "subject": subject,
                          "rep": rep, "completed": completed, "counts": counts.tolist()})
        row = {"human": {**distribution_summary([by_subject[s] for s in sorted(by_subject)]),
                         "unit_ids": sorted(by_subject), "n_segments": len(files), "n_reps": len(reps),
                         "files": files}}
        summary = json.loads((experiment / "development" / level / "summary.json").read_text())
        if not summary["complete"] or summary["split"] != "development":
            raise ValueError(f"Incomplete or wrong-split model data for {level}")
        for condition in ("greedy", "sampled"):
            counts, files = [], []
            folder = experiment / "development" / level / condition
            for trial in manifest["trials"]:
                result_path = folder / f"episode_{trial['episode_index']:04d}.json"
                result = json.loads(result_path.read_text())
                if result["status"] != "ok" or any(result[k] != v for k, v in trial.items()):
                    raise ValueError(f"Invalid scheduled model result: {result_path}")
                trace = result_path.with_name(result_path.stem + "_trace.npz")
                digest = sha(trace)
                if digest != result["artifacts"][trace.name]:
                    raise ValueError(f"Trace hash mismatch: {trace}")
                with np.load(trace, allow_pickle=False) as data:
                    if str(data["checkpoint_sha256"].item()) != manifest["levels"][level]["checkpoint_sha256"]:
                        raise ValueError("Wrong model checkpoint")
                    count = action_counts(data["actions"])
                if int(count.sum()) != result["steps"]:
                    raise ValueError("Trace length disagrees with episode result")
                counts.append(count)
                files.append({"path": str(trace), "sha256": digest, "counts": count.tolist(),
                              "completed": result["completed"], **trial})
            row[condition] = {**distribution_summary(counts), "unit_ids": list(range(len(counts))),
                              "completions": sum(f["completed"] for f in files), "files": files}
        report["levels"][level] = row
        print(f"{level}: {len(by_subject)} humans, {len(reps)} recordings, {len(human_files)} segments; 30 trials/model", flush=True)
    return report


def plot(report, out, buttons=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    matrix = np.stack([complex_movement_to_button_presses(a) for a in range(12)]).astype(float)
    labels = ["Run", "Up", "Down", "Left", "Right", "Jump"] if buttons else LABELS
    colours = {"human": "#333333", "greedy": "#0072B2", "sampled": "#D55E00"}
    names = {"human": "Human", "greedy": "Model: greedy", "sampled": "Model: sampled (T = 0.25)"}
    for ax, (level, row) in zip(axes.flat, report["levels"].items()):
        x = np.arange(len(labels))
        for offset, key in zip((-.26, 0, .26), ("human", "greedy", "sampled")):
            units = np.asarray(row[key]["unit_distributions"])
            if buttons:
                units = units @ matrix
            mean = units.mean(0)
            draws = np.random.default_rng(1511).integers(len(units), size=(4000, len(units)))
            ci = np.quantile(units[draws].mean(1), [.025, .975], axis=0)
            ax.bar(x + offset, 100 * mean, width=.24, color=colours[key], label=names[key], zorder=2)
            ax.errorbar(x + offset, 100 * mean, yerr=100 * np.maximum(0, np.stack([mean-ci[0], ci[1]-mean])),
                        fmt="none", ecolor=colours[key], capsize=2, lw=1, zorder=3)
            if key == "human":
                for i, unit in enumerate(units):
                    ax.scatter(x + offset + (i-(len(units)-1)/2)*.025, 100*unit, s=9,
                               facecolors="white", edgecolors=colours[key], linewidths=.5, zorder=4)
        control = " · control" if level == "Level6-1" else ""
        ax.set_title(f"{level.removeprefix('Level')}{control} | {row['human']['n_units']} participants\n"
                     f"Model completions: greedy {row['greedy']['completions']}/30; sampled {row['sampled']['completions']}/30")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel("Decisions containing button (%)" if buttons else "Share of decisions (%)")
        ax.grid(axis="y", alpha=.2, zorder=0)
        ax.set_ylim(0, 100)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
    fig.suptitle("Human and MuZero button use (derived from action labels)" if buttons else
                 "Human and MuZero action distributions", fontsize=19, y=.995)
    fig.text(.5, .014, "All outcomes included · equal participant / model-episode weights · dots: individual humans · bars: 95% bootstrap intervals\n"
             "Human labels quantized to the 12 model actions in four-frame windows. Groups visit different states. Model data: development trials.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .91))
    stem = "button_distributions" if buttons else "action_distributions"
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir", type=Path, default=Path("outputs/human_trajectories"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run corpus analysis on a compute node")
    args.out.mkdir(parents=True, exist_ok=True)
    report = collect(args.human_dir, args.experiment)
    report["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    report["script_sha256"] = sha(__file__)
    (args.out / "action_distributions.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "action_distributions.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "group", "action", "mean_fraction", "ci95_low", "ci95_high", "n_units", "n_decisions"])
        for level, row in report["levels"].items():
            for group, data in row.items():
                for a, label in enumerate(LABELS):
                    writer.writerow([level, group, label, data["mean"][a], data["bootstrap_ci95"][0][a],
                                     data["bootstrap_ci95"][1][a], data["n_units"], data["n_decisions"]])
    plot(report, args.out)
    plot(report, args.out, buttons=True)


if __name__ == "__main__":
    main()
