"""Compare human players against one or more MuZero runs, per level.

Inputs (all produced by replaying gameplay through the *same* stable-retro env):
  - analysis/comparison/human_attempts.csv        (scripts/replay_human_bk2.sh)
  - analysis/comparison/model_progress*.json       (scripts/model_rollout_*.sh)

Each `model_progress*.json` listed in MODEL_SPECS becomes its own bar group, so
several training runs can be compared side by side. Three panels:
  A. Completion rate per level  — fraction of attempts that reach the flagpole.
  B. Level progress reached (%)  — max world-x reached / level length (flagpole x).
  C. In-game score              — peak SMB score in an attempt.

Bars are means; the coloured dots on each human bar are individual subjects.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
HUMAN_CSV = REPO / "analysis/comparison/human_attempts.csv"
OUT_DIR = REPO / "analysis/comparison"
LEVELS = [f"Level{w}-{s}" for w in range(1, 5) for s in range(1, 4)]

# (label, progress-json, bar colour). Missing files are skipped.
MODEL_SPECS = [
    ("longdecay (small net)", OUT_DIR / "model_progress.json", "#e31a1c"),
    ("medium50 (medium net)", OUT_DIR / "model_progress_medium50.json", "#33a02c"),
    ("medium50-v2 (overhaul)", OUT_DIR / "model_progress_medium50v2.json", "#1f78b4"),
]
HUMAN_BAR = "#a6cee3"


def load_human():
    df = pd.read_csv(HUMAN_CSV)
    df = df[df.get("error").isna()] if "error" in df.columns else df
    df = df[df["level"].isin(LEVELS)].copy()
    df["completed"] = df["completed"].astype(str).str.lower().eq("true")
    if "score" not in df.columns:
        df["score"] = np.nan

    level_len = {}
    for lv in LEVELS:
        cl = df[(df["level"] == lv) & (df["completed"])]["max_x"]
        any_ = df[df["level"] == lv]["max_x"]
        level_len[lv] = float(cl.max()) if len(cl) else (float(any_.max()) if len(any_) else np.nan)
    df["progress"] = df.apply(
        lambda r: min(r["max_x"] / level_len[r["level"]], 1.0) if level_len.get(r["level"]) else np.nan,
        axis=1,
    )
    per_subj = (
        df.groupby(["subject", "level"], observed=True)
        .agg(clear=("completed", "mean"), prog=("progress", "mean"),
             score=("score", "mean"), n=("completed", "size"))
        .reset_index()
    )
    rows = []
    for lv in LEVELS:
        g = per_subj[per_subj["level"] == lv]
        rows.append({
            "level": lv,
            "h_clear": g["clear"].mean() if len(g) else np.nan,
            "h_clear_sd": g["clear"].std(ddof=0) if len(g) else np.nan,
            "h_prog": g["prog"].mean() if len(g) else np.nan,
            "h_prog_sd": g["prog"].std(ddof=0) if len(g) else np.nan,
            "h_score": g["score"].mean() if len(g) else np.nan,
            "h_score_sd": g["score"].std(ddof=0) if len(g) else np.nan,
            "h_attempts": int(df[df["level"] == lv].shape[0]),
        })
    return pd.DataFrame(rows).set_index("level"), per_subj, level_len


def load_model(json_path, level_len):
    data = json.loads(Path(json_path).read_text())
    step = data.get("step")
    if step is None:
        ckpt = data.get("checkpoint", "")
        try:
            ckpt = str(Path(ckpt).resolve())
        except Exception:  # noqa: BLE001
            pass
        m = re.search(r"step_(\d+)\.pt", ckpt)
        step = int(m.group(1)) if m else None
    rows = []
    for lv in LEVELS:
        m = data["levels"].get(lv, {})
        ln = level_len.get(lv)
        prog = (min(m.get("max_x", 0) / ln, 1.0) if ln and m.get("max_x") is not None else np.nan)
        score = m.get("score")
        rows.append({"level": lv,
                     "clear": float(m.get("completed", 0) or 0),
                     "prog": prog,
                     "score": float(score) if score is not None else np.nan})
    return pd.DataFrame(rows).set_index("level").reindex(LEVELS), step


def _dots(ax, per_subj, col, xpos, subjects, sub_colors, scale=1.0):
    """Individual-subject dots on each human bar, coloured by subject."""
    n = len(subjects)
    offsets = np.linspace(-0.16, 0.16, n) if n > 1 else np.array([0.0])
    bw = (xpos[1] - xpos[0]) if len(xpos) > 1 else 1.0
    for off, sub in zip(offsets, subjects):
        sd = per_subj[per_subj["subject"] == sub].set_index("level")
        xs, ys = [], []
        for xi, lv in zip(xpos, LEVELS):
            if lv in sd.index:
                v = sd.loc[lv, col]
                if not (isinstance(v, float) and np.isnan(v)):
                    xs.append(xi + off * 0.45 * bw / 0.16)
                    ys.append(v * scale)
        ax.scatter(xs, ys, s=18, color=sub_colors[sub], edgecolors="white",
                   linewidths=0.4, zorder=6)


def main():
    human, per_subj, level_len = load_human()
    models = []
    for name, path, color in MODEL_SPECS:
        if Path(path).exists():
            df, step = load_model(path, level_len)
            models.append(dict(name=name, df=df, step=step, color=color))
    print(f"models: {[(m['name'], m['step']) for m in models]}")

    # combined CSV for reference
    out = human.copy()
    for m in models:
        tag = m["name"].split()[0]
        for c in ("clear", "prog", "score"):
            out[f"{tag}_{c}"] = m["df"][c]
    out.to_csv(OUT_DIR / "human_vs_model_per_level.csv")

    x = np.arange(len(LEVELS))
    nbars = 1 + len(models)
    gw = 0.82
    bw = gw / nbars
    def pos(j):  # bar slot j: 0 = human, 1.. = models
        return x - gw / 2 + bw * (j + 0.5)
    hpos = pos(0)

    subjects = sorted(per_subj["subject"].dropna().unique())
    scmap = plt.get_cmap("Dark2")
    sub_colors = {s: scmap(i % 8) for i, s in enumerate(subjects)}

    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    def draw(ax, hcol, hsd, mcol, scale, label_fmt):
        ax.bar(hpos, human[hcol] * scale, bw,
               yerr=(human[hsd] * scale if hsd else None), capsize=3,
               color=HUMAN_BAR, label="Human", zorder=2)
        for i, m in enumerate(models):
            vals = m["df"][mcol] * scale
            ax.bar(pos(i + 1), vals, bw, color=m["color"], zorder=2,
                   label=f"{m['name']}  (step {m['step']:,})" if m["step"] else m["name"])
            for xi, v in zip(pos(i + 1), vals):
                if not np.isnan(v):
                    ax.text(xi, v + ax.get_ylim()[1] * 0.012, label_fmt(v),
                            ha="center", va="bottom", fontsize=6.5, color=m["color"], rotation=90)

    # A. completion rate
    draw(axA, "h_clear", "h_clear_sd", "clear", 1.0, lambda v: f"{v:.0%}")
    _dots(axA, per_subj, "clear", hpos, subjects, sub_colors)
    axA.set_ylabel("Completion rate\n(fraction of attempts)")
    axA.set_ylim(0, 1.08)
    axA.set_title("A. Level completion rate — reaches the flagpole")
    bar_leg = axA.legend(loc="upper right", fontsize=8)
    axA.add_artist(bar_leg)
    sub_handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                          markerfacecolor=sub_colors[s], markeredgecolor="white", label=s)
                   for s in subjects]
    axA.legend(handles=sub_handles, loc="upper left", ncol=len(subjects),
               fontsize=8, title="individual subjects", title_fontsize=8)

    # B. progress
    draw(axB, "h_prog", "h_prog_sd", "prog", 100.0, lambda v: f"{v:.0f}%")
    _dots(axB, per_subj, "prog", hpos, subjects, sub_colors, scale=100.0)
    axB.set_ylabel("Level progress reached\n(% of distance to flag)")
    axB.set_ylim(0, 112)
    axB.set_title("B. How far through the level each gets")
    axB.legend(loc="upper right", fontsize=8)

    # C. score
    draw(axC, "h_score", "h_score_sd", "score", 1.0, lambda v: f"{int(v)}")
    _dots(axC, per_subj, "score", hpos, subjects, sub_colors)
    axC.set_ylabel("In-game score\n(peak per attempt)")
    axC.set_title("C. Total in-game score")
    axC.legend(loc="upper right", fontsize=8)
    axC.set_xticks(x)
    axC.set_xticklabels(LEVELS, rotation=45, ha="right")

    for ax in (axA, axB, axC):
        ax.grid(axis="y", alpha=0.3)
        for xi, n in zip(x, human["h_attempts"]):
            if n == 0:
                ax.text(xi, ax.get_ylim()[1] * 0.5, "no human data", rotation=90,
                        ha="center", va="center", fontsize=8, color="0.5", style="italic")

    fig.suptitle("Human players vs MuZero runs, per level", fontsize=14)
    fig.text(0.5, 0.005,
             "Human = CNeuroMod players (bars = mean over subjects ± SD; dots = individual subjects). "
             "MuZero = greedy-MCTS rollout of each run's latest checkpoint. All replayed through the same stable-retro env.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    png = OUT_DIR / "human_vs_model_per_level.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
