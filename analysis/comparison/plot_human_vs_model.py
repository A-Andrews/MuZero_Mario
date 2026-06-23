"""Compare the current-best MuZero agent against human players, per level.

Inputs (both produced by replaying gameplay through the *same* stable-retro env):
  - analysis/comparison/human_attempts.csv   (scripts/replay_human_bk2.sh)
  - analysis/comparison/model_progress.json   (scripts/model_rollout_progress.sh)

Three panels, directly comparable (same emulator, same world-x, same in-game
score, same clear criterion):
  A. Completion rate per level  — fraction of attempts that reach the flagpole.
  B. Level progress reached (%)  — max world-x reached / level length (flagpole x).
  C. In-game score              — peak SMB score in an attempt (coins, enemies,
                                   time/flag bonus on a clear).

Bars are means; the small dots on each human bar are individual subjects (the
agent has one deterministic rollout, so no spread). Humans get many attempts
per level (a .bk2 may span several lives); the agent dies on its first life.
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

REPO = Path(__file__).resolve().parents[2]
HUMAN_CSV = REPO / "analysis/comparison/human_attempts.csv"
MODEL_JSON = REPO / "analysis/comparison/model_progress.json"
OUT_DIR = REPO / "analysis/comparison"
LEVELS = [f"Level{w}-{s}" for w in range(1, 5) for s in range(1, 4)]


def load_human():
    """Return (per-level aggregate df, per-(subject,level) df, level_len dict)."""
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
    return pd.DataFrame(rows), per_subj, level_len


def load_model(level_len: dict):
    data = json.loads(MODEL_JSON.read_text())
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
                     "m_clear": float(m.get("completed", 0) or 0),
                     "m_prog": prog,
                     "m_score": float(score) if score is not None else np.nan})
    return pd.DataFrame(rows), step


def _dots(ax, per_subj, level_col, xcenter, scale=1.0):
    """Scatter individual-subject values as jittered dots on a human bar."""
    rng = np.random.default_rng(0)
    for xi, lv in zip(xcenter, LEVELS):
        vals = per_subj.loc[per_subj["level"] == lv, level_col].dropna().to_numpy() * scale
        if len(vals) == 0:
            continue
        jit = (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), xi) + jit, vals, s=16, c="black",
                   alpha=0.65, zorder=5, edgecolors="none")


def main():
    human, per_subj, level_len = load_human()
    model, mstep = load_model(level_len)
    tab = human.merge(model, on="level")
    tab["level"] = pd.Categorical(tab["level"], categories=LEVELS, ordered=True)
    tab = tab.sort_values("level").reset_index(drop=True)
    tab.to_csv(OUT_DIR / "human_vs_model_per_level.csv", index=False)
    print(tab[["level", "h_clear", "m_clear", "h_prog", "m_prog",
               "h_score", "m_score", "h_attempts"]].to_string(index=False))

    x = np.arange(len(LEVELS))
    w = 0.4
    hx, mx = x - w / 2, x + w / 2
    hc, mc = "#1f77b4", "#d62728"
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # A. completion rate
    axA.bar(hx, tab["h_clear"], w, yerr=tab["h_clear_sd"], capsize=3, color=hc, label="Human (bar = mean)")
    axA.bar(mx, tab["m_clear"], w, color=mc, label="MuZero")
    _dots(axA, per_subj, "clear", hx)
    axA.set_ylabel("Completion rate\n(fraction of attempts)")
    axA.set_ylim(0, 1.05)
    axA.set_title("A. Level completion rate — reaches the flagpole")
    axA.legend(loc="upper right")
    for xi, mv in zip(mx, tab["m_clear"]):
        axA.text(xi, 0.02, f"{mv:.0%}", ha="center", va="bottom", fontsize=7, color=mc)

    # B. progress
    axB.bar(hx, tab["h_prog"] * 100, w, yerr=tab["h_prog_sd"] * 100, capsize=3, color=hc, label="Human")
    axB.bar(mx, tab["m_prog"] * 100, w, color=mc, label="MuZero")
    _dots(axB, per_subj, "prog", hx, scale=100.0)
    axB.set_ylabel("Level progress reached\n(% of distance to flag)")
    axB.set_ylim(0, 105)
    axB.set_title("B. How far through the level each gets")
    axB.legend(loc="upper right")
    for xi, mv in zip(mx, tab["m_prog"]):
        if not np.isnan(mv):
            axB.text(xi, mv * 100 + 1, f"{mv:.0%}", ha="center", va="bottom", fontsize=7, color=mc)

    # C. score
    axC.bar(hx, tab["h_score"], w, yerr=tab["h_score_sd"], capsize=3, color=hc, label="Human")
    axC.bar(mx, tab["m_score"], w, color=mc, label="MuZero")
    _dots(axC, per_subj, "score", hx)
    axC.set_ylabel("In-game score\n(peak per attempt)")
    axC.set_title("C. Total in-game score")
    axC.legend(loc="upper right")
    axC.set_xticks(x)
    axC.set_xticklabels(LEVELS, rotation=45, ha="right")
    for xi, mv in zip(mx, tab["m_score"]):
        if mv is not None and not (isinstance(mv, float) and np.isnan(mv)):
            axC.text(xi, mv + axC.get_ylim()[1] * 0.01, f"{int(mv)}", ha="center", va="bottom", fontsize=7, color=mc)

    for ax in (axA, axB, axC):
        ax.grid(axis="y", alpha=0.3)
        for xi, n in zip(x, tab["h_attempts"]):
            if n == 0:
                ax.text(xi, ax.get_ylim()[1] * 0.5, "no human data", rotation=90,
                        ha="center", va="center", fontsize=8, color="0.5", style="italic")

    step_lbl = f"step {mstep:,}" if mstep else "current-best checkpoint"
    fig.suptitle(f"Human players vs MuZero agent, per level  (agent {step_lbl})", fontsize=14)
    fig.text(0.5, 0.005,
             "Human = CNeuroMod players (bars = mean over subjects ± SD; dots = individual subjects; many attempts/level). "
             "MuZero = greedy-MCTS rollout of the current-best checkpoint. Both replayed through the same stable-retro env.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    png = OUT_DIR / "human_vs_model_per_level.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
