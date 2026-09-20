"""Human actions, policy priors and search outputs on all human-covered levels.

Read saved traces on compute nodes; do not run inference or select successes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_controller_diagnostic import canonical_sha
from scripts.human_level_scope import human_levels
from scripts.plot_action_distributions import LABELS, action_counts, distribution_summary, sha

CONDITIONS = ("greedy", "sampled")
GROUPS = ("policy_probabilities", "search_visits", "policy_argmax", "executed_actions")
SHORT_LABELS = ["None", "Right", "Right + jump", "Right + run", "Right + jump + run", "Jump",
                "Left", "Left + jump", "Left + run", "Left + jump + run", "Down", "Up"]


def probability_counts(values):
    p = np.asarray(values, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 12 or not len(p) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-5, rtol=0)):
        raise ValueError("Expected normalized finite probabilities for each of 12 actions")
    # Remove float32 rounding error only; no temperature or sharpening.
    return (p / p.sum(1, keepdims=True)).sum(0)


def summarize(counts, lengths, unit_ids):
    result = distribution_summary(counts)
    result["n_decisions"] = int(sum(lengths))
    result["unit_ids"] = unit_ids
    return result


def collect_humans(directory, level):
    world, stage = level[5:].split("-")
    paths = sorted(directory.glob(f"sub-*_level-w{world}l{stage}_rep-*_seg*.npz"))
    counts, files = {}, []
    for path in paths:
        subject = path.name.split("_", 1)[0]
        with np.load(path, allow_pickle=False) as data:
            if str(data["level"].item()) != level:
                raise ValueError(f"Human level mismatch: {path}")
            c = action_counts(data["actions"])
            complete = bool(data["completed"].item())
        counts[subject] = counts.get(subject, np.zeros(12, dtype=np.int64)) + c
        files.append({"path": str(path.resolve()), "sha256": sha(path), "subject": subject,
                      "n_decisions": int(c.sum()), "completed": complete})
    if not counts:
        raise ValueError(f"No human action data: {level}")
    subjects = sorted(counts)
    return {**summarize([counts[s] for s in subjects], [counts[s].sum() for s in subjects], subjects),
            "n_segments": len(files), "files": files}


def collect_models(experiment, level):
    manifest = json.loads((experiment / "manifest.json").read_text())
    summary = json.loads((experiment / "development" / level / "summary.json").read_text())
    identity = summary["identity"]
    if (manifest["split"] != "development" or not summary["complete"] or
            summary["split"] != "development" or manifest["episodes"] != 30 or
            len(manifest["trials"]) != 30 or identity["manifest_sha256"] != canonical_sha(manifest) or
            identity["checkpoint_sha256"] != manifest["levels"][level]["checkpoint_sha256"]):
        raise ValueError(f"Incomplete or mismatched model evaluation: {level}")
    model = {"experiment": str(experiment.resolve()), "manifest_sha256": sha(experiment / "manifest.json"),
             "checkpoint_sha256": identity["checkpoint_sha256"], "training_step": identity["training_step"],
             "split": "development", "conditions": {}}
    for condition in CONDITIONS:
        settings = next(c for c in manifest["conditions"] if c["name"] == condition)
        if (settings["eps"] != 0 or settings["leaf_batch"] != 4 or
                settings["temperature"] != (0. if condition == "greedy" else .25) or manifest["num_simulations"] != 50):
            raise ValueError("Unexpected search settings")
        totals = next(c for c in summary["conditions"] if c["condition"] == condition)
        if not totals["complete"] or totals["infrastructure_errors"] or totals["finished_trials"] != 30:
            raise ValueError("Incomplete controller condition")
        counts = {key: [] for key in GROUPS}
        lengths, ids, files = [], [], []
        for trial in manifest["trials"]:
            path = experiment / "development" / level / condition / f"episode_{trial['episode_index']:04d}.json"
            record = json.loads(path.read_text())
            if (record["status"] != "ok" or record["condition"] != condition or
                    record["identity_sha256"] != canonical_sha(identity) or any(record[k] != v for k,v in trial.items())):
                raise ValueError(f"Invalid scheduled episode: {path}")
            trace = path.with_name(path.stem + "_trace.npz")
            digest = sha(trace)
            if digest != record["artifacts"][trace.name]:
                raise ValueError(f"Trace hash mismatch: {trace}")
            with np.load(trace, allow_pickle=False) as data:
                if str(data["checkpoint_sha256"].item()) != identity["checkpoint_sha256"]:
                    raise ValueError("Trace checkpoint mismatch")
                prior, visits, actions = data["priors"], data["policies"], data["actions"]
                if prior.shape != visits.shape or len(actions) != len(prior) or len(actions) != record["steps"]:
                    raise ValueError("Policy/search/action state alignment mismatch")
                counts["policy_probabilities"].append(probability_counts(prior))
                counts["search_visits"].append(probability_counts(visits))
                counts["policy_argmax"].append(action_counts(prior.argmax(1)))
                counts["executed_actions"].append(action_counts(actions))
                if condition == "greedy" and not np.array_equal(visits.argmax(1), actions):
                    raise ValueError("Greedy actions differ from search argmax")
            lengths.append(record["steps"])
            ids.append(trial["episode_index"])
            files.append({"path": str(trace.resolve()), "sha256": digest, "completed": record["completed"],
                          "n_decisions": record["steps"], **trial})
        if sum(f["completed"] for f in files) != totals["level_completions"]:
            raise ValueError("Completion accounting mismatch")
        model["conditions"][condition] = {"n_episodes": len(ids), "completions": totals["level_completions"],
            "files": files, **{key: summarize(counts[key], lengths, ids) for key in GROUPS}}
    return model


def collect(args):
    original = json.loads((args.development / "manifest.json").read_text())
    extension = json.loads((args.extension / "manifest.json").read_text())
    if original["trials"] != extension["trials"]:
        raise ValueError("All-level development seed lists differ")
    model_levels = set(original["levels"]) | set(extension["evaluation_levels"])
    levels = human_levels(args.human_dir)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "script_sha256": sha(__file__), "action_labels": LABELS, "levels": {},
              "excluded_model_levels_without_human_data": sorted(model_levels-set(levels)),
              "notes": [
                  "Human: observed quantized actions; pool within participant, then weight participants equally.",
                  "Model: average within each episode, then weight all 30 scheduled episodes equally, including deaths and stalls.",
                  "Policy probabilities and raw search visits are measured on EXACTLY the same model observations within each controller condition.",
                  "Greedy and sampled conditions visit different states; they are displayed separately. No root noise; 50 simulations; leaf batch 4.",
                  "Search visits are raw training-target probabilities, before temperature/action selection; they are not executed-action frequencies.",
                  "Companion action-choice plots show hypothetical policy-head argmax labels on the logged states versus actual search-selected actions.",
                  "Hypothetical head argmax is not a policy-only rollout. Sampled actions use T=.25; greedy uses T=0.",
                  "Human labels map majority button states in nominal four-frame windows to the nearest of the 12 model actions; they are not raw button frequencies.",
                  "Bootstrap intervals resample participants or episodes, never individual frames. Human dots show individual participants.",
                  "Human and model observations/starts differ. This is descriptive behavior, not matched-human-state policy accuracy or fMRI alignment.",
                  "Human-only levels remain visible with model entries missing, not zero. Historical confirmation samples are not pooled with development.",
              ]}
    for level in levels:
        human = collect_humans(args.human_dir, level)
        source = args.development if level in original["levels"] else args.extension if level in extension["evaluation_levels"] else None
        model = collect_models(source, level) if source else None
        report["levels"][level] = {"human": human, "model": model,
                                   "missing_model_reason": None if model else "No completed model evaluation in the all-level development set"}
        print(f"{level}: {human['n_units']} participants, {human['n_segments']} segments; model={'30 episodes per condition' if model else 'unavailable'}", flush=True)
    return report


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#555555", "#0072B2", "#D55E00"]
    variants = {
        "probabilities": ("policy_probabilities", "search_visits", ["Human actions", "Policy-head probabilities", "Raw search visits"]),
        "action_choices": ("policy_argmax", "executed_actions", ["Human actions", "Head argmax (hypothetical)", "Search actions (executed)"])}
    for variant, (head, search, names) in variants.items():
        with PdfPages(out / f"{variant}_by_level.pdf") as pdf:
            for level, row in report["levels"].items():
                fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, sharey=True)
                for ax, condition in zip(axes, CONDITIONS):
                    model = row["model"]["conditions"][condition] if row["model"] else None
                    groups = [row["human"], model[head] if model else None, model[search] if model else None]
                    x = np.arange(12)
                    for offset, group, color in zip((-.26,0,.26), groups, colors):
                        if group is None:
                            continue
                        means = np.asarray(group["mean"])*100
                        ci = np.asarray(group["bootstrap_ci95"])*100
                        ax.bar(x+offset, means, width=.24, color=color)
                        ax.errorbar(x+offset, means, yerr=np.maximum(0,np.stack([means-ci[0],ci[1]-means])),
                                    fmt="none", ecolor=color, capsize=2, lw=.8)
                    for i, unit in enumerate(row["human"]["unit_distributions"]):
                        ax.scatter(x-.26+(i-2)*.023, np.asarray(unit)*100, s=10, facecolors="white", edgecolors=colors[0], linewidths=.5, zorder=4)
                    title = ("Greedy rollouts (T=0)" if condition == "greedy" else "Sampled rollouts (T=0.25)")
                    if model:
                        title += f" · {model['n_episodes']} episodes · {model['completions']} completions · {model[head]['n_decisions']:,} decisions"
                    else:
                        title = "Human only — model evaluation unavailable"
                        ax.text(.5,.7,"No model distribution available; missing values are not zero",transform=ax.transAxes,ha="center",color="#555555")
                    ax.set_title(title, loc="left", fontsize=11)
                    ax.set_ylabel("Average probability (%)" if variant=="probabilities" else "Share of action choices (%)")
                    ax.set_ylim(0,100)
                    ax.grid(axis="y",alpha=.18)
                    ax.set_axisbelow(True)
                axes[-1].set_xticks(np.arange(12), SHORT_LABELS, rotation=35, ha="right")
                control = " · control" if level=="Level6-1" else ""
                fig.suptitle(f"{level[5:]}{control} · Humans, policy head and search\n{row['human']['n_units']} participants · {row['human']['n_segments']} human segments",fontsize=17,y=.99)
                fig.legend(handles=[Patch(color=c,label=n) for c,n in zip(colors,names)],loc="upper center",bbox_to_anchor=(.5,.915),ncol=3,frameon=False)
                specific = ("Policy and raw visits share the same model states. Raw visits precede temperature and action selection." if variant=="probabilities" else
                            "Head argmax is hypothetical on logged model states; search actions were executed. This is not a policy-only rollout.")
                fig.text(.5,.025,"All outcomes included · equal participant / episode weights · 95% bootstrap intervals · dots: individual humans\n"+
                         specific+"\nHuman labels use the 12 model actions in four-frame windows. Human and model groups visit different states.",ha="center",fontsize=9)
                fig.tight_layout(rect=(0,.11,1,.86))
                pdf.savefig(fig)
                if variant=="probabilities":
                    fig.savefig(out/f"probabilities_{level}.png",dpi=150)
                plt.close(fig)
    # Two-page overview: exactly the same scale, action order and levels in all panels.
    levels = list(report["levels"])
    with PdfPages(out/"probabilities_overview.pdf") as pdf:
        for condition in CONDITIONS:
            fig, axes = plt.subplots(1,3,figsize=(19,12),sharey=True)
            for ax,key,title in zip(axes,("human","policy_probabilities","search_visits"),variants["probabilities"][2]):
                matrix=[]
                for level in levels:
                    row=report["levels"][level]
                    group=row["human"] if key=="human" else row["model"]["conditions"][condition][key] if row["model"] else None
                    matrix.append(np.asarray(group["mean"])*100 if group else np.full(12,np.nan))
                values=np.asarray(matrix)
                cmap=plt.get_cmap("Blues").copy(); cmap.set_bad("#eeeeee")
                im=ax.imshow(values,aspect="auto",vmin=0,vmax=100,cmap=cmap)
                for i in range(len(levels)):
                    for j in range(12):
                        v=values[i,j]
                        ax.text(j,i,"—" if np.isnan(v) else f"{v:.1f}",ha="center",va="center",fontsize=6.5,color="white" if v>60 else "#333333")
                ax.set_title(title,fontsize=13)
                ax.set_xticks(np.arange(12),SHORT_LABELS,rotation=60,ha="right",fontsize=8)
                ax.set_yticks(np.arange(len(levels)),[level[5:]+(" · control" if level=="Level6-1" else "") for level in levels],fontsize=9)
            fig.suptitle(f"Human actions → policy-head probabilities → raw search visits\n{condition.capitalize()} rollout states · all human-covered levels",fontsize=17,y=.98)
            fig.subplots_adjust(left=.07,right=.93,bottom=.20,top=.90,wspace=.12)
            bar=fig.add_axes([.945,.24,.012,.60]); fig.colorbar(im,cax=bar,label="Average share (%)")
            fig.text(.5,.035,"All outcomes included · equal participant / model-episode weights · model n=30 per level and condition\n"
                     "Head and search share the same model states; human states differ. Search visits are probabilities, not executed actions.\n"
                     "Grey cells: model data unavailable (5-3). Human action labels are quantized four-frame windows.",ha="center",fontsize=10)
            pdf.savefig(fig)
            fig.savefig(out/f"probabilities_overview_{condition}.png",dpi=150)
            plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir",type=Path,default=Path("outputs/human_trajectories"))
    parser.add_argument("--development",type=Path,default=Path("outputs/controller_policy/v1b-20260907"))
    parser.add_argument("--extension",type=Path,default=Path("outputs/controller_policy/coverage-v1-20260911"))
    parser.add_argument("--out",type=Path,default=Path("images/action_policy_search"))
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Corpus/trace analysis and plotting require a compute node")
    args.out.mkdir(parents=True,exist_ok=True)
    report=collect(args)
    (args.out/"distributions.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.out/"distributions.csv").open("w") as f:
        writer=csv.writer(f)
        writer.writerow(["level","condition","group","action","mean_fraction","ci95_low","ci95_high","n_units","n_decisions"])
        for level,row in report["levels"].items():
            groups=[("human","human",row["human"])]
            if row["model"]:
                groups += [(condition,key,row["model"]["conditions"][condition][key]) for condition in CONDITIONS for key in GROUPS]
            for condition,key,data in groups:
                for i,label in enumerate(LABELS):
                    writer.writerow([level,condition,key,label,data["mean"][i],data["bootstrap_ci95"][0][i],data["bootstrap_ci95"][1][i],data["n_units"],data["n_decisions"]])
    plot(report,args.out)
    print(f"Wrote plots and data to {args.out}",flush=True)


if __name__=="__main__":
    main()
