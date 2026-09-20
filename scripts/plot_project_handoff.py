"""Render the internship handoff from its portable, source-hashed evidence snapshot.

No emulator, checkpoints, torch, cluster filesystem, or network required.
Run from any directory: python scripts/plot_project_handoff.py
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "muzero-handoff-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"greedy": "#2878A8", "sampled": "#D77C22", "stall_sampled": "#138A72"}
LABELS = {"greedy": "Greedy MCTS", "sampled": "Always sampled", "stall_sampled": "Stall-triggered"}


def validate(data):
    rows = data["benchmark"]
    assert len(rows) == 22 and sum(bool(r["arms"]) for r in rows) == 21
    assert {r["level"] for r in rows}.isdisjoint({"Level2-2", "Level7-2"})
    assert [r["level"] for r in rows if not r["arms"]] == ["Level5-3"]
    for r in rows:
        h = r["human"]
        assert np.isclose(h["no_death_rate"], h["n_no_death"] / h["n_reps"])
        for a in r["arms"].values():
            assert a["n"] == a["completed"] + a["timeouts"] + a["deaths"] == 30
            assert len(a["successful_steps"]) == a["completed"]
            assert np.isclose(a["rate"], a["completed"] / a["n"])
    for r in data["confirmation"]:
        assert len(r["conditions"]) == 3
        for a in r["conditions"]:
            assert a["complete"] and not a["infrastructure_errors"]
            assert a["level_completions"] + a["deaths"] + a["timeouts"] == a["finished_trials"] == 100
    assert len({r["root_id"] for r in data["commitment_roots"]}) == 56


def save(fig, out, name, footer):
    fig.text(.035, .015, footer, fontsize=9, color="#465365", va="bottom")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, facecolor="white")
        if ext == "svg":
            path = out / f"{name}.{ext}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def benchmark(data, out):
    rows = data["benchmark"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 12), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.17, wspace=.13)
    fig.suptitle("Specialists can outperform the recorded human reference on some levels\nBroad, reliable human-level play is not yet established", x=.06, ha="left", fontsize=19)
    rate, speed = axes
    for i, row in enumerate(rows):
        h = row["human"]
        rate.plot(100*h["no_death_rate"], i, "D", color="#343A40", ms=5,
                  label="Human first life (corpus reference)" if i == 0 else None)
        speed.plot([h["length_p10"]/15, h["length_p90"]/15], [i, i], color="#BDC4CA", lw=6)
        speed.plot(h["length_median"]/15, i, "|", color="#343A40", ms=12)
        for arm, offset in [("greedy", -.16), ("sampled", .16)]:
            if arm not in row["arms"]:
                continue
            a = row["arms"][arm]; p = a["rate"]*100
            lo, hi = np.array(a["wilson95"])*100
            rate.errorbar(p, i+offset, xerr=[[max(0, p-lo)], [max(0, hi-p)]],
                          fmt="o", ms=5, color=COLORS[arm], capsize=2,
                          label=LABELS[arm]+" (95% Wilson)" if i == 0 else None)
            if a["successful_median"] is not None:
                speed.plot(a["successful_median"]/15, i+offset, "o", ms=5, color=COLORS[arm])
        if not row["arms"]:
            rate.text(43, i, "MODEL NOT EVALUATED", fontsize=8, va="center", color="#8E4555")
            speed.text(110, i+.2, "No model benchmark", fontsize=8, color="#8E4555")
    rate.set_yticks(range(len(rows)), [r["level"].replace("Level", "") for r in rows])
    rate.invert_yaxis(); rate.set_xlim(-3, 103)
    rate.set_xlabel("First-life completion (%) • every model attempt counted")
    speed.set_xlabel("Successful duration (emulated seconds) • lower is faster")
    rate.set_title("Reliability | 30 development attempts per model/controller", fontsize=11, loc="left")
    speed.set_title("Speed | human p10–p90 band, median tick; model median dots", fontsize=11, loc="left")
    speed.set_xlim(left=0)
    for ax in axes: ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True)
    handles, labels = rate.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .085),
               ncol=3, fontsize=9, frameon=False)
    save(fig, out, "01_human_context", "22 human-covered levels; 21 evaluated specialists. Human rates pool repetitions from five participants, without population CIs.\nHuman speed includes successful respawn lives; starts, practice and budgets differ. 15 decisions/s is simulated game time, not inference speed.\nIntervals reflect fixed-checkpoint trial uncertainty, not independent training runs. Missing model results are not zero scores. Source: docs/handoff/evidence.json.")


def failure_balance(data, out):
    fig, ax = plt.subplots(figsize=(12, 5.2)); fig.subplots_adjust(left=.18, right=.97, top=.72, bottom=.29)
    fig.suptitle("Always-on sampling removes stalls, but increases deaths", x=.04, ha="left", fontsize=19)
    for i, arm in enumerate(("greedy", "sampled")):
        counts = [sum(r["arms"].get(arm, {}).get(key, 0) for r in data["benchmark"])
                  for key in ("completed", "timeouts", "deaths")]
        left = 0
        for value, color in zip(counts, ("#138A72", "#D8AC43", "#BC5267")):
            ax.barh(i, value, left=left, color=color, height=.5)
            if value: ax.text(left+value/2, i, str(value), ha="center", va="center", color="white", weight="bold", fontsize=14)
            left += value
    ax.set_yticks([0, 1], [LABELS["greedy"], LABELS["sampled"]]); ax.invert_yaxis()
    ax.set_xlim(0, 630); ax.set_xlabel("Attempts across 21 evaluated human-covered levels (30 each)")
    ax.legend(handles=[Patch(color=c, label=l) for c,l in zip(("#138A72", "#D8AC43", "#BC5267"), ("Completed", "Timed out", "Died"))], loc="upper center", bbox_to_anchor=(.5, 1.35), ncol=3, frameon=False)
    save(fig, out, "02_failure_balance", "Development evidence, fixed checkpoints: 203/630 (32.2%) greedy versus 202/630 (32.1%) sampled completions.\nEqual attempts per level; no claim of universal controller superiority. Level5-3 is missing; 2-2/7-2 are excluded for absent human data.\nThese totals intentionally differ from the historical 22-model-level report, which included 2-2.")


def confirmation(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    fig.subplots_adjust(top=.77, bottom=.25, wspace=.18)
    fig.suptitle("Intervene at stalls: a confirmed controller improvement on three levels", x=.04, ha="left", fontsize=18)
    for ax, row in zip(axes, data["confirmation"]):
        for i, a in enumerate(row["conditions"]):
            p = a["level_completions"]; lo,hi=np.array(a["wilson95_observed"])*100
            ax.bar(i, p, color=COLORS[a["condition"]], width=.65)
            ax.errorbar(i,p,yerr=[[max(0,p-lo)],[max(0,hi-p)]], color="#263442", capsize=4)
            ax.text(i, min(p+10,111), f"{p}/100", ha="center", fontsize=12, weight="bold")
        h=next(r["human"]["no_death_rate"] for r in data["benchmark"] if r["level"]==row["level"])
        ax.axhline(h*100, color="#343A40", ls="--", lw=1)
        ax.set_title(row["level"].replace("Level", "Level "))
        ax.set_xticks(range(3), ["Greedy", "Always\nsampled", "Stall-\ntriggered"])
        ax.set_ylim(0,118)
    axes[0].set_ylabel("Completion (%) • 100 new seed pairs per arm/level")
    save(fig,out,"03_confirmed_controller", "900 completed episodes; new confirmation seeds, unchanged checkpoints, 50 simulations and no root noise. Error bars: Wilson 95%.\nDashed line: descriptive human first-life corpus rate (protocols differ). Stall trigger uses extra RAM x/player-state input.\nOn 1-1, gated sampling rescues 76/78 greedy timeouts but loses one greedy success. Evidence does not establish 86% > 83% or all-level benefit.")


def mechanisms(data, out):
    specs=[("6-1 obstacle, late", "Level6-1", "obstacle_1394",222),
           ("6-1 pit, early", "Level6-1", "pit_2807",270),
           ("6-1 successful references", "Level6-1", "successful_reference",None),
           ("1-1 pipe failures", "Level1-1", "pipe_898",None),
           ("1-1 successful references", "Level1-1", "successful_reference",None)]
    fig,(ax,gap)=plt.subplots(1,2,figsize=(16,7),gridspec_kw={"width_ratios":[1,1.2]})
    fig.subplots_adjust(left=.20,right=.96,top=.76,bottom=.26,wspace=.35)
    fig.suptitle("Two distinct clues: action timing and imagined-value inconsistency",x=.03,ha="left",fontsize=19)
    vals=[]; labels=[]; cells=[]
    for label, lv, sig, step in specs:
        roots=[r for r in data["commitment_roots"] if r["level"]==lv and r["signature"]==sig and r["condition"]=="greedy" and (step is None or r["step"]==step)]
        counts=[(sum(r["holds"][str(h)]["completed"] for r in roots),sum(r["holds"][str(h)]["n"] for r in roots)) for h in [1,2,4,8]]
        vals.append([k/n for k,n in counts]); cells.append(counts)
        labels.append(f"{label}\n{len(set(r['episode'] for r in roots))} episodes / {len(roots)} roots")
    ax.imshow(vals,cmap="Blues",vmin=0,vmax=1,aspect="auto")
    for y,row in enumerate(cells):
        for x,(k,n) in enumerate(row): ax.text(x,y,f"{k}/{n}",ha="center",va="center",color="white" if k/n>.55 else "#17324D",fontsize=11)
    ax.set_yticks(range(len(labels)),labels,fontsize=10); ax.set_xticks(range(4),["1 (normal)","2","4","8"])
    ax.set_xlabel("Decisions holding the controller's own first action")
    ax.set_title("Conditional completions: holding can rescue AND harm",fontsize=11,pad=14)
    for lv,group,color,label in [("Level1-1","pipe_898","#BC5267","1-1 pipe failures"),("Level1-1","successful_reference","#2878A8","1-1 successful references")]:
        records=sorted([r for r in data["value_gaps"] if r["level"]==lv and r["group"]==group],key=lambda r:r["horizon"])
        for key,style,desc in [("imagined_abs_gap","-","imagined"),("reanchored_abs_gap","--","reanchored")]:
            gap.plot([r["horizon"] for r in records],[r[key] for r in records],style,color=color,marker="o",ms=4,label=f"{label}: {desc}")
    gap.set_xlabel("Diagnostic prefix horizon (decisions)"); gap.set_ylabel("Mean absolute gap to observed-state value\n(reward/value units)")
    gap.set_title("Reanchoring reduces the pipe group's multi-step gap",fontsize=11,pad=14)
    gap.grid(alpha=.18); gap.legend(fontsize=8,frameon=False)
    save(fig,out,"04_failure_mechanisms", "Left: selected greedy-source groups; 10 conditional seeds/root, not independent level attempts. 1 decision = 4 emulator frames.\nRight: equally weighted roots × four diagnostic prefixes; 18 pipe roots from 9 episodes, 7 reference roots from 3 episodes. No population CIs.\nObserved-state values come from the same network, not ground truth. Fixed prefixes need not be paths explored by MCTS.\nThese results do not isolate dynamics, BatchNorm, or a universal action-hold fix. Full evidence and other groups: docs/VALUE_COMMITMENT_RESULTS_V1.md.")


def architecture(out):
    fig,ax=plt.subplots(figsize=(15,8)); fig.subplots_adjust(left=.02,right=.98,top=.9,bottom=.16)
    ax.set_xlim(0,15); ax.set_ylim(0,8); ax.axis("off")
    fig.suptitle("What MuZero does at each decision — and where failures can arise",x=.04,ha="left",fontsize=19)
    def box(x,y,w,h,title,body,color="#EAF2F8"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.12",facecolor=color,edgecolor="#CBD5DF"))
        ax.text(x+.15,y+h-.25,title,weight="bold",fontsize=12,va="top")
        ax.text(x+.15,y+h-.8,body,fontsize=10,va="top",linespacing=1.5)
    def arrow(a,b,text=None):
        ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","color":"#536273","lw":1.7})
        if text: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.15,text,ha="center",fontsize=9)
    box(.2,5.4,2.5,1.8,"Observe","RGB → grayscale + pooling\n4 × 96 × 96 frame stack")
    box(3.2,5.4,2.7,1.8,"Represent: h(o)","Encode the real observation\nas a compact hidden state")
    box(6.4,5.4,3.0,1.8,"Predict: f(s)","Policy: prior over 12 actions\nValue: expected future return")
    box(6.4,2.9,3.0,1.7,"Imagine: g(s, a)","Next hidden state + reward\nRepeat prediction at new state")
    box(10,5.4,4.3,1.8,"Search + act","MCTS combines priors, rewards, values\nVisits → greedy or sampled action\nExecute 4 frames; observe again")
    arrow((2.8,6.3),(3.1,6.3)); arrow((6,6.3),(6.3,6.3)); arrow((9.5,6.3),(9.9,6.3))
    arrow((10.4,5.3),(9.5,3.8),"try actions"); arrow((7.9,4.7),(7.9,5.3))
    box(.2,2.9,5.4,1.7,"Learn from replay","Search visits teach policy; rewards and returns teach predictions.\nFive recurrent training steps + latent consistency loss.\nSelected runs also learn from human demonstrations.","#EAF5ED")
    box(.2,.25,4.4,1.9,"Observed: action selection","Greedy can repeat an unhelpful action.\nTargeted sampling releases many stalls.\nExtra RAM input belongs to that controller.","#FFF3DD")
    box(5.1,.25,4.4,1.9,"Observed: sequence sensitivity","A useful initial jump can be abandoned.\nHolding it rescues some tested roots,\nbut harms successful references.","#FFF3DD")
    box(10,.25,4.4,1.9,"Open cause: learned evaluation","Imagined and observed values disagree.\nObserved values can also misjudge returns.\nLeaf substitution has selective effects.","#FBECEF")
    save(fig,out,"05_inside_muzero","Conceptual map, not a measured attribution diagram. MuZero predicts task-relevant latent states, not future screenshots.\nThe benchmark uses 50 simulations with leaf batch 4. The network is reused after every real action; search does not commit to a whole plan.\nRead docs/handoff/REPORT.md for evidence strength, human comparison limits and the next discriminating experiments.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,default=ROOT/"docs/handoff/evidence.json")
    parser.add_argument("--out",type=Path,default=ROOT/"images/handoff")
    args=parser.parse_args(); data=json.loads(args.evidence.read_text()); validate(data)
    args.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    benchmark(data,args.out); failure_balance(data,args.out); confirmation(data,args.out); mechanisms(data,args.out); architecture(args.out)
    with (args.out/"benchmark.csv").open("w",newline="") as f:
        writer=csv.writer(f, lineterminator="\n"); writer.writerow(["level","human_first_life_completed","human_repetitions","human_rate","controller","model_completed","model_attempts","model_deaths","model_timeouts"])
        for r in data["benchmark"]:
            h=r["human"]
            for arm,a in (r["arms"].items() or [("not_evaluated",{})]):
                writer.writerow([r["level"],h["n_no_death"],h["n_reps"],h["no_death_rate"],arm]+[a.get(k,"") for k in ["completed","n","deaths","timeouts"]])
    (args.out/"provenance.json").write_text(json.dumps({"as_of":data["as_of"],"evidence_sha256":hashlib.sha256(args.evidence.read_bytes()).hexdigest(),"renderer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"sources":data["sources"],"matplotlib":matplotlib.__version__,"numpy":np.__version__},indent=2)+"\n")
    print(f"Validated evidence; wrote five figures in PNG/PDF/SVG, benchmark.csv and provenance.json to {args.out}")


if __name__ == "__main__":
    main()
