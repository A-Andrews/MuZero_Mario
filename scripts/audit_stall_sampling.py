"""Audit completed gated-controller records and equality before intervention."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_controller_diagnostic import atomic_json, canonical_sha, file_sha


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    report = {"manifest_sha256": canonical_sha(manifest), "script_sha256": file_sha(__file__),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "levels": {}, "complete": False}
    for level in manifest["levels"]:
        folder = root / manifest["split"] / level
        summary = json.loads((folder / "summary.json").read_text())
        if not summary["complete"]:
            raise ValueError(f"Incomplete {level}")
        identity_sha = canonical_sha(summary["identity"])
        data = {}
        for condition in ("greedy", "sampled", "stall_sampled"):
            data[condition] = []
            for trial in manifest["trials"]:
                path = folder / condition / f"episode_{trial['episode_index']:04d}.json"
                record = json.loads(path.read_text())
                if record["status"] != "ok" or record["identity_sha256"] != identity_sha or any(record[k] != v for k,v in trial.items()):
                    raise ValueError(f"Invalid episode identity: {path}")
                trace_path = path.with_name(path.stem+"_trace.npz")
                if file_sha(trace_path) != record["artifacts"][trace_path.name]:
                    raise ValueError(f"Trace hash mismatch: {trace_path}")
                data[condition].append((record, trace_path))
        checks = []
        for (greedy, gp), (gated, cp) in zip(data["greedy"], data["stall_sampled"]):
            with np.load(gp, allow_pickle=False) as g, np.load(cp, allow_pickle=False) as c:
                triggers = np.flatnonzero(c["rescue_trigger"])
                cutoff = int(triggers[0]) if len(triggers) else len(c["actions"])
                if len(triggers) != gated["rescue_triggers"] or int((c["controller_temperature"]>0).sum()) != gated["sampled_decisions"]:
                    raise ValueError("Controller counters disagree with trace")
                for key in ("actions", "x_before", "x_after", "rewards", "priors", "policies"):
                    if not np.array_equal(g[key][:cutoff], c[key][:cutoff]):
                        raise ValueError(f"{level}: episode {gated['episode_index']} differs before first trigger in {key}")
                if not len(triggers) and len(g["actions"]) != len(c["actions"]):
                    raise ValueError("Untriggered run differs in length")
                checks.append({"episode_index": gated["episode_index"], "first_trigger": int(triggers[0]) if len(triggers) else None,
                               "matching_prefix_decisions": cutoff, "gated_completed": gated["completed"],
                               "greedy_completed": greedy["completed"], "gated_death": gated["died"]})
        gated_rows = [r for r,_ in data["stall_sampled"]]
        comparisons = {}
        for reference in ("greedy", "sampled"):
            pairs = [(r,g) for (r,_),g in zip(data[reference],gated_rows)]
            comparisons[reference] = {"rescues": sum(not r["completed"] and g["completed"] for r,g in pairs),
                                      "lost_successes": sum(r["completed"] and not g["completed"] for r,g in pairs),
                                      "saved_timeouts": sum(r["timed_out"] and g["completed"] for r,g in pairs)}
        sampled_steps = sum(r["sampled_decisions"] for r in gated_rows)
        total_steps = sum(r["steps"] for r in gated_rows)
        report["levels"][level] = {"n_per_arm": len(gated_rows), "all_prefixes_match": True,
                                    "paired_comparisons": comparisons, "episode_checks": checks,
                                    "triggered_episodes": sum(r["rescue_triggers"]>0 for r in gated_rows),
                                    "bursts": sum(r["rescue_triggers"] for r in gated_rows),
                                    "sampled_decisions": sampled_steps, "total_decisions": total_steps,
                                    "pooled_sampled_fraction": sampled_steps/total_steps,
                                    "mean_episode_sampled_fraction": float(np.mean([r["sampled_decision_fraction"] for r in gated_rows]))}
        n = len(gated_rows)
        print(f"{level}: verified all {n} paired prefixes and {3*n} trace hashes", flush=True)
    report["complete"] = True
    atomic_json(root / "audit.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run trace analysis on a compute node")
    audit(args.experiment)
