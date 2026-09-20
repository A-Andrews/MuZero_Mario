"""Verify exact control replay and retained control provenance before expansion."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_controller_diagnostic import atomic_json, file_sha


def check(experiment):
    plan = json.loads((experiment / "manifest.json").read_text())
    reference = Path(plan["reused_control"]["experiment"])
    old = json.loads((reference / "manifest.json").read_text())
    if plan["trials"] != old["trials"]:
        raise ValueError("Development trial list changed")
    if plan["levels"]["Level6-1"]["checkpoint_sha256"] != old["levels"]["Level6-1"]["checkpoint_sha256"]:
        raise ValueError("Control checkpoint changed")
    summary = json.loads((reference / "development/Level6-1/summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Reused control is incomplete")
    comparisons = {}
    for name in ("greedy", "sampled"):
        original_path = reference / "development/Level6-1" / name / "episode_0000.json"
        new_path = experiment / "smoke/Level6-1" / name / "episode_0000.json"
        a, b = json.loads(original_path.read_text()), json.loads(new_path.read_text())
        keys = ("status", "completed", "died", "timed_out", "steps", "final_x", "initial_observation_sha256",
                "env_seed", "search_seed", "noop_frames_requested")
        if any(a[k] != b[k] for k in keys):
            raise ValueError(f"{name}: smoke control outcome differs from original")
        # Manifest metadata differs, so compare trajectory arrays directly.
        import numpy as np
        with np.load(original_path.with_name("episode_0000_trace.npz"), allow_pickle=False) as x, np.load(new_path.with_name("episode_0000_trace.npz"), allow_pickle=False) as y:
            for field in ("actions", "x_before", "x_after", "rewards", "policies", "priors"):
                if not np.array_equal(x[field], y[field]):
                    raise ValueError(f"{name}: control trace differs in {field}")
        comparisons[name] = {"matched": True, "source_result_sha256": file_sha(original_path),
                             "new_result_sha256": file_sha(new_path)}
    atomic_json(experiment / "control_validation.json", {"passed": True, "comparisons": comparisons,
                "reused_control_summary": summary, "note": "Smoke repeats are not added to the existing 30-trial control denominator."})
    print("Exact Level6-1 control reproduction passed; original n=30 per arm reused", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Compute nodes only")
    check(args.experiment)
