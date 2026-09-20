"""Check unchanged candidate provenance and exact development-seed replay."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_controller_diagnostic import atomic_json, file_sha, validate_manifest


def check(root, smoke=False):
    plan = json.loads((root / "manifest.json").read_text())
    validate_manifest(plan)
    reference = Path(plan["development_experiment"])
    old = json.loads((reference / "manifest.json").read_text())
    for key in ("confirmation_trials", "previous_confirmation_trials", "conditions",
                "stall_controller", "max_steps", "num_simulations", "bank_stride", "bank_tail"):
        if plan[key] != old[key]:
            raise ValueError(f"Changed development protocol: {key}")
    if plan["split"] != "confirmation" or plan["development_trials"] != old["trials"]:
        raise ValueError("Incorrect split or development provenance")
    if set(plan["levels"]) != set(old["levels"]):
        raise ValueError("Changed level set")
    # The evaluator's manifest validation changes; inference, environment,
    # search, and the complete stall-controller implementation must not change.
    hashes = json.loads((reference / "source_hashes.json").read_text())
    checked = {}
    for name, digest in hashes.items():
        if name.startswith("src/") or name == "scripts/stall_sampling_controller.py":
            if file_sha(root / "source" / name) != digest:
                raise ValueError(f"Changed behavior source: {name}")
            checked[name] = digest
    comparisons = []
    for level, settings in plan["levels"].items():
        if settings["checkpoint_sha256"] != old["levels"][level]["checkpoint_sha256"]:
            raise ValueError(f"Changed checkpoint: {level}")
        if not smoke:
            continue
        for condition in plan["conditions"]:
            name = condition["name"]
            a = reference / "development" / level / name / "episode_0000.json"
            b = root / "smoke" / level / name / "episode_0000.json"
            original, replay = json.loads(a.read_text()), json.loads(b.read_text())
            keys = ("status", "completed", "died", "timed_out", "steps", "final_x",
                    "initial_observation_sha256", "env_seed", "search_seed", "noop_frames_requested")
            if any(original[k] != replay[k] for k in keys):
                raise ValueError(f"Changed smoke outcome: {level}/{name}")
            with np.load(a.with_name("episode_0000_trace.npz"), allow_pickle=False) as x, np.load(b.with_name("episode_0000_trace.npz"), allow_pickle=False) as y:
                if set(x.files) != set(y.files):
                    raise ValueError("Changed trace fields")
                for field in x.files:
                    numeric = x[field].dtype.kind in "fc"
                    if not np.array_equal(x[field], y[field], equal_nan=numeric):
                        raise ValueError(f"Changed smoke trace: {level}/{name}/{field}")
            comparisons.append({"level": level, "condition": name, "matched": True})
    atomic_json(root / ("control_validation.json" if smoke else "protocol_validation.json"),
                {"passed": True, "development_manifest_sha256": file_sha(reference / "manifest.json"),
                 "unchanged_source_hashes": checked, "smoke_comparisons": comparisons,
                 "note": "Smoke uses old development seeds only; all confirmation baselines run afresh."})
    print(f"Unchanged candidate verified; {len(comparisons)} exact smoke replays", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Compute nodes only")
    check(args.experiment, args.smoke)
