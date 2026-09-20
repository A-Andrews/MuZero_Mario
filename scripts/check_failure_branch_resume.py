"""Check project file quota, write access, and saved branches before exact retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Compute nodes only")
    root = args.experiment
    plan = json.loads((root/"branch_plan.json").read_text())
    quota = subprocess.check_output(["lfs", "quota", "-p", "1483806624", "/projects"], text=True)
    print(quota, flush=True)
    row = next(line.split() for line in quota.splitlines() if line.strip().startswith("/projects"))
    free_files = int(row[7])-int(row[5].rstrip("*"))
    if free_files < 12000:
        raise RuntimeError(f"Only {free_files} project file slots remain; need 12000 before retrying")
    # Creates and removes only this job's own temporary storage probe.
    with tempfile.TemporaryDirectory(prefix="resume-write-probe-", dir=root) as tmp:
        path = Path(tmp)/"probe"
        with path.open("wb") as f:
            f.write(b"\0"*(16*1024*1024))
            f.flush()
            os.fsync(f.fileno())
    report = {"job_id": os.environ["SLURM_JOB_ID"], "passed": False,
              "free_project_file_slots": free_files, "quota": quota, "cases": [], "saved_branches": 0}
    for case in plan["cases"]:
        folder = root/"full"/f"case_{case['case_id']:03d}"
        paths = sorted(folder.glob("root_*_seed_*.json"))
        groups = {}
        if paths:
            identity = json.loads((folder/"identity.json").read_text())
            # Use the frozen experiment helper's exact canonical encoding.
            import sys
            sys.path.insert(0, str(root/"source"))
            from scripts.eval_controller_diagnostic import canonical_sha
            identity_sha = canonical_sha(identity)
            if identity["plan_sha256"] != canonical_sha(plan) or identity["case"] != case:
                raise ValueError("Saved case identity changed")
            if identity["script_sha256"] != digest(root/"source/scripts/diagnose_failure_branches.py"):
                raise ValueError("Frozen runner changed")
            seen = set()
            for path in paths:
                b = json.loads(path.read_text())
                key = (b["root_index"], b["seed"], b["arm"])
                if key in seen or b["identity_sha256"] != identity_sha or digest(path.with_suffix(".npz")) != b["trace_sha256"]:
                    raise ValueError(f"Changed or duplicate saved branch: {path}")
                seen.add(key)
                group = groups.setdefault(f"root_{b['root_index']}/{b['arm']}", {"n": 0, "completions": 0, "deaths": 0, "timeouts": 0})
                group["n"] += 1
                group["completions"] += b["completed"]
                group["deaths"] += b["died"]
                group["timeouts"] += b["timed_out"]
        report["saved_branches"] += len(paths)
        report["cases"].append({"case_id": case["case_id"], "level": case["level"],
                                "group": case["group"], "saved_branches": len(paths), "groups": groups})
    report["passed"] = True
    (root/"resume_preflight.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Verified {report['saved_branches']} saved branches; storage write check passed", flush=True)


if __name__ == "__main__":
    main()
