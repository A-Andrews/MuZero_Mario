"""Freeze diagnostic source, checkpoint identities, and a predeclared trial plan.

Run on a Slurm compute node; checkpoint hashing reads substantial files.
Checkpoints are copied once to protect the experiment from best.pt replacement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(template, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Prepare on a Slurm compute node, not the login node")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Already prepared: {manifest_path}; reuse it without preparing again")
    plan = json.loads(Path(template).read_text())
    if "Level6-1" not in plan["levels"]:
        raise ValueError("Level6-1 control is required")
    plan["template_sha256"] = sha256(template)
    snapshot = output / "source"
    if snapshot.exists():
        raise FileExistsError(f"Partial preparation exists at {snapshot}; use a new output directory")
    snapshot.mkdir()
    for directory in ("src", "scripts", "tests", "conf", "docs"):
        shutil.copytree(ROOT / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_hashes = {
        str(path.relative_to(snapshot)): sha256(path)
        for path in sorted(snapshot.rglob("*")) if path.is_file()
    }
    plan["source_sha256"] = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    (output / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    try:
        plan["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        (output / "working_changes.patch").write_bytes(subprocess.check_output(
            ["git", "diff", "--binary"], cwd=ROOT))
    except subprocess.CalledProcessError:
        plan["git_head"] = "unavailable"
    frozen = output / "checkpoints"
    frozen.mkdir()
    for level, settings in plan["levels"].items():
        source = (ROOT / settings["checkpoint"]).resolve()
        target = frozen / f"{level}.pt"
        initial = sha256(source)
        if settings.get("expected_checkpoint_sha256", initial) != initial:
            raise ValueError(f"Checkpoint differs from the development experiment: {source}")
        shutil.copyfile(source, target)
        copied = sha256(target)
        if copied != initial:
            raise RuntimeError(f"Checkpoint changed while copying {source}")
        settings["original_checkpoint"] = str(source)
        settings["checkpoint"] = str(target)
        settings["checkpoint_sha256"] = copied
    plan["prepared_by_slurm_job"] = os.environ["SLURM_JOB_ID"]
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    tmp.replace(manifest_path)
    print(f"Prepared {manifest_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="docs/controller_policy_diagnostic_v1.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prepare(args.template, args.out)
