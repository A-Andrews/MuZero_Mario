#!/usr/bin/env python
"""Package the per-level specialist checkpoints into a self-contained zip.

Produces an archive a collaborator can unzip and use with nothing from this
repo and no stable-retro/ROM install:

    muzero_mario_models_<date>/
      checkpoints/Level1-1.pt ...   weights + cfg_snapshot, no optimizer state
      code/src/muzero/{networks,transforms}.py
      code/src/env/preprocess.py
      load_model.py                 loader + activation extraction
      manifest.json                 provenance + sha256 per checkpoint
      README.md

Checkpoint selection is by recorded self-play completion rate: for each level
the T6 `spec-<level>` and the T7 rescue `spec-imit-<level>` runs compete on
`checkpoints/best.json` and the higher rate wins, so the imitation rescues take
1-3 and 4-3 on merit rather than by a hard-coded rule (same policy as
`eval_human_benchmark.py`). Greedy run-through stats are attached from
`images/human_vs_agent_runthrough.json` where the run matches.

Usage:
    python scripts/package_models.py                    # models only -> zip
    python scripts/package_models.py --no-zip           # staging dir only
    python scripts/package_models.py --human-data       # + 1.9G corpus archive
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs" / "runs"
BENCHMARK = REPO / "images" / "human_vs_agent_runthrough.json"
HUMAN_DIR = REPO / "outputs" / "human_trajectories"
DEFAULT_OUT = Path("/projects/u6oz/atdandrews/MuZero_Mario/exports")

STATE_DIR = REPO / "mario.stimuli" / "SuperMarioBros-Nes"


def all_levels() -> list[str]:
    """Every level the integration ships a state file for, world-stage ordered.

    Derived rather than hard-coded so a new specialist fleet (T9 added worlds
    5-8) lands in the bundle without editing this script.
    """
    def key(name: str):
        w, _, st = name.removeprefix("Level").partition("-")
        return (int(w), (0, int(st)) if st.isdigit() else (1, 0))
    return sorted((p.stem for p in STATE_DIR.glob("Level*.state")), key=key)

# Files copied verbatim so the bundle can rebuild the net and the observation
# pipeline standalone. src/env/__init__.py is deliberately NOT copied: it pulls
# in the whole stable-retro emulation stack, which the bundle must not need.
CODE_FILES = [
    "src/__init__.py",
    "src/muzero/__init__.py",
    "src/muzero/networks.py",
    "src/muzero/transforms.py",
    "src/env/preprocess.py",
]


def git_sha() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(REPO), "status", "--porcelain"], text=True
            ).strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", False


def read_best_rate(run: str) -> float | None:
    p = RUNS / run / "checkpoints" / "best.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["completion_rate_100ep"])
    except Exception:
        return None


def select_checkpoints(levels: list[str], explicit: bool = True,
                       allow_latest: bool = False, pins: dict | None = None) -> list[dict]:
    """Pick the winning run per level: highest recorded self-play rate.

    A level whose runs never completed has no `best.pt` (it is only written on a
    new completion-rate high). With `allow_latest` such a level falls back to
    `latest.pt` — the final checkpoint of a run that never finished the level —
    and the entry is marked so the manifest never passes it off as a best.
    `pins` forces a specific run for a level, which is how a fallback picks
    between two equally-uncompleted runs on measured evidence rather than
    alphabetical order.
    """
    bench = {}
    if BENCHMARK.exists():
        for row in json.loads(BENCHMARK.read_text()).get("levels", []):
            bench[row["level"]] = row
    pins = pins or {}

    chosen = []
    for level in levels:
        tag = level.replace("Level", "level")
        runs = [pins[level]] if level in pins else [f"spec-{tag}", f"spec-imit-{tag}"]

        candidates = [(read_best_rate(r), r, RUNS / r / "checkpoints" / "best.pt", "best.pt")
                      for r in runs if (RUNS / r / "checkpoints" / "best.pt").exists()]
        fallback = False
        if not candidates and allow_latest:
            candidates = [(None, r, RUNS / r / "checkpoints" / "latest.pt", "latest.pt")
                          for r in runs if (RUNS / r / "checkpoints" / "latest.pt").exists()]
            fallback = bool(candidates)

        if not candidates:
            if explicit:
                print(f"  !! {level}: no checkpoint in spec-{tag} or spec-imit-{tag}, skipping")
            continue

        rate, run, ckpt, kind = max(candidates, key=lambda c: (c[0] if c[0] is not None else -1.0))
        entry = {"level": level, "run": run, "source_checkpoint": str(ckpt.relative_to(REPO)),
                 "selfplay_completion_rate": rate, "checkpoint_kind": kind}
        if fallback:
            entry["completed_level"] = False
            entry["note"] = ("this run never completed the level, so no best.pt exists; "
                             "this is the final checkpoint of the run")
        b = bench.get(level)
        if b and b.get("run") == run:
            entry["greedy_runthrough"] = {
                "n_completed": b.get("n_completed"),
                "n_rollouts": b.get("n_rollouts"),
                "steps_median": b.get("steps_median"),
                "noop_max": b.get("noop_max"),
                "skip_to_control": b.get("skip_to_control"),
            }
        chosen.append(entry)
    return chosen


def strip_checkpoint(src: Path, dst: Path) -> dict:
    """Write weights + config + provenance, dropping optimizer/scheduler/rng."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out = {
        "online": ck["online"],
        "training_step": ck.get("training_step"),
        "env_step": ck.get("env_step"),
        "cfg_snapshot": ck.get("cfg_snapshot"),
    }
    torch.save(out, dst)
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {
        "training_step": out["training_step"],
        "env_step": out["env_step"],
        "bytes": dst.stat().st_size,
        "sha256": h.hexdigest(),
    }


def zip_dir(stage: Path, archive: Path, level: int = 6) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=level,
                         allowZip64=True) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(stage.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help=f"where the bundle + zip are written (default {DEFAULT_OUT})")
    ap.add_argument("--levels", nargs="*", default=None,
                    help="default: every level with both a state file and a trained run")
    ap.add_argument("--no-zip", action="store_true", help="leave the staging dir unzipped")
    ap.add_argument("--human-data", action="store_true",
                    help="also archive outputs/human_trajectories (~1.9G, stored not deflated)")
    ap.add_argument("--name", default=None, help="bundle name (default muzero_mario_models_<date>)")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="also ship levels whose runs never completed, using latest.pt")
    ap.add_argument("--pin", action="append", default=[], metavar="LEVEL=RUN",
                    help="force a run for a level, e.g. --pin Level5-3=spec-level5-3")
    args = ap.parse_args()

    sha, dirty = git_sha()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    name = args.name or f"muzero_mario_models_{stamp}"
    stage = args.out_dir / name
    if stage.exists():
        print(f"Removing existing staging dir {stage}")
        shutil.rmtree(stage)
    (stage / "checkpoints").mkdir(parents=True)
    print(f"Staging -> {stage}")

    print("Selecting checkpoints...")
    pins = dict(p.split("=", 1) for p in args.pin)
    chosen = select_checkpoints(args.levels or all_levels(),
                                explicit=args.levels is not None,
                                allow_latest=args.include_incomplete, pins=pins)
    if not chosen:
        print("No checkpoints found; nothing to package.", file=sys.stderr)
        return 1

    arch = None
    for entry in chosen:
        src = REPO / entry["source_checkpoint"]
        dst = stage / "checkpoints" / f"{entry['level']}.pt"
        info = strip_checkpoint(src, dst)
        entry.update(info)
        entry["file"] = f"checkpoints/{entry['level']}.pt"
        ck_arch = torch.load(dst, map_location="cpu", weights_only=False)["cfg_snapshot"]["model"]
        if arch is None:
            arch = ck_arch
        elif ck_arch != arch:
            entry["architecture_differs"] = ck_arch
            print(f"  !! {entry['level']}: architecture differs from the others")
        print(f"  {entry['level']:<10} {entry['run']:<20} step {entry['training_step']:>8}  "
              f"{info['bytes']/1e6:6.1f} MB")

    print("Copying code...")
    for rel in CODE_FILES:
        dst = stage / "code" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    # Empty stand-in: the real src/env/__init__.py imports the emulation stack.
    (stage / "code" / "src" / "env" / "__init__.py").write_text("")

    manifest = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_sha": sha,
        "git_dirty": dirty,
        "checkpoint_kind": "per level: best.pt (highest rolling self-play completion "
                           "rate) unless the level entry says latest.pt",
        "n_levels": len(chosen),
        "architecture": arch,
        "obs_spec": {
            "shape": [4, 96, 96],
            "dtype_on_disk": "uint8",
            "model_input": "obs.float() / 255.0",
            "pipeline": "RGB -> grayscale -> resize 84x84 -> /255 -> max over consecutive "
                        "frame pairs -> stack 4 at frame_skip 4 -> edge-pad to 96x96",
            "n_frame_stack": 4,
            "frame_skip": 4,
        },
        "levels": chosen,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    shutil.copy2(REPO / "scripts" / "_bundle_load_model.py", stage / "load_model.py")
    shutil.copy2(REPO / "scripts" / "_bundle_README.md", stage / "README.md")

    total = sum(e["bytes"] for e in chosen)
    print(f"Staged {len(chosen)} checkpoints, {total/1e6:.0f} MB raw")

    if not args.no_zip:
        archive = args.out_dir / f"{name}.zip"
        print(f"Zipping -> {archive} (deflate, ~45% of raw; takes a few minutes)")
        zip_dir(stage, archive)
        print(f"  {archive}  {archive.stat().st_size/1e6:.0f} MB")

    if args.human_data:
        archive = args.out_dir / f"muzero_mario_human_trajectories_{stamp}.zip"
        print(f"Archiving human corpus -> {archive}")
        print("  (npz members are already deflated; storing without re-compression)")
        files = sorted(HUMAN_DIR.glob("*.npz")) + [
            p for p in (HUMAN_DIR / "conversion_report.json",
                        HUMAN_DIR / "human_level_stats.json") if p.exists()
        ]
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for i, p in enumerate(files):
                z.write(p, f"human_trajectories/{p.name}")
                if i % 1000 == 0:
                    print(f"    {i}/{len(files)}")
        print(f"  {archive}  {archive.stat().st_size/1e9:.2f} GB  ({len(files)} files)")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
