"""Replay every human .bk2 for the model's levels -> per-attempt outcomes CSV.

Walks a mario-dataset root for `gamelogs/*_level-w{W}l{L}_rep-*.bk2` files whose
level is in the target set, replays each through the stable-retro integration
(see replay_bk2.replay_bk2), and writes one row per attempt:

    subject, session, level, bk2, completed, game_over, n_frames, duration_s,
    max_x, final_lives

Replays are independent emulator runs, parallelised across processes.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Import the single-file replayer from the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_bk2 import replay_bk2  # noqa: E402

SUB_RE = re.compile(r"(sub-\d+)")
SES_RE = re.compile(r"(ses-\d+)")
LEVEL_RE = re.compile(r"level-w(\d+)l(\d+)")

DEFAULT_LEVELS = [f"Level{w}-{s}" for w in range(1, 5) for s in range(1, 4)]


def bk2_level(path: Path) -> str | None:
    m = LEVEL_RE.search(path.name)
    return f"Level{int(m.group(1))}-{int(m.group(2))}" if m else None


def find_bk2(mario_root: Path, levels: set[str], limit_per_level: int | None) -> list[Path]:
    found: dict[str, list[Path]] = {lv: [] for lv in levels}
    for p in sorted(mario_root.glob("sub-*/ses-*/gamelogs/*_rep-*.bk2")):
        lv = bk2_level(p)
        if lv in levels:
            found[lv].append(p)
    out: list[Path] = []
    for lv, paths in found.items():
        out.extend(paths[:limit_per_level] if limit_per_level else paths)
    return out


def _one(args: tuple[str, str]) -> dict | None:
    bk2, int_path = args
    p = Path(bk2)
    try:
        rec = replay_bk2(bk2, int_path)
    except Exception as e:  # noqa: BLE001 — one bad movie shouldn't kill the batch
        return {"bk2": p.name, "error": str(e)[:200],
                "subject": (SUB_RE.search(str(p)) or [None, None])[1] if SUB_RE.search(str(p)) else None}
    sub = SUB_RE.search(str(p))
    ses = SES_RE.search(str(p))
    rec.update(bk2=p.name,
               subject=sub.group(1) if sub else None,
               session=ses.group(1) if ses else None)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mario-root", required=True)
    ap.add_argument("--int-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit-per-level", type=int, default=None,
                    help="cap attempts per level (for quick tests)")
    ap.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    args = ap.parse_args()

    levels = set(args.levels.split(","))
    mario_root = Path(args.mario_root)
    int_path = str(Path(args.int_path).resolve())

    bk2s = find_bk2(mario_root, levels, args.limit_per_level)
    print(f"Found {len(bk2s)} .bk2 across {len(levels)} levels; replaying with {args.jobs} workers")

    fields = ["subject", "session", "level", "completed", "game_over",
              "n_frames", "duration_s", "max_x", "score", "final_lives", "bk2", "error"]
    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_one, (str(b), int_path)) for b in bk2s]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec:
                rows.append(rec)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(bk2s)} replayed", flush=True)

    errors = sum(1 for r in rows if r.get("error"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r.get("level") or "", r.get("bk2") or "")))
    print(f"wrote {out}  ({len(rows)} rows, {errors} errors)")


if __name__ == "__main__":
    main()
