"""Snapshot the current-best MuZero run's per-level metrics from its wandb log.

Reads the (offline) wandb run datastore for the live `autocurriculum-longdecay`
run and writes a small JSON with per-level completion / return / length, so the
comparison plot doesn't depend on the moving wandb file.

Run from the repo root with the project venv active:
    LD_LIBRARY_PATH=<libffi dir>:$LD_LIBRARY_PATH python analysis/comparison/extract_model_metrics.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

REPO = Path(__file__).resolve().parents[2]
RUN_GLOB = str(REPO / "outputs/runs/autocurriculum-longdecay/wandb/run-*/run-*.wandb")
LEVELS = [f"Level{w}-{s}" for w in range(1, 5) for s in range(1, 4)]
OUT = REPO / "analysis/comparison/model_metrics.json"


def read_summary(wandb_file: str) -> dict:
    """Return the latest summary (last value of every logged metric)."""
    ds = DataStore()
    ds.open_for_scan(wandb_file)
    summary: dict = {}
    while True:
        try:
            rec = ds.scan_record()
        except Exception:
            break
        if rec is None:
            break
        pbrec = pb.Record()
        try:
            pbrec.ParseFromString(rec[1] if isinstance(rec, tuple) else rec)
        except Exception:
            continue
        if pbrec.HasField("summary"):
            for it in pbrec.summary.update:
                key = it.key if it.key else "/".join(it.nested_key)
                summary[key] = it.value_json
    return summary


def jval(summary: dict, key: str):
    v = summary.get(key)
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def main() -> None:
    wandb_file = sorted(glob.glob(RUN_GLOB))[-1]
    summary = read_summary(wandb_file)
    out = {
        "run": "autocurriculum-longdecay",
        "wandb_file": os.path.relpath(wandb_file, REPO),
        "env_step": jval(summary, "train/env_step"),
        "wandb_step": jval(summary, "_step"),
        "completion_rate_100ep": jval(summary, "selfplay/completion_rate_100ep"),
        "levels": {},
    }
    for lv in LEVELS:
        out["levels"][lv] = {
            "selfplay_completed": jval(summary, f"selfplay/completed/{lv}"),
            "completion_rate": jval(summary, f"autocurriculum/completion_rate/{lv}"),
            "replay_completed": jval(summary, f"replay/{lv}_completed"),
            "replay_return": jval(summary, f"replay/{lv}_return"),
            "replay_length": jval(summary, f"replay/{lv}_length"),
            "selfplay_episode_return": jval(summary, f"selfplay/episode_return/{lv}"),
            "selfplay_episode_length": jval(summary, f"selfplay/episode_length/{lv}"),
            "mean_length": jval(summary, f"autocurriculum/mean_length/{lv}"),
        }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}  (env_step={out['env_step']})")


if __name__ == "__main__":
    main()
