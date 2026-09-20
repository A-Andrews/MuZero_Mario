"""Paired short action interventions from exactly replayed development states.

Tests local stall escape, not full-level performance. Replays emulator inputs
from reset for each branch, avoiding incomplete emulator/wrapper snapshots.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_controller_diagnostic import (atomic_json, canonical_sha, file_sha,
    numpy_search_rng, observation_uint8, reset_with_provenance, x_position)

ARMS = {"logged_prefix": None, "release": [0]*8, "right_jump": [2]*8,
        "right_run_jump": [4]*8, "left_then_right_run_jump": [6]*4+[4]*4}


def stall_step(x_before, bank_steps, *, window=96, tolerance=16, horizon=128):
    """First saved root following sustained low x variation away from spawn."""
    x = np.asarray(x_before)
    for t in bank_steps:
        t = int(t)
        if t < window or t+horizon > len(x):
            continue
        history = x[t-window:t+1]
        if x[t] > 256 and np.ptp(history) <= tolerance:
            return t
    return None


def control_step(bank_steps, trace_length, fraction):
    candidates = np.asarray(bank_steps)
    candidates = candidates[candidates+8 < trace_length]
    if not len(candidates):
        raise ValueError("No control root with an eight-decision prefix")
    return int(candidates[np.argmin(abs(candidates-fraction*trace_length))])


def read_source(folder, episode, digest, observations=False):
    stem = f"episode_{episode:04d}"
    result_path = folder / f"{stem}.json"
    result = json.loads(result_path.read_text())
    if result["status"] != "ok" or result["episode_index"] != episode:
        raise ValueError("Invalid source episode")
    paths = {suffix: folder / f"{stem}_{suffix}.npz" for suffix in ("bank", "trace")}
    for path in paths.values():
        if file_sha(path) != result["artifacts"][path.name]:
            raise ValueError(f"Source artifact changed: {path}")
    with np.load(paths["bank"], allow_pickle=False) as bank, np.load(paths["trace"], allow_pickle=False) as trace:
        if any(str(data["checkpoint_sha256"].item()) != digest for data in (bank, trace)):
            raise ValueError("Source checkpoint mismatch")
        data = {key: trace[key] for key in ("actions", "x_before", "x_after", "rewards", "done", "completed", "died")}
        data["bank_steps"] = bank["steps"]
        if observations:
            data["obs"] = bank["obs"]
            data["priors"] = bank["priors"]
            data["policies"] = bank["policies"]
    return result, data


def prepare_cases(manifest, old, out):
    original = json.loads((old / "manifest.json").read_text())
    plan = {"manifest_sha256": canonical_sha(manifest), "development_manifest_sha256": file_sha(old / "manifest.json"),
            "cases": [], "excluded": [], "source_timeout_count": 0, "arms": ARMS,
            "horizon": 128, "pulse_decisions": 8, "window": 96, "x_tolerance": 16, "escape_pixels": 64}
    for trial in original["trials"]:
        episode = trial["episode_index"]
        result, trace = read_source(old / "development/Level1-1/greedy", episode,
                                    manifest["levels"]["Level1-1"]["checkpoint_sha256"])
        if not result["timed_out"]:
            continue
        plan["source_timeout_count"] += 1
        t = stall_step(trace["x_before"], trace["bank_steps"])
        if t is None:
            plan["excluded"].append({"episode_index": episode, "reason": "No eligible 96-decision low-progress window with 128 remaining decisions"})
            continue
        ctrl, ctrl_trace = read_source(old / "development/Level6-1/greedy", episode,
                                       manifest["levels"]["Level6-1"]["checkpoint_sha256"])
        ct = control_step(ctrl_trace["bank_steps"], len(ctrl_trace["actions"]), t/len(trace["actions"]))
        plan["cases"].append({"episode_index": episode, "continuation_search_seed": 9100001+episode,
                              "levels": {"Level1-1": {"step": t, "original_completed": result["completed"]},
                                         "Level6-1": {"step": ct, "original_completed": ctrl["completed"]}}})
    if not plan["cases"]:
        raise ValueError("No eligible cases; do not silently change selection rules")
    atomic_json(out, plan)
    print(f"Frozen {len(plan['cases'])} paired cases from {plan['source_timeout_count']} timeouts; exclusions={len(plan['excluded'])}", flush=True)


def replay_to_branch(env, original, trace, t):
    obs, provenance = reset_with_provenance(env)
    for key in ("initial_observation_sha256", "noop_frames_requested", "initial_x", "initial_player_state", "initial_lives"):
        if provenance[key] != original[key]:
            raise ValueError(f"Reset mismatch: {key}")
    for step, action in enumerate(trace["actions"][:t]):
        obs, reward, done, info = env.step(int(action))
        if done or x_position(info) != int(trace["x_after"][step]) or not np.isclose(reward, trace["rewards"][step], atol=1e-7, rtol=0):
            raise ValueError(f"Replay diverged at source decision {step}")
    index = np.flatnonzero(trace["bank_steps"] == t)
    if len(index) != 1 or not np.array_equal(observation_uint8(obs), trace["obs"][index[0]]):
        raise ValueError("Replayed branch observation differs from saved bank")
    if x_position(env.last_info) != int(trace["x_before"][t]):
        raise ValueError("Replayed branch position mismatch")
    return obs


def summarize_branch(rows, anchor_x):
    end = rows[-1]
    completed = any(r["completed"] for r in rows)
    died = any(r["died"] for r in rows)
    return {"decisions": len(rows), "completed": completed, "died": died,
            "final_x": end["x_after"], "delta_x": end["x_after"]-anchor_x,
            "max_delta_x": max(r["x_after"] for r in rows)-anchor_x,
            "forward_escape": bool(completed or (not died and end["x_after"] >= anchor_x+64)),
            "terminal_other": bool(end["done"] and not completed and not died),
            "local_return": sum(r["reward"] for r in rows)}


def run_level(args, manifest, plan):
    import torch
    from PIL import Image
    from src.checkpoint import load_checkpoint
    from src.env.env import create_train_env
    from src.muzero.mcts import MCTS
    from src.muzero.networks import MuZeroNet
    torch.set_num_threads(1)
    settings = manifest["levels"][args.level]
    checkpoint = Path(settings["checkpoint"])
    if file_sha(checkpoint) != settings["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    state = load_checkpoint(checkpoint, map_location="cpu")
    cfg = state["cfg_snapshot"]
    keys = ("input_channels", "input_spatial", "hidden_channels", "hidden_spatial", "num_actions",
            "value_support", "reward_support", "rep_blocks", "dyn_blocks", "pred_blocks")
    net = MuZeroNet(**{key: cfg["model"][key] for key in keys})
    net.load_state_dict(state["online"], strict=True)
    del state
    net.to("cuda").eval().requires_grad_(False)
    ec = cfg["env"]
    cases = plan["cases"][:1] if args.smoke else plan["cases"]
    horizon = 16 if args.smoke else plan["horizon"]
    target = args.out / ("smoke" if args.smoke else "full") / args.level
    target.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": canonical_sha(manifest), "plan_sha256": canonical_sha(plan),
                "script_sha256": file_sha(__file__), "checkpoint_sha256": settings["checkpoint_sha256"],
                "smoke": args.smoke, "horizon": horizon, "level": args.level}
    identity_path = target / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Resume identity mismatch")
    atomic_json(identity_path, identity)
    report = {"identity": identity, "planned_cases": len(cases), "complete": False, "cases": []}
    for case in cases:
        ep, t = case["episode_index"], case["levels"][args.level]["step"]
        original, trace = read_source(args.development / "development" / args.level / "greedy", ep,
                                      settings["checkpoint_sha256"], observations=True)
        anchor = int(trace["x_before"][t])
        index = int(np.flatnonzero(trace["bank_steps"] == t)[0])
        result = {"episode_index": ep, "source_step": t, "anchor_x": anchor,
                  "original_completed": original["completed"], "original_timed_out": original["timed_out"],
                  "continuation_search_seed": case["continuation_search_seed"],
                  "original_prior": trace["priors"][index].tolist(), "original_visits": trace["policies"][index].tolist(), "arms": {}}
        for name, pulse in ARMS.items():
            result_path = target / f"episode_{ep:04d}_{name}.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if saved["identity_sha256"] != canonical_sha(identity):
                    raise ValueError("Branch identity mismatch")
                result["arms"][name] = saved
                continue
            env = create_train_env(level=args.level, int_path=ec["int_path"], n_frame=ec["n_frame_stack"],
                downsample=ec["frame_skip"], pad_to=cfg["model"]["input_spatial"] if ec["pad_to_input_spatial"] else None,
                seed=original["env_seed"], noop_max=ec.get("noop_max", 0), skip_to_control=ec.get("skip_to_control", False),
                done_on_life_loss=ec.get("done_on_life_loss", True), completion_bonus=ec.get("completion_bonus", 100.))
            rows = []
            try:
                obs = replay_to_branch(env, original, trace, t)
                if name == "logged_prefix":
                    Image.fromarray(env.latest_raw_rgb()).save(target / f"episode_{ep:04d}_branch.png")
                actions = trace["actions"][t:t+8].tolist() if pulse is None else pulse
                search = MCTS(discount=cfg["muzero"]["discount"], num_simulations=50,
                    root_dirichlet_alpha=.25, root_exploration_eps=0., pb_c_base=cfg["mcts"]["pb_c_base"],
                    pb_c_init=cfg["mcts"]["pb_c_init"], device=torch.device("cuda"), leaf_batch=4)
                with numpy_search_rng(case["continuation_search_seed"]), torch.inference_mode():
                    lives = int(env.last_info["lives"])
                    for step in range(horizon):
                        if step < 8:
                            action = int(actions[step])
                        else:
                            action, _, _ = search.run(obs, net, temperature=0., deterministic=False)
                        obs, reward, done, info = env.step(action)
                        new_lives = int(info["lives"])
                        rows.append({"step": step, "action": int(action), "x_after": x_position(info),
                                     "reward": float(reward), "done": bool(done), "died": new_lives < lives,
                                     "completed": bool(done and info.get("level_complete", False))})
                        lives = new_lives
                        if done:
                            break
                Image.fromarray(env.latest_raw_rgb()).save(target / f"episode_{ep:04d}_{name}_end.png")
            finally:
                env.close()
            branch = {**summarize_branch(rows, anchor), "identity_sha256": canonical_sha(identity),
                      "pulse_actions": actions, "trace": rows, "replay_verified": True}
            atomic_json(result_path, branch)
            result["arms"][name] = branch
            print(f"{args.level} episode={ep} {name} escape={branch['forward_escape']} dx={branch['delta_x']} died={branch['died']}", flush=True)
        report["cases"].append(result)
        atomic_json(target / "summary.json", report)
    report["complete"] = True
    report["arm_totals"] = {name: {"n": len(cases), "escapes": sum(c["arms"][name]["forward_escape"] for c in report["cases"]),
                                    "deaths": sum(c["arms"][name]["died"] for c in report["cases"]),
                                    "completions": sum(c["arms"][name]["completed"] for c in report["cases"])} for name in ARMS}
    atomic_json(target / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--level", choices=["Level1-1", "Level6-1"])
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Compute nodes only")
    manifest = json.loads((args.out / "manifest.json").read_text())
    path = args.out / "pulse_plan.json"
    if args.prepare:
        if path.exists():
            raise FileExistsError("Plan already frozen")
        prepare_cases(manifest, args.development, path)
    else:
        if not args.level:
            parser.error("--level is required")
        plan = json.loads(path.read_text())
        if plan["manifest_sha256"] != canonical_sha(manifest) or plan["arms"] != ARMS:
            raise ValueError("Plan identity mismatch")
        run_level(args, manifest, plan)


if __name__ == "__main__":
    main()
