"""Frozen, paired action branches for early deaths and post-stall failures.

Compute nodes only. Source confirmation outcomes select diagnostic cases;
these conditional branch experiments are not new whole-level performance rates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_controller_diagnostic import (atomic_json, atomic_npz, canonical_sha,
    file_sha, numpy_search_rng, observation_uint8, reset_with_provenance, x_position)
from scripts.diagnose_stall_pulses import replay_to_branch
from scripts.stall_sampling_controller import StallSamplingController, SETTINGS

ARMS = ["continue", "logged_prefix", "release", "right_jump", "right_run_jump"]


def source_data(root, level, condition, episode, digest, observations=True):
    folder = root / "confirmation" / level / condition
    path = folder / f"episode_{episode:04d}.json"
    record = json.loads(path.read_text())
    identity = json.loads((folder.parent / "identity.json").read_text())
    if (record["status"] != "ok" or record["condition"] != condition or
            record["episode_index"] != episode or identity["checkpoint_sha256"] != digest or
            record["identity_sha256"] != canonical_sha(identity)):
        raise ValueError(f"Invalid source identity: {path}")
    artifacts = {}
    for suffix in ("trace", "bank"):
        artifact = path.with_name(path.stem + f"_{suffix}.npz")
        if file_sha(artifact) != record["artifacts"][artifact.name]:
            raise ValueError(f"Source hash mismatch: {artifact}")
        with np.load(artifact, allow_pickle=False) as arrays:
            if str(arrays["checkpoint_sha256"].item()) != digest:
                raise ValueError("Source bank checkpoint mismatch")
            artifacts[suffix] = {key: arrays[key] for key in arrays.files
                                if observations or key != "obs"}
    trace = artifacts["trace"]
    trace["bank_steps"] = artifacts["bank"]["steps"]
    if observations:
        trace["obs"] = artifacts["bank"]["obs"]
    return record, trace


def pre_states(record, trace):
    return np.r_[record["initial_player_state"], trace["player_state"][:-1]]


def death_anchors(record, trace, offsets):
    death = np.flatnonzero((trace["player_state"] == 11) | trace["died"] |
                          trace.get("below_playfield", np.zeros_like(trace["died"])))
    if not record["died"] or not len(death):
        raise ValueError("Expected a death with a recorded onset")
    onset = int(death[0])  # post-action signal: branch strictly before it
    states = pre_states(record, trace)
    roots = []
    for offset in offsets:
        eligible = trace["bank_steps"][(trace["bank_steps"] <= onset-offset)]
        eligible = eligible[states[eligible] == 8]
        if not len(eligible):
            raise ValueError("No eligible pre-death bank; do not silently shift protocol")
        t = int(eligible[-1])
        if t+8 > len(trace["actions"]):
            raise ValueError("Insufficient logged prefix")
        roots.append({"step": t, "offset_requested": offset,
                      "actual_decisions_before_onset": onset-t,
                      "x": int(trace["x_before"][t])})
    if len({r["step"] for r in roots}) != len(roots):
        raise ValueError("Two probe points collapsed; review selection explicitly")
    return onset, roots


def replay_height(cfg, level, record, trace):
    """Read vertical RAM absent from old traces, verifying the logged replay.

    Mario's normal ground body y is 208 on vertical page 1. Crossing body y
    224 on that page (absolute 480), or farther down, marks a fall below the
    playable floor. This catches pits before the delayed death/life signal.
    """
    env = make_env(cfg, level, record)
    heights = []
    try:
        _, provenance = reset_with_provenance(env)
        for key in ("initial_observation_sha256", "noop_frames_requested", "initial_x", "initial_player_state", "initial_lives"):
            if provenance[key] != record[key]:
                raise ValueError(f"Height replay reset mismatch: {key}")
        for t, action in enumerate(trace["actions"]):
            _, reward, done, info = env.step(int(action))
            if (x_position(info) != int(trace["x_after"][t]) or bool(done) != bool(trace["done"][t]) or
                    int(info["player_state"]) != int(trace["player_state"][t]) or
                    int(info["lives"]) != int(trace["lives"][t]) or
                    not np.isclose(reward, trace["rewards"][t], atol=1e-7, rtol=0)):
                raise ValueError(f"Height replay mismatch at {t}")
            heights.append(256*int(info["player_y_screen"])+int(info["player_y_pos"]))
    finally:
        env.close()
    trace["absolute_y_after"] = np.asarray(heights)
    trace["below_playfield"] = trace["absolute_y_after"] >= 480


def match_success(roots, failed_record, successes):
    """Choose one successful episode by x proximity, then reset phase and index.

    Matching does not establish equality of velocity, hidden state, or enemies.
    Reuse is allowed and must be reported rather than inflating independent n.
    """
    choices = []
    for record, trace in successes:
        if not record["completed"]:
            raise ValueError("Successful reference must complete intact")
        steps = trace["bank_steps"]
        steps = steps[(pre_states(record, trace)[steps] == 8) & (steps+8 <= len(trace["actions"]))]
        if not len(steps):
            continue
        matched = [int(steps[np.argmin(abs(trace["x_before"][steps]-r["x"]))]) for r in roots]
        error = sum(abs(int(trace["x_before"][t])-r["x"]) for t,r in zip(matched, roots))
        phase = abs(record["noop_frames_requested"]-failed_record["noop_frames_requested"])
        choices.append(((error, phase, record["episode_index"]), record, trace, matched))
    if not choices:
        raise ValueError("No successful reference")
    _, record, trace, steps = min(choices, key=lambda item: item[0])
    return record, [{"step": t, "x": int(trace["x_before"][t]),
                     "matched_failure_step": r["step"], "x_error": int(trace["x_before"][t])-r["x"]}
                    for t,r in zip(steps, roots)]


def prepare(root, manifest):
    old = Path(manifest["source_experiment"])
    if file_sha(old / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    original = json.loads((old / "manifest.json").read_text())
    if manifest["stall_controller"] != SETTINGS or manifest["arms"] != ARMS:
        raise ValueError("Controller/prefix protocol changed")
    if not json.loads((old / "audit.json").read_text())["complete"]:
        raise ValueError("Source audit incomplete")
    cases, lost = [], []
    for level, settings in manifest["levels"].items():
        from src.checkpoint import load_checkpoint
        cfg = load_checkpoint(settings["checkpoint"], map_location="cpu")["cfg_snapshot"]
        digest = settings["checkpoint_sha256"]
        if digest != original["levels"][level]["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed from source confirmation")
        successes, failures = [], []
        for trial in original["trials"]:
            ep = trial["episode_index"]
            greedy, gt = source_data(old, level, "greedy", ep, digest, False)
            gated, st = source_data(old, level, "stall_sampled", ep, digest, False)
            if any(greedy[k] != trial[k] or gated[k] != trial[k] for k in trial):
                raise ValueError("Source seed differs from reserved trial")
            if greedy["completed"]:
                successes.append((greedy, gt))
            if gated["completed"]:
                continue
            if not gated["died"]:
                raise ValueError("Unexpected non-death failure; review protocol")
            if gated["rescue_triggers"] == 0:
                group, condition, record, trace = "early_death", "greedy", greedy, gt
                if not greedy["died"]:
                    raise ValueError("Untriggered failure differs from greedy")
            else:
                group = "lost_greedy_success" if greedy["completed"] else "post_rescue_death"
                condition, record, trace = "stall_sampled", gated, st
            replay_height(cfg, level, record, trace)
            onset, roots = death_anchors(record, trace, manifest["offsets_before_death_onset"])
            failures.append({"level": level, "episode_index": ep, "source_condition": condition,
                             "continuation": condition, "group": group, "death_onset": onset,
                             "onset_signal": "below_playfield" if trace["below_playfield"][onset] else "death_state_or_life_loss",
                             "absolute_y_at_onset": int(trace["absolute_y_after"][onset]),
                             "original_completed": False, "roots": roots, "arms": ARMS})
            if group == "lost_greedy_success":
                first = int(gated["rescue_trigger_steps"][0])
                steps = gt["bank_steps"]
                eligible = steps[(steps <= first) & (pre_states(greedy, gt)[steps] == 8)]
                t = int(eligible[-1])
                if t > first:
                    raise ValueError("Trigger comparison starts too late")
                lost.append({"level": level, "episode_index": ep, "source_condition": "greedy",
                             "continuation": "greedy", "group": "lost_success_trigger_comparison",
                             "original_completed": True, "original_first_trigger": first,
                             "roots": [{"step": t, "x": int(gt["x_before"][t])}],
                             "arms": ["continue_greedy", "continue_stall"]})
        if len(failures) != manifest["expected_failures"][level]:
            raise ValueError(f"Unexpected failure count for {level}: {len(failures)}")
        for failure in failures:
            failure["case_id"] = len(cases)
            cases.append(failure)
            record, _ = source_data(old, level, failure["source_condition"], failure["episode_index"], digest, False)
            success, roots = match_success(failure["roots"], record, successes)
            cases.append({"case_id": len(cases), "level": level, "episode_index": success["episode_index"],
                          "source_condition": "greedy", "continuation": "greedy",
                          "group": "successful_reference", "matched_failure_case": failure["case_id"],
                          "original_completed": True, "roots": roots, "arms": ARMS})
    for case in lost:
        case["case_id"] = len(cases)
        cases.append(case)
    if len(lost) != 1:
        raise ValueError("Expected exactly one lost-success trigger comparison")
    plan = {"manifest_sha256": canonical_sha(manifest), "cases": cases,
            "n_source_failures": sum(manifest["expected_failures"].values()),
            "planned_branches": sum(len(c["roots"])*len(c["arms"])*len(manifest["branch_seeds"]) for c in cases)}
    path = root / "branch_plan.json"
    if path.exists():
        raise FileExistsError("Branch plan already frozen")
    atomic_json(path, plan)
    print(f"Frozen {len(cases)} cases, {plan['planned_branches']} branches", flush=True)


def load_model(manifest, level):
    import torch
    from src.checkpoint import load_checkpoint
    from src.muzero.networks import MuZeroNet
    settings = manifest["levels"][level]
    if file_sha(settings["checkpoint"]) != settings["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    checkpoint = load_checkpoint(settings["checkpoint"], map_location="cpu")
    cfg = checkpoint["cfg_snapshot"]
    keys = ("input_channels", "input_spatial", "hidden_channels", "hidden_spatial", "num_actions",
            "value_support", "reward_support", "rep_blocks", "dyn_blocks", "pred_blocks")
    net = MuZeroNet(**{key: cfg["model"][key] for key in keys})
    net.load_state_dict(checkpoint["online"], strict=True)
    return net.to("cuda").eval().requires_grad_(False), cfg


def make_env(cfg, level, record):
    from src.env.env import create_train_env
    ec = cfg["env"]
    return create_train_env(level=level, int_path=ec["int_path"], n_frame=ec["n_frame_stack"],
        downsample=ec["frame_skip"], pad_to=cfg["model"]["input_spatial"] if ec["pad_to_input_spatial"] else None,
        seed=record["env_seed"], noop_max=ec.get("noop_max", 0), skip_to_control=ec.get("skip_to_control", False),
        done_on_life_loss=ec.get("done_on_life_loss", True), completion_bonus=ec.get("completion_bonus", 100.))


def make_search(cfg, manifest):
    from src.muzero.mcts import MCTS
    return MCTS(discount=cfg["muzero"]["discount"], num_simulations=manifest["num_simulations"],
                root_dirichlet_alpha=.25, root_exploration_eps=0., pb_c_base=cfg["mcts"]["pb_c_base"],
                pb_c_init=cfg["mcts"]["pb_c_init"], device="cuda", leaf_batch=manifest["leaf_batch"])


def monitor_at(record, trace, step, enabled):
    if not enabled:
        return None
    monitor = StallSamplingController(**SETTINGS)
    states = pre_states(record, trace)
    for i in range(step):
        monitor.observe(int(trace["x_before"][i]), int(states[i]))
    return monitor


@contextmanager
def capture_root():
    """Observe the existing search tree without changing selection or RNG use."""
    import src.muzero.mcts as module
    original = module.Node
    roots = []
    def construct(*args, **kwargs):
        node = original(*args, **kwargs)
        roots.append(node)
        return node
    with patch.object(module, "Node", construct):
        yield roots


def observed_search(search, obs, net, temperature):
    with capture_root() as roots:
        result = search.run(obs, net, temperature=temperature, deterministic=False)
    if len(roots) != 1:
        raise ValueError("Unexpected root construction count")
    root = roots[0]
    diagnostic = {"visits": root.child_N.tolist(), "root_q": float(root.Q),
                  "child_return_estimates": [float(c.rwd+search.discount*c.Q) if c.N else None for c in root.children],
                  "child_reward_estimates": [float(c.rwd) if c.N else None for c in root.children]}
    return result, diagnostic


def prefix_actions(arm, logged):
    if arm in ("continue", "continue_greedy", "continue_stall"):
        return []
    if arm == "logged_prefix":
        if len(logged) != 8:
            raise ValueError("Logged prefix must contain eight actions")
        return [int(a) for a in logged]
    return [{"release": 0, "right_jump": 2, "right_run_jump": 4}[arm]]*8


def predict_prefix(net, obs, actions, discount):
    import torch
    tensor = torch.from_numpy(obs).to("cuda", dtype=torch.float32).unsqueeze(0)
    hidden, logits, value = net.initial_inference(tensor)
    result = {"prior": torch.softmax(logits, -1)[0].cpu().tolist(), "root_value": float(value.item())}
    rewards = []
    for action in actions:
        hidden, reward, _, value = net.recurrent_inference(hidden, torch.tensor([action], device="cuda"))
        rewards.append(float(reward.item()))
    result.update(prefix_reward_predictions=rewards, imagined_endpoint_value=float(value.item()),
                  imagined_prefix_return=sum(discount**i*r for i,r in enumerate(rewards))+discount**len(actions)*float(value.item()))
    return result


def full_source_replay(cfg, case, record, trace, folder):
    """Verify entire logged episode, including saved observations and death onset."""
    from PIL import Image
    env = make_env(cfg, case["level"], record)
    frames, frame_steps, positions = [], [], []
    windows = [r["step"] for r in case["roots"]]
    banks = {int(t): i for i,t in enumerate(trace["bank_steps"])}
    monitor = StallSamplingController(**SETTINGS) if case["source_condition"] == "stall_sampled" else None
    try:
        obs, provenance = reset_with_provenance(env)
        for key in ("initial_observation_sha256", "noop_frames_requested", "initial_x", "initial_player_state", "initial_lives"):
            if provenance[key] != record[key]:
                raise ValueError(f"Source reset mismatch: {key}")
        for t, action in enumerate(trace["actions"]):
            positions.append({"step": t, **{key: int(env.last_info[key]) for key in
                              ("player_y_screen", "player_y_pos", "player_y_speed", "player_x_speed", "player_state")}})
            if monitor is not None:
                metadata = monitor.observe(x_position(env.last_info), int(env.last_info["player_state"]))
                if any(metadata[key] != trace[key][t] for key in metadata):
                    raise ValueError(f"Source monitor reconstruction mismatch at {t}")
            if t in banks and not np.array_equal(observation_uint8(obs), trace["obs"][banks[t]]):
                raise ValueError(f"Source observation mismatch at {t}")
            if x_position(env.last_info) != int(trace["x_before"][t]):
                raise ValueError(f"Source pre-action x mismatch at {t}")
            if any(start-8 <= t <= start+48 for start in windows):
                frames.append(env.latest_raw_rgb().copy())
                frame_steps.append(t)
            obs, reward, done, info = env.step(int(action))
            if (x_position(info) != int(trace["x_after"][t]) or bool(done) != bool(trace["done"][t]) or
                    int(info["player_state"]) != int(trace["player_state"][t]) or
                    int(info["lives"]) != int(trace["lives"][t]) or
                    not np.isclose(reward, trace["rewards"][t], atol=1e-7, rtol=0)):
                raise ValueError(f"Source replay mismatch at {t}")
        if bool(info.get("level_complete", False) and done) != record["completed"]:
            raise ValueError("Source final completion mismatch")
    finally:
        env.close()
    artifact = folder / "source_frames.npz"
    atomic_npz(artifact, {"rgb": np.stack(frames), "steps": np.asarray(frame_steps)})
    # Compact contact sheet retains annotated times; raw RGB sequence is also saved.
    from PIL import ImageDraw
    chosen = np.linspace(0, len(frames)-1, min(12, len(frames))).astype(int)
    width, height = frames[0].shape[1], frames[0].shape[0]
    sheet = Image.new("RGB", (4*width, ((len(chosen)+3)//4)*(height+22)), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(chosen):
        x, y = slot%4*width, slot//4*(height+22)
        sheet.paste(Image.fromarray(frames[index]), (x,y+22))
        draw.text((x+4,y+4), f"decision {frame_steps[index]}", fill="black")
    sheet.save(folder / "source_contact_sheet.png")
    atomic_json(folder / "source_replay.json", {"passed": True, "decisions": len(trace["actions"]),
                "source_record": record, "frames_sha256": file_sha(artifact), "positions": positions})


def summarize_rows(rows, anchor_x, discount, local_horizon, allowance):
    def outcome(part):
        return {"decisions": len(part), "completed": any(r["completed"] for r in part),
                "died": any(r["died"] for r in part), "final_x": part[-1]["x_after"],
                "delta_x": part[-1]["x_after"]-anchor_x,
                "max_delta_x": max(r["x_after"] for r in part)-anchor_x,
                "discounted_return": sum(discount**i*r["reward"] for i,r in enumerate(part))}
    result = outcome(rows)
    result["timed_out"] = bool(not rows[-1]["done"] and len(rows) == allowance)
    result["local"] = outcome(rows[:local_horizon])
    return result


def run_case(root, manifest, plan, case_id, smoke=False):
    import torch
    torch.set_num_threads(1)
    case = plan["cases"][case_id]
    net, cfg = load_model(manifest, case["level"])
    folder = root / ("smoke" if smoke else "full") / f"case_{case_id:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": canonical_sha(manifest), "plan_sha256": canonical_sha(plan),
                "case": case, "smoke": smoke, "script_sha256": file_sha(__file__),
                "source_sha256": manifest["source_sha256"], "torch_version": torch.__version__,
                "numpy_version": np.__version__}
    identity_sha = canonical_sha(identity)
    path = folder / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Resume identity mismatch")
    atomic_json(path, identity)
    old = Path(manifest["source_experiment"])
    record, trace = source_data(old, case["level"], case["source_condition"], case["episode_index"],
                                manifest["levels"][case["level"]]["checkpoint_sha256"])
    full_source_replay(cfg, case, record, trace, folder)
    seeds = manifest["branch_seeds"][:1] if smoke else manifest["branch_seeds"]
    roots = case["roots"][:1] if smoke else case["roots"]
    report = {"identity": identity, "complete": False, "planned_branches": len(roots)*len(seeds)*len(case["arms"]), "branches": []}
    for root_index, root_spec in enumerate(roots):
        t = root_spec["step"]
        for seed in seeds:
            for arm in case["arms"]:
                path = folder / f"root_{root_index}_seed_{seed}_{arm}.json"
                if path.exists():
                    branch = json.loads(path.read_text())
                    if branch["identity_sha256"] != identity_sha or file_sha(path.with_suffix(".npz")) != branch["trace_sha256"]:
                        raise ValueError("Saved branch changed")
                    report["branches"].append(branch)
                    continue
                env = make_env(cfg, case["level"], record)
                rows = []
                enabled = arm == "continue_stall" or (case["continuation"] == "stall_sampled" and arm != "continue_greedy")
                monitor = monitor_at(record, trace, t, enabled)
                forced = prefix_actions(arm, trace["actions"][t:t+8])
                search = make_search(cfg, manifest)
                allowance = min(16, manifest["max_steps"]-t) if smoke else manifest["max_steps"]-t
                try:
                    obs = replay_to_branch(env, record, trace, t)
                    with torch.inference_mode(), numpy_search_rng(seed):
                        prediction = predict_prefix(net, obs, forced, search.discount)
                        endpoint_value = None
                        lives = int(env.last_info["lives"])
                        for step in range(allowance):
                            metadata = monitor.observe(x_position(env.last_info), int(env.last_info["player_state"])) if monitor else {"controller_temperature": 0., "rescue_trigger": False}
                            temperature = metadata["controller_temperature"]
                            if step == 0:
                                # Verify instrumentation leaves RNG and output exactly intact.
                                state = np.random.get_state()
                                plain = search.run(obs, net, temperature=temperature, deterministic=False)
                                after = np.random.get_state()
                                np.random.set_state(state)
                                (action, visits, q), diagnostic = observed_search(search, obs, net, temperature)
                                observed_after = np.random.get_state()
                                if (plain[0] != action or plain[2] != q or not np.array_equal(plain[1], visits) or
                                        any(not np.array_equal(a,b) for a,b in zip(after, observed_after))):
                                    raise ValueError("Search instrumentation changed behavior")
                                diagnostic.update(selected_action=int(action), visit_distribution=visits.tolist(),
                                                  original_action=int(trace["actions"][t]),
                                                  original_prior=trace["priors"][t].tolist(),
                                                  original_visits=trace["policies"][t].tolist())
                            else:
                                action, visits, q = search.run(obs, net, temperature=temperature, deterministic=False)
                            chosen = int(forced[step]) if step < len(forced) else int(action)
                            obs, reward, done, info = env.step(chosen)
                            new_lives = int(info["lives"])
                            rows.append({"action": chosen, "controller_action": int(action),
                                         "temperature": temperature, "rescue_trigger": metadata["rescue_trigger"],
                                         "forced": step < len(forced), "x_after": x_position(info),
                                         "player_state": int(info["player_state"]), "reward": float(reward),
                                         "done": bool(done), "died": new_lives < lives,
                                         "completed": bool(done and info.get("level_complete", False)), "root_q": float(q)})
                            lives = new_lives
                            if step+1 == len(forced):
                                if done:
                                    endpoint_value = 0.
                                else:
                                    _, _, value = net.initial_inference(torch.from_numpy(obs).to("cuda", dtype=torch.float32).unsqueeze(0))
                                    endpoint_value = float(value.item())
                            if done:
                                break
                finally:
                    env.close()
                branch = {"identity_sha256": identity_sha, "case_id": case_id, "root_index": root_index,
                          "step": t, "seed": seed, "arm": arm, "forced_actions": forced,
                          "replay_verified": True, "prediction": prediction, "root_search": diagnostic,
                          **summarize_rows(rows, int(trace["x_before"][t]), search.discount, manifest["local_horizon"], allowance)}
                prefix_rows = rows[:len(forced)]
                branch["observed_prefix_discounted_reward"] = sum(search.discount**i*r["reward"] for i,r in enumerate(prefix_rows))
                branch["prefix_executed_decisions"] = len(prefix_rows)
                branch["observed_endpoint_network_value"] = endpoint_value
                branch["note"] = "Full return is realized under the continuation controller; learned values and eight-action predictions are not completion probabilities."
                atomic_npz(path.with_suffix(".npz"), {key: np.asarray([r[key] for r in rows]) for key in rows[0]})
                branch["trace_sha256"] = file_sha(path.with_suffix(".npz"))
                atomic_json(path, branch)
                report["branches"].append(branch)
                atomic_json(folder / "summary.json", report)
                print(f"case={case_id} root={root_index} seed={seed} {arm}: completed={branch['completed']} died={branch['died']} dx={branch['delta_x']}", flush=True)
    report["complete"] = len(report["branches"]) == report["planned_branches"]
    atomic_json(folder / "summary.json", report)


def aggregate(root, manifest, plan):
    report = {"complete": False, "manifest_sha256": canonical_sha(manifest), "cases": [],
              "planned_branches": plan["planned_branches"], "finished_branches": 0,
              "interpretation": "Conditional diagnostic branches; seeds and roots within episodes are not independent level attempts."}
    for case in plan["cases"]:
        folder = root / "full" / f"case_{case['case_id']:03d}"
        summary = json.loads((folder / "summary.json").read_text())
        if not summary["complete"] or summary["identity"]["plan_sha256"] != canonical_sha(plan):
            raise ValueError("Incomplete or mismatched case")
        for b in summary["branches"]:
            path = folder / f"root_{b['root_index']}_seed_{b['seed']}_{b['arm']}.npz"
            if file_sha(path) != b["trace_sha256"]:
                raise ValueError("Branch trace changed")
        per_root = []
        for i, spec in enumerate(case["roots"]):
            arms = {}
            for arm in case["arms"]:
                rows = [b for b in summary["branches"] if b["root_index"] == i and b["arm"] == arm]
                if sorted(b["seed"] for b in rows) != manifest["branch_seeds"]:
                    raise ValueError("Missing or duplicated branch seed")
                arms[arm] = {"n_branch_seeds": len(rows), "completions": sum(b["completed"] for b in rows),
                             "deaths": sum(b["died"] for b in rows), "timeouts": sum(b["timed_out"] for b in rows),
                             "local_deaths": sum(b["local"]["died"] for b in rows),
                             "local_dx_median": float(np.median([b["local"]["delta_x"] for b in rows]))}
            per_root.append({**spec, "arms": arms})
        report["cases"].append({**case, "results": per_root})
        report["finished_branches"] += len(summary["branches"])
    report["unique_successful_reference_episodes"] = len({(c["level"],c["episode_index"]) for c in plan["cases"] if c["group"] == "successful_reference"})
    report["complete"] = report["finished_branches"] == report["planned_branches"]
    atomic_json(root / "summary.json", report)
    print(f"Audited {report['finished_branches']} branches", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["prepare", "run", "aggregate"], required=True)
    parser.add_argument("--case", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Compute nodes only")
    manifest = json.loads((args.out / "manifest.json").read_text())
    if args.mode == "prepare":
        prepare(args.out, manifest)
        return
    plan = json.loads((args.out / "branch_plan.json").read_text())
    if plan["manifest_sha256"] != canonical_sha(manifest):
        raise ValueError("Plan manifest mismatch")
    if args.mode == "aggregate":
        aggregate(args.out, manifest, plan)
    else:
        if args.case is None or not 0 <= args.case < len(plan["cases"]):
            parser.error("Valid --case required")
        # One process per case prevents concurrent writes or accidental retries.
        import fcntl
        lockpath = args.out / f".case_{args.case}_{args.smoke}.lock"
        with lockpath.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_case(args.out, manifest, plan, args.case, args.smoke)


if __name__ == "__main__":
    main()
