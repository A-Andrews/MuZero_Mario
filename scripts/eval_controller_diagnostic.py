"""Fixed-attempt development or reserved-seed confirmation of frozen controllers.

Run on a SLURM compute node. The manifest fixes checkpoint hashes, paired
environment/search seeds, and all controller settings. This is diagnostic
split is recorded throughout. Confirmation settings and seeds are frozen before
execution and cannot be used for smoke tests. No training occurs.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONDITIONS = {
    "greedy": (4, 0.0, 0.25, 0.0),
    "sequential_greedy": (1, 0.0, 0.25, 0.0),
    "root_noise": (4, 0.25, 0.25, 0.0),
    "sampled": (4, 0.0, 0.25, 0.25),
    "noise_and_sampling": (4, 0.25, 0.25, 0.25),
}
ALL_CONDITIONS = {**CONDITIONS, "stall_sampled": (4, 0.0, 0.25, 0.0)}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest):
    split = manifest.get("split")
    profile = manifest.get("evaluation_profile", "original")
    if manifest.get("schema_version") != 1 or split not in ("development", "confirmation"):
        raise ValueError("Expected schema_version=1 and development or confirmation split")
    if not manifest.get("experiment_id"):
        raise ValueError("experiment_id is required")
    count = 30 if split == "development" else 100
    if manifest.get("episodes") != count or len(manifest.get("trials", [])) != count:
        raise ValueError(f"{split} manifest must schedule exactly {count} trials")
    if split == "confirmation":
        if manifest["trials"] != manifest.get("confirmation_trials"):
            raise ValueError("Confirmation must use the unchanged reserved trials")
        development = manifest.get("development_trials", [])
        if len(development) != 30:
            raise ValueError("Confirmation requires development seed provenance")
        old = {t[k] for t in development for k in ("env_seed", "search_seed")}
        new = {t[k] for t in manifest["trials"] for k in ("env_seed", "search_seed")}
        if old & new:
            raise ValueError("Confirmation seeds overlap development seeds")
        if profile != "stall_gated" and set(manifest["levels"]) != {"Level1-1", "Level6-1"}:
            raise ValueError("Confirmation fixes Level1-1 and Level6-1 control")
    for key in ("max_steps", "num_simulations", "bank_stride", "bank_tail"):
        if not isinstance(manifest.get(key), int) or manifest[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if "Level6-1" not in manifest.get("levels", {}):
        raise ValueError("Level6-1 control must be included")
    for level in manifest["levels"].values():
        sha = level.get("checkpoint_sha256", "")
        if not level.get("checkpoint") or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Each checkpoint needs its precomputed SHA256")
    conditions = manifest.get("conditions", [])
    if profile not in ("original", "coverage_extension", "stall_gated"):
        raise ValueError("Unknown evaluation profile")
    if profile == "coverage_extension" and split != "development":
        raise ValueError("Coverage extension must use development seeds")
    expected_names = set(CONDITIONS) if split == "development" and profile == "original" else {"greedy", "sampled"}
    if profile == "stall_gated":
        from scripts.stall_sampling_controller import SETTINGS
        if set(manifest["levels"]) != {"Level1-1", "Level6-1", "Level1-3"}:
            raise ValueError("Stall-gated experiment requires 1-1 and both 6-1/1-3 controls")
        if manifest.get("stall_controller") != SETTINGS:
            raise ValueError("Stall-controller settings differ from the frozen protocol")
        old_seeds = {t[k] for t in manifest.get("previous_confirmation_trials", []) for k in ("env_seed", "search_seed")}
        reserved = manifest.get("confirmation_trials", [])
        new_seeds = {t[k] for t in reserved for k in ("env_seed", "search_seed")}
        development = manifest["trials"] if split == "development" else manifest["development_trials"]
        dev_seeds = {t[k] for t in development for k in ("env_seed", "search_seed")}
        if len(reserved) != 100 or len(new_seeds) != 200 or new_seeds & (old_seeds | dev_seeds):
            raise ValueError("Reserve 100 new seed pairs disjoint from prior confirmation and development")
        expected_names = {"greedy", "sampled", "stall_sampled"}
    if len(conditions) != len(expected_names) or {c["name"] for c in conditions} != expected_names:
        raise ValueError("Exactly the declared controller conditions are required")
    for c in conditions:
        actual = tuple(c[k] for k in ("leaf_batch", "eps", "alpha", "temperature"))
        expected = ALL_CONDITIONS[c["name"]]
        # Alpha is inactive when eps=0; allow the explicit disabled value too.
        if actual != expected and not (actual[1] == 0 and actual[2] == 0 and
                                       (actual[0], actual[1], actual[3]) ==
                                       (expected[0], expected[1], expected[3])):
            raise ValueError(f"Unexpected settings for {c['name']}")
    env_seeds, search_seeds = set(), set()
    for i, trial in enumerate(manifest["trials"]):
        if trial.get("episode_index") != i:
            raise ValueError("Trial indices must be consecutive from zero")
        for field, seen in (("env_seed", env_seeds), ("search_seed", search_seeds)):
            seed = trial.get(field)
            if not isinstance(seed, int) or not 0 <= seed < 2**32 or seed in seen:
                raise ValueError(f"{field} must contain unique uint32 seeds")
            seen.add(seed)
    if env_seeds & search_seeds:
        raise ValueError("Environment and search seed lists must be distinct")


def select_plan(manifest, *, smoke=False, episodes=None, conditions=None):
    validate_manifest(manifest)
    if smoke and manifest["split"] == "confirmation":
        raise ValueError("Do not consume confirmation seeds for smoke tests")
    if not smoke and (episodes is not None or conditions is not None):
        raise ValueError("Trial or condition overrides require --smoke")
    n = manifest["episodes"] if episodes is None else episodes
    if not 1 <= n <= manifest["episodes"]:
        raise ValueError("episodes must lie within the scheduled trial list")
    declared_names = [c["name"] for c in manifest["conditions"]]
    names = declared_names if conditions is None else conditions
    if not names or len(set(names)) != len(names) or any(n not in declared_names for n in names):
        raise ValueError("Unknown, empty, or duplicate condition subset")
    chosen = [c for c in manifest["conditions"] if c["name"] in names]
    return [(condition, trial) for trial in manifest["trials"][:n] for condition in chosen]


@contextmanager
def numpy_search_rng(seed):
    """MCTS uses global NumPy; keep its stream separate from reset and callers."""
    previous = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(previous)


class ResetRngRecorder:
    def __init__(self, rng):
        self.rng = rng
        self.draws = []

    def integers(self, *args, **kwargs):
        value = self.rng.integers(*args, **kwargs)
        self.draws.append(int(value))
        return value

    def __getattr__(self, name):
        return getattr(self.rng, name)


def reset_with_provenance(env):
    recorder = ResetRngRecorder(env.rng) if hasattr(env, "rng") else None
    if recorder is not None:
        env.rng = recorder
    try:
        obs = env.reset()
    finally:
        if recorder is not None:
            env.rng = recorder.rng
    info = getattr(env, "last_info", {})
    return obs, {
        "reset_rng_integer_draws": recorder.draws if recorder else [],
        "noop_frames_requested": recorder.draws[0] if recorder and len(recorder.draws) == 1 else None,
        "noop_count_note": "RNG draw, not measured executed frames if reset terminates early",
        "initial_observation_sha256": hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest(),
        "initial_x": x_position(info, getattr(env, "last_x", -1)),
        "initial_player_state": int(info.get("player_state", -1)),
        "initial_lives": int(info.get("lives", getattr(env, "curr_lives", -1))),
    }


def x_position(info, fallback=-1):
    if "player_x_posHi" in info and "player_x_posLo" in info:
        return 256 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
    return int(fallback)


def observation_uint8(obs):
    array = np.asarray(obs)
    if array.dtype == np.uint8:
        return array.copy()
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def distribution_stats(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    top = np.sort(probabilities)[-2:]
    return float(top[-1] - top[-2]), int(np.count_nonzero(probabilities == top[-1]))


def run_episode(env, decide, trial, *, max_steps, bank_stride, bank_tail):
    """Execute one scheduled trial; factories and network remain injectable for tests.

    decide(obs) returns (prior before noise, raw visits, selected action, root_q).
    Observation banks retain stride states plus the last bank_tail pre-action
    states, in chronological order with duplicates removed. No RGB is retained.
    """
    trace = {key: [] for key in (
        "priors", "policies", "actions", "steps", "x_before", "x_after",
        "prior_margin", "visit_margin", "prior_top_ties", "visit_top_ties",
        "rewards", "root_q", "done", "completed", "died", "timed_out",
        "player_state", "lives", "action_run_length",
    )}
    stride_obs, tail_obs = {}, deque(maxlen=bank_tail)
    total_return, completed, done, died = 0.0, False, False, False
    run_length, longest_run, previous_action, action_runs = 0, 0, None, 0
    try:
        obs, provenance = reset_with_provenance(env)
        final_x = provenance["initial_x"]
        lives = provenance["initial_lives"]
        # The reset has finished before the independent global search stream begins.
        with numpy_search_rng(trial["search_seed"]):
            for step in range(max_steps):
                stored_obs = observation_uint8(obs)
                if step % bank_stride == 0:
                    stride_obs[step] = stored_obs
                tail_obs.append((step, stored_obs))
                prior, visits, action, root_q = decide(obs)
                prior, visits = np.asarray(prior, np.float32), np.asarray(visits, np.float32)
                if (prior.ndim != 1 or visits.shape != prior.shape or len(prior) < 2 or
                        not np.isfinite(prior).all() or not np.isfinite(visits).all() or
                        np.any(prior < 0) or np.any(visits < 0) or
                        not np.isclose(prior.sum(), 1) or not np.isclose(visits.sum(), 1)):
                    raise ValueError("Controller returned invalid probability distributions")
                action = int(action)
                if not 0 <= action < len(prior):
                    raise ValueError("Controller returned invalid action")
                x_before = final_x
                obs, reward, done, info = env.step(action)
                final_x = x_position(info, final_x)
                new_lives = int(info.get("lives", lives))
                died_this_step = lives >= 0 and new_lives < lives
                died = died or died_this_step
                lives = new_lives
                completed = bool(done and info.get("level_complete", False))
                timed_out = bool(not done and step + 1 == max_steps)
                total_return += float(reward)
                if action == previous_action:
                    run_length += 1
                else:
                    action_runs += 1
                    run_length = 1
                previous_action = action
                longest_run = max(longest_run, run_length)
                prior_margin, prior_ties = distribution_stats(prior)
                visit_margin, visit_ties = distribution_stats(visits)
                row = dict(priors=prior.copy(), policies=visits.copy(), actions=action,
                           steps=step, x_before=x_before, x_after=final_x,
                           prior_margin=prior_margin, visit_margin=visit_margin,
                           prior_top_ties=prior_ties, visit_top_ties=visit_ties,
                           rewards=float(reward), root_q=float(root_q), done=bool(done),
                           completed=completed, died=died_this_step, timed_out=timed_out,
                           player_state=int(info.get("player_state", -1)), lives=lives,
                           action_run_length=run_length)
                for key in trace:
                    trace[key].append(row[key])
                if done:
                    break
    finally:
        env.close()
    trace = {key: np.asarray(value) for key, value in trace.items()}
    retained = {**stride_obs, **dict(tail_obs)}
    bank_steps = np.asarray(sorted(retained), dtype=np.int32)
    bank = {key: trace[key][bank_steps] for key in
            ("priors", "policies", "actions", "steps", "x_before", "x_after")}
    bank["obs"] = np.stack([retained[int(step)] for step in bank_steps])
    n_steps = len(trace["steps"])
    timed_out = bool(not done and n_steps >= max_steps)
    summary = {
        **provenance, **trial, "status": "ok", "completed": completed,
        "timed_out": timed_out, "died": bool(died),
        "terminal_failure": bool(done and not completed),
        "terminal_other": bool(done and not completed and not died),
        "steps": n_steps, "final_x": final_x, "total_return": total_return,
        "longest_action_run": longest_run, "n_action_runs": action_runs,
        "prior_top_tie_fraction": float(np.mean(trace["prior_top_ties"] > 1)),
        "visit_top_tie_fraction": float(np.mean(trace["visit_top_ties"] > 1)),
        "prior_margin_mean": float(np.mean(trace["prior_margin"])),
        "visit_margin_mean": float(np.mean(trace["visit_margin"])),
        "bank_states": len(bank_steps),
    }
    return summary, trace, bank


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def wilson_interval(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize_records(plan, records, split="development"):
    rows = []
    for name in dict.fromkeys(condition["name"] for condition, _ in plan):
        planned = sum(condition["name"] == name for condition, _ in plan)
        condition_records = [r for r in records if r["condition"] == name]
        good = [r for r in condition_records if r["status"] == "ok"]
        n, successes = len(good), sum(r["completed"] for r in good)
        errors = sum(r["status"] == "error" for r in condition_records)
        rows.append({
            "condition": name, "planned_trials": planned, "finished_trials": n,
            "pending_trials": planned - n, "infrastructure_errors": errors,
            "complete": n == planned, "level_completions": successes,
            "observed_completion_rate": successes / n if n else None,
            "wilson95_observed": wilson_interval(successes, n),
            "timeouts": sum(r["timed_out"] for r in good),
            "deaths": sum(r["died"] for r in good),
            "terminal_other": sum(r["terminal_other"] for r in good),
            "final_x_median": float(np.median([r["final_x"] for r in good])) if good else None,
            "distinct_noop_draws": sorted({r["noop_frames_requested"] for r in good
                                           if r["noop_frames_requested"] is not None}),
        })
    return {"split": split, "complete": all(r["complete"] for r in rows),
            "interpretation": f"Fixed-checkpoint {split} diagnostic; partial counts are interim",
            "conditions": rows}


def execute_plan(out, identity, plan, episode_runner):
    """Persist each scheduled result; only exact same-run artifacts can resume."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Advisory process lock prevents two jobs writing the same scheduled trial.
    import fcntl
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = out / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Resume refused: manifest/checkpoint/code identity differs")
        else:
            if any(out.glob("*/episode_*.json")):
                raise ValueError("Resume refused: results exist without identity")
            atomic_json(identity_path, identity)
        records = []
        starts = {}

        def check_matched_start(result):
            fingerprint = (result["initial_observation_sha256"],
                           result["noop_frames_requested"], result["initial_x"],
                           result["initial_player_state"], result["initial_lives"])
            key = result["episode_index"]
            if key in starts and starts[key] != fingerprint:
                raise ValueError(f"Controller starting conditions differ for trial {key}")
            starts[key] = fingerprint
        for condition, trial in plan:
            folder = out / condition["name"]
            folder.mkdir(exist_ok=True)
            stem = f"episode_{trial['episode_index']:04d}"
            result_path = folder / f"{stem}.json"
            expected = {"identity_sha256": canonical_sha(identity),
                        "condition": condition["name"], **trial}
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if any(saved.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Resume identity mismatch: {result_path}")
                if saved["status"] == "ok":
                    for artifact, sha in saved["artifacts"].items():
                        if file_sha(folder / artifact) != sha:
                            raise ValueError(f"Resume artifact mismatch: {folder / artifact}")
                    check_matched_start(saved)
                    records.append(saved)
                    continue
            try:
                result, trace, bank = episode_runner(condition, trial)
                result.update(expected)
                check_matched_start(result)
                metadata = {"episode_index": trial["episode_index"],
                            "condition": condition["name"], "completed": result["completed"],
                            "timed_out": result["timed_out"],
                            "checkpoint_sha256": identity["checkpoint_sha256"],
                            "manifest_sha256": identity["manifest_sha256"],
                            "env_seed": trial["env_seed"], "search_seed": trial["search_seed"]}
                artifacts = {}
                for suffix, arrays in (("trace", trace), ("bank", bank)):
                    path = folder / f"{stem}_{suffix}.npz"
                    artifact_metadata = metadata.copy()
                    if suffix == "trace":
                        artifact_metadata["episode_completed"] = artifact_metadata.pop("completed")
                        artifact_metadata["episode_timed_out"] = artifact_metadata.pop("timed_out")
                    atomic_npz(path, {**arrays, **artifact_metadata})
                    artifacts[path.name] = file_sha(path)
                result["artifacts"] = artifacts
                atomic_json(result_path, result)
            except Exception as error:
                result = {**expected, "status": "error", "error": repr(error),
                          "traceback": traceback.format_exc(),
                          "retry_policy": "Rerun this exact scheduled trial; never substitute a seed"}
                atomic_json(result_path, result)
                with (out / "infrastructure_errors.jsonl").open("a") as history:
                    history.write(json.dumps(result, sort_keys=True) + "\n")
                records.append(result)
                atomic_json(out / "summary.json", {**summarize_records(plan, records, identity.get("split", "development")),
                                                    "identity": identity})
                raise
            records.append(result)
            atomic_json(out / "summary.json", {**summarize_records(plan, records, identity.get("split", "development")), "identity": identity})
            print(f"{condition['name']} trial={trial['episode_index']} "
                  f"completed={result['completed']} steps={result['steps']} "
                  f"x={result['final_x']}", flush=True)
        summary = {**summarize_records(plan, records, identity.get("split", "development")), "identity": identity}
        atomic_json(out / "summary.json", summary)
        return summary


def implementation_sha():
    paths = [Path(__file__).resolve(), *[REPO_ROOT / name for name in (
        "src/env/env.py", "src/env/preprocess.py", "src/env/emulation.py",
        "src/env/mario_actions.py", "src/muzero/mcts.py", "src/muzero/node.py",
        "src/muzero/utils_mcts.py", "src/muzero/networks.py", "src/muzero/transforms.py",
        "scripts/stall_sampling_controller.py")]]
    return canonical_sha({str(path.relative_to(REPO_ROOT)): file_sha(path) for path in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--level", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--conditions", nargs="+")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Model evaluation must run inside a SLURM compute allocation")
    manifest = json.loads(args.manifest.read_text())
    plan = select_plan(manifest, smoke=args.smoke, episodes=args.episodes, conditions=args.conditions)
    if args.level not in manifest["levels"]:
        parser.error("Level is not in the frozen manifest")
    level = manifest["levels"][args.level]
    checkpoint = Path(level["checkpoint"])
    if file_sha(checkpoint) != level["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 does not match the frozen manifest")

    import torch
    from src.checkpoint import load_checkpoint
    from src.env.env import create_train_env
    from src.muzero.mcts import MCTS
    from src.muzero.networks import MuZeroNet

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    state = load_checkpoint(checkpoint, map_location=device)
    if file_sha(checkpoint) != level["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during loading")
    cfg = state["cfg_snapshot"]
    m = cfg["model"]
    net = MuZeroNet(
        input_channels=m["input_channels"], input_spatial=m["input_spatial"],
        hidden_channels=m["hidden_channels"], hidden_spatial=m["hidden_spatial"],
        num_actions=m["num_actions"], value_support=tuple(m["value_support"]),
        reward_support=tuple(m["reward_support"]), rep_blocks=tuple(m["rep_blocks"]),
        dyn_blocks=m["dyn_blocks"], pred_blocks=m["pred_blocks"],
    ).to(device)
    net.load_state_dict(state["online"], strict=True)
    net.eval().requires_grad_(False)
    env_cfg = cfg["env"]
    reset_settings = {
        "noop_max": int(env_cfg.get("noop_max", 0)),
        "skip_to_control": bool(env_cfg.get("skip_to_control", False)),
        "done_on_life_loss": bool(env_cfg.get("done_on_life_loss", True)),
        "completion_bonus": float(env_cfg.get("completion_bonus", 100.0)),
    }
    identity = {
        "manifest_sha256": canonical_sha(manifest), "checkpoint_sha256": level["checkpoint_sha256"],
        "checkpoint": str(checkpoint.resolve()), "implementation_sha256": implementation_sha(),
        "level": args.level, "experiment_id": manifest["experiment_id"], "split": manifest["split"],
        "smoke": args.smoke, "plan_sha256": canonical_sha(plan), "device": str(device),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "training_step": int(state.get("training_step", -1)), "env_step": int(state.get("env_step", -1)),
        "reset_settings": reset_settings,
        "inference": {"amp": False, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                      "cudnn_benchmark": torch.backends.cudnn.benchmark,
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
    }

    def episode_runner(condition, trial):
        from scripts.stall_sampling_controller import StallSamplingController
        monitor = (StallSamplingController(**manifest["stall_controller"])
                   if condition["name"] == "stall_sampled" else None)
        decision_metadata = []
        search = MCTS(
            discount=float(cfg["muzero"]["discount"]), num_simulations=manifest["num_simulations"],
            root_dirichlet_alpha=condition["alpha"], root_exploration_eps=condition["eps"],
            pb_c_base=float(cfg["mcts"]["pb_c_base"]), pb_c_init=float(cfg["mcts"]["pb_c_init"]),
            device=device, leaf_batch=condition["leaf_batch"],
        )

        @torch.inference_mode()
        def decide(obs):
            if monitor is not None:
                metadata = monitor.observe(x_position(env.last_info, env.last_x),
                                           int(env.last_info.get("player_state", -1)))
            else:
                metadata = {"controller_temperature": condition["temperature"],
                            "rescue_active": False, "rescue_trigger": False,
                            "rescue_remaining_after_action": 0, "stall_window_span": -1}
            decision_metadata.append(metadata)
            tensor = torch.from_numpy(obs).to(device, dtype=torch.float32).unsqueeze(0)
            _, logits, _ = net.initial_inference(tensor)
            prior = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            action, visits, root_q = search.run(obs, net, temperature=metadata["controller_temperature"],
                                               deterministic=False)
            return prior, visits, action, root_q

        env = create_train_env(
            level=args.level, int_path=env_cfg["int_path"], player_actions=None,
            n_frame=int(env_cfg["n_frame_stack"]), downsample=int(env_cfg["frame_skip"]),
            pad_to=int(m["input_spatial"]) if env_cfg["pad_to_input_spatial"] else None,
            seed=trial["env_seed"], **reset_settings,
        )
        result, trace, bank = run_episode(env, decide, trial, max_steps=manifest["max_steps"],
                                         bank_stride=manifest["bank_stride"], bank_tail=manifest["bank_tail"])
        for key in decision_metadata[0]:
            trace[key] = np.asarray([row[key] for row in decision_metadata])
        result["rescue_triggers"] = int(trace["rescue_trigger"].sum())
        result["rescue_decisions"] = int(trace["rescue_active"].sum())
        result["sampled_decisions"] = int((trace["controller_temperature"] > 0).sum())
        result["sampled_decision_fraction"] = result["sampled_decisions"] / result["steps"]
        result["rescue_trigger_steps"] = trace["steps"][trace["rescue_trigger"]].tolist()
        result["rescue_trigger_x"] = trace["x_before"][trace["rescue_trigger"]].tolist()
        return result, trace, bank

    execute_plan(args.out, identity, plan, episode_runner)


if __name__ == "__main__":
    main()
