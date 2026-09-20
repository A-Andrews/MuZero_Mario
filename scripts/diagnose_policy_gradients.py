"""Root versus averaged recurrent policy gradients on fresh, frozen rollout data.

No optimizer step. This isolates policy CE and is not a reconstruction of
historical replay, Adam updates, or the complete value/reward/consistency loss.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_controller_diagnostic import file_sha, atomic_json
from src.checkpoint import load_checkpoint
from src.muzero.networks import MuZeroNet


def choose_roots(steps, trace_length, unroll, count):
    """Cover elapsed decision time, not bank row index (banks oversample tails)."""
    steps = np.asarray(steps)
    eligible = np.flatnonzero(steps + unroll < trace_length)
    if not len(eligible):
        return np.array([], dtype=int)
    grid = np.linspace(steps[eligible[0]], steps[eligible[-1]], min(count, len(eligible)))
    return np.unique([eligible[np.argmin(abs(steps[eligible] - point))] for point in grid])


def load_sequences(folder, digest, unroll, roots_per_episode, max_episodes):
    rows, provenance = [], []
    for path in sorted(Path(folder).glob("episode_*_bank.npz"))[:max_episodes]:
        result_path = path.with_name(path.name.replace("_bank.npz", ".json"))
        trace_path = path.with_name(path.name.replace("_bank.npz", "_trace.npz"))
        result = json.loads(result_path.read_text())
        if result["status"] != "ok":
            raise ValueError(f"Invalid result {result_path}")
        for artifact in (path, trace_path):
            if file_sha(artifact) != result["artifacts"][artifact.name]:
                raise ValueError(f"Artifact hash mismatch: {artifact}")
        with np.load(path, allow_pickle=False) as bank, np.load(trace_path, allow_pickle=False) as trace:
            if any(str(data["checkpoint_sha256"].item()) != digest for data in (bank, trace)):
                raise ValueError("Wrong checkpoint in rollout data")
            steps = bank["steps"]
            actions, policies = trace["actions"], trace["policies"]
            if not np.array_equal(trace["steps"], np.arange(len(actions))):
                raise ValueError("Nonconsecutive trace steps")
            indices = choose_roots(steps, len(actions), unroll, roots_per_episode)
            # Access the compressed observation array once per episode.
            observations = bank["obs"]
            episode_rows = []
            for index in indices:
                t = int(steps[index])
                if trace["done"][t:t+unroll].any():
                    raise ValueError("Unroll crosses a terminal")
                episode_rows.append({"obs": observations[index], "actions": actions[t:t+unroll],
                                     "policies": policies[t:t+unroll+1], "step": t,
                                     "episode_index": result["episode_index"]})
            rows.append(episode_rows)
            provenance.append({"bank": str(path), "trace": str(trace_path),
                               "artifacts": result["artifacts"], "episode_index": result["episode_index"],
                               "completed": result["completed"], "selected_steps": [r["step"] for r in episode_rows],
                               "excluded_short_episode": not bool(episode_rows)})
    # Interleave episodes: each batch contains at most one root from each episode.
    batches = [[episode[i] for episode in rows if i < len(episode)]
               for i in range(max(map(len, rows), default=0))]
    if not batches:
        raise ValueError("No nonterminal unroll windows available")
    return batches, provenance


def group_name(name):
    if name.startswith("prediction.policy_"):
        return "policy_head"
    if name.startswith("prediction.blocks."):
        return "shared_prediction"
    if name.startswith("representation."):
        return "representation"
    if name.startswith("dynamics.") and not name.startswith("dynamics.reward_"):
        return "dynamics_trunk"
    return None


def gradient_pair_stats(root, recurrent):
    aa = sum(float((a.double() * a.double()).sum()) for a in root)
    bb = sum(float((b.double() * b.double()).sum()) for b in recurrent)
    ab = sum(float((a.double() * b.double()).sum()) for a, b in zip(root, recurrent))
    total = max(0., aa + bb + 2*ab)
    return {"root_norm": aa**.5, "recurrent_mean_norm": bb**.5,
            "recurrent_to_root_norm_ratio": (bb/aa)**.5 if aa else None,
            "cosine": ab/(aa*bb)**.5 if aa and bb else None,
            "recurrent_projection_on_root": ab/aa if aa else None,
            "combined_norm": total**.5,
            "combined_cosine_with_root": (aa+ab)/(aa*total)**.5 if aa and total else None}


def policy_losses(net, obs, actions, targets, hook_scale=.5):
    h, logits, _ = net.initial_step(obs)
    root = -(targets[:, 0] * F.log_softmax(logits.float(), -1)).sum(-1).mean()
    recurrent = []
    for k in range(actions.shape[1]):
        h, _, logits, _ = net.recurrent_step(h, actions[:, k])
        h.register_hook(lambda grad: grad * hook_scale)
        recurrent.append(-(targets[:, k+1] * F.log_softmax(logits.float(), -1)).sum(-1).mean())
    return root, torch.stack(recurrent).mean(), recurrent


def probe_batch(net, batch, device, hook_scale):
    obs = torch.from_numpy(np.stack([r["obs"] for r in batch])).to(device).float()/255
    actions = torch.from_numpy(np.stack([r["actions"] for r in batch])).to(device).long()
    targets = torch.from_numpy(np.stack([r["policies"] for r in batch])).to(device).float()
    names, params = zip(*[(n, p) for n, p in net.named_parameters() if group_name(n)])
    root, recurrent, per_step = policy_losses(net, obs, actions, targets, hook_scale)
    grad_root = torch.autograd.grad(root, params, retain_graph=True, allow_unused=True)
    grad_recur = torch.autograd.grad(recurrent, params, allow_unused=True)
    groups = {}
    for group in dict.fromkeys(group_name(n) for n in names):
        indices = [i for i, name in enumerate(names) if group_name(name) == group]
        left = [grad_root[i] if grad_root[i] is not None else torch.zeros_like(params[i]) for i in indices]
        right = [grad_recur[i] if grad_recur[i] is not None else torch.zeros_like(params[i]) for i in indices]
        groups[group] = {"n_parameters": sum(params[i].numel() for i in indices),
                         **gradient_pair_stats(left, right)}
    return {"roots": [{"episode_index": r["episode_index"], "step": r["step"]} for r in batch],
            "root_ce": root.item(), "recurrent_mean_ce": recurrent.item(),
            "recurrent_step_ce": [v.item() for v in per_step], "groups": groups}


def configure_mode(net, batch_stats):
    net.eval()
    for module in net.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.training = batch_stats
            # Same normalization as train mode, without running-stat writes.
            module.track_running_stats = not batch_stats


def summarize_batches(batches):
    out = {}
    for group in batches[0]["groups"]:
        metrics = {}
        for key in ("root_norm", "recurrent_mean_norm", "recurrent_to_root_norm_ratio", "cosine",
                    "recurrent_projection_on_root", "combined_cosine_with_root"):
            values = [b["groups"][group][key] for b in batches if b["groups"][group][key] is not None]
            metrics[key] = {"median": float(np.median(values)), "min": min(values), "max": max(values)} if values else None
        cosines = [b["groups"][group]["cosine"] for b in batches if b["groups"][group]["cosine"] is not None]
        metrics["negative_cosine_batches"] = sum(c < 0 for c in cosines)
        out[group] = metrics
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--banks-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roots-per-episode", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=30)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run gradient diagnostics on a compute node")
    if min(args.roots_per_episode, args.max_episodes) < 1:
        parser.error("Sample counts must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(1927)
    digest = file_sha(args.checkpoint)
    state = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = state["cfg_snapshot"]
    if args.level not in cfg["env"]["levels"]:
        raise ValueError("Level/checkpoint mismatch")
    keys = ("input_channels", "input_spatial", "hidden_channels", "hidden_spatial", "num_actions",
            "value_support", "reward_support", "rep_blocks", "dyn_blocks", "pred_blocks")
    net = MuZeroNet(**{k: cfg["model"][k] for k in keys})
    net.load_state_dict(state["online"], strict=True)
    del state
    reference = {name: t.detach().clone() for name, t in net.state_dict().items()}
    net.to(args.device)
    unroll = int(cfg["muzero"]["unroll_K"])
    report = {"diagnostic": "policy_gradients", "level": args.level, "checkpoint_sha256": digest,
              "script_sha256": file_sha(__file__), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "unroll_K": unroll, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "limits": ["Fresh development data, not historical replay; no learning occurs.",
                         "Policy CE only: root plus mean recurrent CE, equal importance weights, fp32, before clipping/Adam.",
                         "Value, reward, consistency losses, prioritized sampling and optimizer state are not measured.",
                         "Roots require K future nonterminal decisions; terminal/padded targets are excluded and counts recorded.",
                         "Batches reuse episodes at different times; batch ranges are descriptive, not independent confidence intervals.",
                         "Batch-stat and eval modes both retain BatchNorm and hidden min-max normalization.",
                         "Large recurrent gradients may reflect larger errors and need not be harmful; conflict is not behavioral causation."],
              "conditions": {}, "complete": False}
    for condition in ("greedy", "sampled"):
        batches, provenance = load_sequences(args.banks_dir / condition, digest, unroll,
                                             args.roots_per_episode, args.max_episodes)
        result = {"files": provenance, "n_episodes": sum(bool(p["selected_steps"]) for p in provenance),
                  "n_roots": sum(map(len, batches)), "n_batches": len(batches), "modes": {}}
        for name, batch_stats, hook in (("eval_hook_half", False, .5),
                                        ("batch_stats_hook_half", True, .5),
                                        ("batch_stats_hook_one", True, 1.)):
            configure_mode(net, batch_stats)
            measurements = []
            for index, batch in enumerate(batches):
                measurements.append(probe_batch(net, batch, args.device, hook))
                print(f"{args.level} {condition} {name} batch {index+1}/{len(batches)}", flush=True)
            if any(not torch.equal(t.detach().cpu(), reference[n]) for n, t in net.state_dict().items()):
                raise RuntimeError("Diagnostic changed weights or running statistics")
            result["modes"][name] = {"batches": measurements, "summary": summarize_batches(measurements)}
        report["conditions"][condition] = result
        atomic_json(args.out, report)
    report["complete"] = True
    report["parameters_and_buffers_unchanged"] = True
    if file_sha(args.checkpoint) != digest:
        raise RuntimeError("Checkpoint changed during diagnostic")
    atomic_json(args.out, report)


if __name__ == "__main__":
    main()
