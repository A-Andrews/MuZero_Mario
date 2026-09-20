"""Measure policy sensitivity to BatchNorm statistics on frozen rollout banks.

Run on a compute node. This changes statistics used by isolated inference copies,
not the checkpoint or a deployed controller. Real root observations do not
reproduce the learner's mixture of real and imagined latent-state batches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.checkpoint import load_checkpoint
from src.muzero.networks import MuZeroNet

MODES = ("eval", "prediction_batch_stats", "all_batch_stats")
LIMITS = [
    "Fresh rollout/search targets are not the historical replay used for training.",
    "Only real root observations are probed; batches do not reproduce the learner's real/imagined-state mix.",
    "Batch statistics depend on the other sampled observations and are not a proposed online controller.",
    "Search targets may contain exploration and finite-search variation; lower target KL does not establish better actions.",
    "No behavioral superiority or completion improvement is tested by this probe.",
    "Regroupings reuse the same states and do not increase the independent episode count.",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probabilities(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 2 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite N by A probability matrix")
    if (value < 0).any() or not np.allclose(value.sum(-1), 1, atol=2e-5):
        raise ValueError(f"{name} must contain normalized nonnegative probabilities")
    return value


def load_banks(folder, checkpoint_sha256, max_episodes=12, states_per_episode=24):
    """Load a bounded, evenly sampled bank separately for each controller arm."""
    if max_episodes < 1 or states_per_episode < 1:
        raise ValueError("episode and state limits must be positive")
    grouped = {}
    files = sorted(Path(folder).glob("*/episode_*_bank.npz"))
    for path in files:
        grouped.setdefault(path.parent.name, []).append(path)
    if not grouped:
        raise ValueError(f"no controller episode banks found under {folder}")
    result = {}
    for condition, paths in sorted(grouped.items()):
        chunks = []
        provenance = []
        episode_ids_seen = set()
        for path in paths[:max_episodes]:
            with np.load(path, allow_pickle=False) as data:
                sha = str(np.asarray(data["checkpoint_sha256"]).item())
                if sha != checkpoint_sha256:
                    raise ValueError(f"checkpoint SHA mismatch in {path}")
                obs = np.asarray(data["obs"])
                if obs.dtype != np.uint8 or obs.ndim != 4 or not len(obs):
                    raise ValueError(f"{path}: obs must be nonempty uint8 N by C by H by W")
                priors = _probabilities(data["priors"], f"{path}: priors")
                targets = _probabilities(data["policies"], f"{path}: policies")
                steps = np.asarray(data["steps"])
                episode_index = np.asarray(data["episode_index"])
                if episode_index.size == 1:
                    episode_id = int(episode_index.item())
                elif episode_index.shape == (len(obs),) and np.all(episode_index == episode_index[0]):
                    episode_id = int(episode_index[0])
                else:
                    raise ValueError(f"{path}: expected one episode_index for the whole bank")
                if episode_id in episode_ids_seen:
                    raise ValueError(f"{condition}: duplicate episode index {episode_id}")
                episode_ids_seen.add(episode_id)
                if priors.shape != targets.shape or len(priors) != len(obs) or steps.shape != (len(obs),):
                    raise ValueError(f"{path}: bank fields have inconsistent shapes")
                indices = np.linspace(0, len(obs) - 1, min(states_per_episode, len(obs)), dtype=int)
                chunks.append((obs[indices], priors[indices], targets[indices], steps[indices],
                               np.full(len(indices), episode_id, dtype=np.int64)))
                provenance.append({"path": str(path), "episode_index": episode_id,
                                   "available_states": len(obs), "selected_indices": indices.tolist(),
                                   "selected_steps": steps[indices].tolist()})
        result[condition] = {
            key: np.concatenate([chunk[i] for chunk in chunks], axis=0)
            for i, key in enumerate(("obs", "priors", "policies", "steps", "episode_index"))
        }
        result[condition]["files"] = provenance
    return result


def evaluate_mode(original, obs_u8, device, mode, batch_size, order=None):
    """Use an isolated copy and leave every source parameter, buffer and flag intact."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if batch_size < 1 or len(obs_u8) < 1:
        raise ValueError("batch_size and observation count must be positive")
    order = np.arange(len(obs_u8)) if order is None else np.asarray(order)
    if order.shape != (len(obs_u8),) or not np.array_equal(np.sort(order), np.arange(len(obs_u8))):
        raise ValueError("order must be a permutation of all observation indices")
    original_flags = {name: module.training for name, module in original.named_modules()}
    original_bn_flags = {name: module.track_running_stats for name, module in original.named_modules()
                         if isinstance(module, nn.modules.batchnorm._BatchNorm)}
    net = copy.deepcopy(original).to(device).eval()
    selected = []
    for name, module in net.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and (
            mode == "all_batch_stats" or
            (mode == "prediction_batch_stats" and (name == "prediction" or name.startswith("prediction.")))
        ):
            # training=True chooses batch statistics. Disabling tracking passes
            # no running buffers into BatchNorm and does not update counters.
            module.training = True
            module.track_running_stats = False
            selected.append(name)
    result = None
    batch_sizes = []
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            obs = torch.from_numpy(np.asarray(obs_u8[index])).to(device).float().div_(255.0)
            _, logits, _ = net.initial_step(obs)
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            if result is None:
                result = np.empty((len(obs_u8), probabilities.shape[1]), dtype=np.float64)
            result[index] = probabilities
            batch_sizes.append(len(index))
    # The copy must not learn or update running statistics either. Equality to
    # the untouched source checks parameters, means, variances and counters.
    source_state = original.state_dict()
    if any(not torch.equal(value.detach().cpu(), source_state[name].detach().cpu())
           for name, value in net.state_dict().items()):
        raise RuntimeError("probe copy changed model parameters or running buffers")
    if original_flags != {name: module.training for name, module in original.named_modules()}:
        raise RuntimeError("probe changed original module training flags")
    if original_bn_flags != {name: module.track_running_stats for name, module in original.named_modules()
                            if isinstance(module, nn.modules.batchnorm._BatchNorm)}:
        raise RuntimeError("probe changed original BatchNorm tracking flags")
    return result, {"batch_stat_modules": selected, "batch_sizes": batch_sizes,
                    "original_flags_unchanged": True, "parameters_and_buffers_unchanged": True}


def entropy(p):
    return -(p * np.log(np.maximum(p, 1e-30))).sum(axis=-1)


def kl(p, q):
    return (p * (np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(q, 1e-30)))).sum(axis=-1)


def metric_vectors(reference, candidate, targets):
    reference = _probabilities(reference, "reference")
    candidate = _probabilities(candidate, "candidate")
    targets = _probabilities(targets, "targets")
    if reference.shape != candidate.shape or reference.shape != targets.shape:
        raise ValueError("policy matrices must have identical shapes")
    top = np.sort(candidate, axis=-1)[:, -2:]
    ref_top = np.sort(reference, axis=-1)[:, -2:]
    margin = top[:, 1] - top[:, 0]
    ref_margin = ref_top[:, 1] - ref_top[:, 0]
    return {
        "tv_from_eval": 0.5 * np.abs(reference - candidate).sum(-1),
        "kl_eval_to_candidate": kl(reference, candidate),
        "kl_candidate_to_eval": kl(candidate, reference),
        "top1_changed_from_eval": (reference.argmax(-1) != candidate.argmax(-1)).astype(float),
        "entropy": entropy(candidate),
        "entropy_change_from_eval": entropy(candidate) - entropy(reference),
        "top1_probability": candidate.max(-1),
        "top2_margin": margin,
        "top2_margin_change_from_eval": margin - ref_margin,
        "fresh_target_entropy": entropy(targets),
        "fresh_target_kl": kl(targets, candidate),
        "fresh_target_kl_change_from_eval": kl(targets, candidate) - kl(targets, reference),
    }


def summarize(vectors, episode_indices, seed=314159):
    """Episode bootstrap avoids treating correlated states as independent trials."""
    episode_indices = np.asarray(episode_indices)
    unique = np.unique(episode_indices)
    episode_means = {
        key: np.asarray([values[episode_indices == episode].mean() for episode in unique])
        for key, values in vectors.items()
    }
    out = {
        "n_states": len(episode_indices), "n_episodes": len(unique),
        "pooled_state_means": {key: float(values.mean()) for key, values in vectors.items()},
        "per_episode_means": [
            {"episode_index": int(episode), "n_states": int((episode_indices == episode).sum()),
             **{key: float(values[i]) for key, values in episode_means.items()}}
            for i, episode in enumerate(unique)
        ],
        "episode_mean_ci95": {},
    }
    draws = (np.random.default_rng(seed).integers(0, len(unique), (2000, len(unique)))
             if len(unique) > 1 else None)
    for key, values in episode_means.items():
        interval = np.quantile(values[draws].mean(-1), [0.025, 0.975]).tolist() if draws is not None else None
        out["episode_mean_ci95"][key] = {"mean": float(values.mean()), "bootstrap_ci95": interval}
    return out


def probe_condition(net, bank, device, batch_size=32, regroupings=3, seed=1729, include_all=True):
    if regroupings < 2:
        raise ValueError("at least two regroupings are needed to measure batch composition dependence")
    obs, targets, episodes = bank["obs"], bank["policies"], bank["episode_index"]
    reference, reference_checks = evaluate_mode(net, obs, device, "eval", batch_size)
    result = {
        "bank_files": bank["files"],
        "frozen_eval": summarize(metric_vectors(reference, reference, targets), episodes),
        "frozen_eval_checks": reference_checks,
        "recorded_prior_comparison": summarize(metric_vectors(reference, bank["priors"], targets), episodes),
        "regrouping_seeds": [seed + i for i in range(regroupings)], "arms": {},
    }
    modes = MODES[1:] if include_all else MODES[1:2]
    for mode in modes:
        predictions, all_vectors, records = [], [], []
        for repetition in range(regroupings):
            order = np.random.default_rng(seed + repetition).permutation(len(obs))
            probabilities, checks = evaluate_mode(net, obs, device, mode, batch_size, order)
            vectors = metric_vectors(reference, probabilities, targets)
            predictions.append(probabilities)
            all_vectors.append(vectors)
            records.append({"regrouping": repetition, "seed": seed + repetition,
                            "checks": checks, "metrics": summarize(vectors, episodes)})
        averaged_vectors = {key: np.stack([v[key] for v in all_vectors]).mean(0) for key in all_vectors[0]}
        composition_tv, composition_top1 = [], []
        for left, right in combinations(predictions, 2):
            composition_tv.append(0.5 * np.abs(left - right).sum(-1))
            composition_top1.append((left.argmax(-1) != right.argmax(-1)).astype(float))
        composition = {"pairwise_regrouping_tv": np.stack(composition_tv).mean(0),
                       "pairwise_regrouping_top1_change": np.stack(composition_top1).mean(0)}
        result["arms"][mode] = {
            "mean_metrics_across_regroupings": summarize(averaged_vectors, episodes),
            "composition_dependence": summarize(composition, episodes), "regroupings": records,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--banks-dir", required=True)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=12, help="Per controller condition")
    parser.add_argument("--states-per-episode", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--regroupings", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--prediction-only", action="store_true", help="Omit the whole-network BN arm")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this diagnostic inside a SLURM compute allocation")
    torch.set_num_threads(1)
    if args.batch_size < 1 or args.regroupings < 2:
        parser.error("batch-size must be positive and regroupings must be at least two")
    digest = file_sha256(args.checkpoint)
    banks = load_banks(args.banks_dir, digest, args.max_episodes, args.states_per_episode)
    state = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = state["cfg_snapshot"]
    if args.level not in cfg["env"]["levels"]:
        raise ValueError("requested level is absent from checkpoint configuration")
    model_keys = ("input_channels", "input_spatial", "hidden_channels", "hidden_spatial", "num_actions",
                  "value_support", "reward_support", "rep_blocks", "dyn_blocks", "pred_blocks")
    net = MuZeroNet(**{key: cfg["model"][key] for key in model_keys})
    net.load_state_dict(state["online"], strict=True)
    del state
    net.to(args.device).eval()
    report = {"schema_version": 1, "diagnostic": "policy_batchnorm", "level": args.level,
              "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": digest,
              "args": vars(args), "limits": LIMITS, "conditions": {}}
    for condition, bank in banks.items():
        print(f"[batchnorm] {args.level} {condition}: {len(bank['obs'])} states, "
              f"{len(bank['files'])} episodes", flush=True)
        report["conditions"][condition] = probe_condition(
            net, bank, args.device, args.batch_size, args.regroupings, args.seed, not args.prediction_only)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"[batchnorm] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
