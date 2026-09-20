"""Measure policy fit and state information on frozen-checkpoint eval banks.

Run inside a SLURM allocation. These banks contain *fresh evaluation search
targets*, not the original replay targets on which the checkpoint was trained.
Sequential and batched searches are compared as interventions, never as oracles.
No training or production search code is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ACTION_NAMES = [
    "NOOP", "right", "right+A", "right+B", "right+A+B", "A",
    "left", "left+A", "left+B", "left+A+B", "down", "up",
]
BUTTON_GROUPS = {
    "right": [1, 2, 3, 4], "left": [6, 7, 8, 9],
    "jump": [2, 4, 5, 7, 9], "run": [3, 4, 8, 9],
    "noop": [0], "down": [10], "up": [11],
}
LOG_FLOOR = 1e-12


def probabilities(values, name="probabilities"):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or not len(p) or p.shape[1] < 2:
        raise ValueError(f"{name}: expected nonempty (states, actions) array")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f"{name}: probabilities must be finite and nonnegative")
    if not np.allclose(p.sum(1), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name}: rows must sum to one; pass raw normalized visits")
    return p / p.sum(1, keepdims=True)


def entropy(p):
    return -(p * np.log(np.clip(p, LOG_FLOOR, 1.0))).sum(1)


def cross_entropy(target, prior):
    return -(target * np.log(np.clip(prior, LOG_FLOOR, 1.0))).sum(1)


def distribution_summary(p):
    ordered = np.sort(p, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    top = p == p.max(1, keepdims=True)
    summary = {
        "entropy_mean_nats": float(entropy(p).mean()),
        "max_probability_mean": float(ordered[:, -1].mean()),
        "top_two_margin_mean": float(margin.mean()),
        "top_two_margin_p10": float(np.quantile(margin, 0.1)),
        "top_two_margin_p50": float(np.quantile(margin, 0.5)),
        "top_two_margin_p90": float(np.quantile(margin, 0.9)),
        "exact_top_tie_fraction": float((top.sum(1) > 1).mean()),
        "top_tie_size_mean": float(top.sum(1).mean()),
        "action_probability_marginal": p.mean(0).tolist(),
        "argmax_action_marginal": (np.bincount(p.argmax(1), minlength=p.shape[1]) / len(p)).tolist(),
        "per_action_probability_std": p.std(0).tolist(),
    }
    if p.shape[1] == 12:
        summary["button_group_probability_mass"] = {
            key: float(p[:, indices].sum(1).mean()) for key, indices in BUTTON_GROUPS.items()
        }
    return summary


def fit_metrics(target, prior):
    """Separate irreducible target entropy from mismatch; retain exact top ties."""
    target, prior = probabilities(target, "target"), probabilities(prior, "prior")
    if target.shape != prior.shape:
        raise ValueError("target and prior shapes differ")
    ce = cross_entropy(target, prior)
    kl = ce - entropy(target)
    target_top = target == target.max(1, keepdims=True)
    chosen = prior.argmax(1)
    return {
        "n_states": len(target),
        "cross_entropy_mean_nats": float(ce.mean()),
        "target_entropy_mean_nats": float(entropy(target).mean()),
        "kl_target_to_prior_mean_nats": float(kl.mean()),
        "kl_target_to_prior_p90_nats": float(np.quantile(kl, 0.9)),
        "total_variation_mean": float((0.5 * np.abs(target - prior).sum(1)).mean()),
        "argmax_agreement": float((target.argmax(1) == chosen).mean()),
        "tie_aware_argmax_agreement": float(target_top[np.arange(len(target)), chosen].mean()),
        "target": distribution_summary(target),
        "prior": distribution_summary(prior),
    }


def repeated_target_metrics(draws, prior):
    """Separate agreement with a mean noisy target from random target variation."""
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[1] < 2:
        raise ValueError("expected states by at least two repeats by actions")
    n, repeats, actions = draws.shape
    draws = probabilities(draws.reshape(-1, actions)).reshape(n, repeats, actions)
    prior = probabilities(prior)
    if prior.shape != (n, actions):
        raise ValueError("prior shape does not match repeated targets")
    mean_target = draws.mean(1)
    single_kl = np.stack([
        cross_entropy(draws[:, i], prior) - entropy(draws[:, i])
        for i in range(repeats)
    ]).mean(0)
    mean_kl = cross_entropy(mean_target, prior) - entropy(mean_target)
    return {
        "n_states": n, "search_repeats_per_state": repeats,
        "mean_target_fit": fit_metrics(mean_target, prior),
        "mean_individual_target_kl_nats": float(single_kl.mean()),
        "target_variation_contribution_nats": float((single_kl - mean_kl).mean()),
        "per_state_mean_target_kl_nats": mean_kl.tolist(),
        "per_state_individual_target_kl_nats": single_kl.tolist(),
        "mean_target_probabilities": mean_target.tolist(),
        "interpretation": "Finite-repeat target mean estimates this search controller's expectation; not historical training targets or action-quality ground truth.",
    }


def state_information(target, prior, episode_ids, seed=0, repeats=16):
    """Descriptive controls: permuting predictions equals permuting observations.

    Frozen eval-mode inference is a per-observation function, so reusing its
    predictions avoids redundant network work. Each shuffle is a derangement
    within an episode. The constant policies are computed on the same sample;
    their comparison is descriptive, not held-out generalization performance.
    """
    target, prior = probabilities(target), probabilities(prior)
    episode_ids = np.asarray(episode_ids)
    if target.shape != prior.shape or episode_ids.shape != (len(target),):
        raise ValueError("state-information arrays are not aligned")
    correct_ce = float(cross_entropy(target, prior).mean())
    constant_prior = np.broadcast_to(prior.mean(0), prior.shape)
    constant_target = np.broadcast_to(target.mean(0), target.shape)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(episode_ids == key) for key in np.unique(episode_ids)]
    eligible = np.concatenate([g for g in groups if len(g) > 1]) if any(len(g) > 1 for g in groups) else np.array([], dtype=int)
    shuffled_ce = []
    for _ in range(repeats if len(eligible) else 0):
        permutation = np.arange(len(target))
        for group in groups:
            if len(group) > 1:
                order = rng.permutation(group)
                permutation[order] = np.roll(order, 1)
        shuffled_ce.append(float(cross_entropy(target[eligible], prior[permutation[eligible]]).mean()))
    baseline = float(cross_entropy(target[eligible], prior[eligible]).mean()) if len(eligible) else None
    result = {
        "correct_state_ce_nats": correct_ce,
        "constant_mean_prior_ce_nats": float(cross_entropy(target, constant_prior).mean()),
        "constant_target_marginal_ce_nats_in_sample": float(cross_entropy(target, constant_target).mean()),
        "constant_policy_caveat": "Both constants use these same sampled states; this is not a held-out score.",
        "within_episode_shuffle_n_states": int(len(eligible)),
        "within_episode_shuffle_repeats": len(shuffled_ce),
        "within_episode_shuffle_ce_mean_nats": float(np.mean(shuffled_ce)) if shuffled_ce else None,
        "within_episode_shuffle_ce_std_nats": float(np.std(shuffled_ce)) if shuffled_ce else None,
        "within_episode_shuffle_ce_minus_correct_nats": float(np.mean(shuffled_ce) - baseline) if shuffled_ce else None,
        "interpretation": "Positive shuffle-minus-correct CE means the policy carries state information about these fresh search targets; it does not establish action quality.",
    }
    result["constant_mean_prior_ce_minus_correct_nats"] = result["constant_mean_prior_ce_nats"] - correct_ce
    return result


def stratified_indices(steps, x_before, limit, tail_steps=200, seed=0):
    """Mix elapsed-step, observed-progress and tail coverage without duplicates."""
    steps, xs = np.asarray(steps), np.asarray(x_before)
    if steps.ndim != 1 or xs.shape != steps.shape or not len(steps) or limit < 1:
        raise ValueError("invalid state selection inputs")
    if not np.isfinite(steps).all() or not np.isfinite(xs).all():
        raise ValueError("state positions must be finite")
    n = min(int(limit), len(steps))
    if n == len(steps):
        return np.argsort(steps, kind="stable")
    selected = []

    def add_coverage(candidates, coordinate, count):
        if not len(candidates) or count < 1:
            return
        vals = coordinate[candidates]
        for value in np.linspace(vals.min(), vals.max(), count):
            index = int(candidates[np.argmin(np.abs(vals - value))])
            if index not in selected:
                selected.append(index)

    all_rows = np.arange(len(steps))
    temporal_count = max(2, n // 2) if n > 1 else 1
    add_coverage(all_rows, steps, temporal_count)
    add_coverage(all_rows, xs, n // 4)
    tail = all_rows[steps >= steps.max() - max(1, tail_steps) + 1]
    add_coverage(tail, steps, n - temporal_count - n // 4)
    remaining = np.setdiff1d(all_rows, selected)
    if len(selected) < n:
        selected.extend(np.random.default_rng(seed).choice(remaining, n - len(selected), replace=False).tolist())
    return np.asarray(sorted(selected[:n], key=lambda i: (steps[i], i)), dtype=np.int64)


def choose_episode_files(banks_dir, max_episodes):
    """The cap is per condition; evenly span available episode indices."""
    grouped = {}
    for path in sorted(Path(banks_dir).rglob("episode_*_bank.npz")):
        grouped.setdefault(path.parent, []).append(path)
    selected = []
    for files in grouped.values():
        rows = np.linspace(0, len(files) - 1, min(max_episodes, len(files)), dtype=int)
        selected.extend(files[i] for i in rows)
    return selected


def _scalar(bank, key):
    value = np.asarray(bank[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    return value.reshape(-1)[0].item()


def read_bank(path, expected_sha256, states_per_episode, seed=0, tail_steps=200):
    required = {
        "obs", "priors", "policies", "actions", "steps", "x_before", "x_after",
        "episode_index", "condition", "completed", "timed_out", "checkpoint_sha256",
        "env_seed", "search_seed",
    }
    with np.load(path, allow_pickle=False) as bank:
        missing = required - set(bank.files)
        if missing:
            raise ValueError(f"{path}: missing bank fields {sorted(missing)}")
        if str(_scalar(bank, "checkpoint_sha256")) != expected_sha256:
            raise ValueError(f"{path}: checkpoint SHA256 mismatch")
        obs = bank["obs"]
        if obs.ndim != 4 or obs.dtype != np.uint8 or not len(obs):
            raise ValueError(f"{path}: obs must be nonempty uint8 (N,C,H,W)")
        n = len(obs)
        arrays = {key: bank[key] for key in ("priors", "policies", "actions", "steps", "x_before", "x_after")}
        if any(len(value) != n for value in arrays.values()):
            raise ValueError(f"{path}: inconsistent state counts")
        priors = probabilities(arrays["priors"], "stored priors")
        policies = probabilities(arrays["policies"], "stored policies")
        if priors.shape != policies.shape:
            raise ValueError(f"{path}: prior and target shapes differ")
        for key in ("actions", "steps", "x_before", "x_after"):
            if arrays[key].shape != (n,) or not np.issubdtype(arrays[key].dtype, np.integer):
                raise ValueError(f"{path}: {key} must be an integer vector")
        if (arrays["actions"] < 0).any() or (arrays["actions"] >= priors.shape[1]).any():
            raise ValueError(f"{path}: action outside action space")
        if (arrays["steps"] < 0).any() or len(np.unique(arrays["steps"])) != n:
            raise ValueError(f"{path}: steps must be unique and nonnegative")
        episode = np.asarray(bank["episode_index"])
        if episode.size not in (1, n) or len(np.unique(episode)) != 1:
            raise ValueError(f"{path}: each bank must contain one episode")
        indices = stratified_indices(arrays["steps"], arrays["x_before"], states_per_episode, tail_steps, seed)
        condition = str(_scalar(bank, "condition"))
        episode_index = int(episode.reshape(-1)[0])
        metadata = {
            "path": str(Path(path).resolve()), "condition": condition,
            "episode_index": episode_index, "episode_key": f"{condition}:{episode_index}",
            "completed": bool(_scalar(bank, "completed")), "timed_out": bool(_scalar(bank, "timed_out")),
            "env_seed": int(_scalar(bank, "env_seed")), "search_seed": int(_scalar(bank, "search_seed")),
            "n_available_correlated_states": n, "n_selected_correlated_states": len(indices),
            "selected_bank_rows": indices.tolist(), "selected_steps": arrays["steps"][indices].tolist(),
            "selected_x_before": arrays["x_before"][indices].tolist(),
            "last_saved_step": int(arrays["steps"].max()),
        }
        return {
            "metadata": metadata, "obs": obs[indices].copy(),
            "priors": priors[indices], "policies": policies[indices],
            **{key: value[indices] for key, value in arrays.items() if key not in ("priors", "policies")},
        }


def balanced_search_indices(episode_ids, limit, seed=0):
    """Round-robin over episodes, first spreading episode selection by condition."""
    rng = np.random.default_rng(seed)
    episode_ids = np.asarray(episode_ids)
    by_condition = {}
    for key in np.unique(episode_ids):
        by_condition.setdefault(str(key).rsplit(":", 1)[0], []).append(key)
    episode_order = []
    while any(by_condition.values()):
        for keys in by_condition.values():
            if keys:
                episode_order.append(keys.pop(0))
    groups = [rng.permutation(np.flatnonzero(episode_ids == key)).tolist() for key in episode_order]
    selected = []
    while len(selected) < min(limit, len(episode_ids)):
        for group in groups:
            if group:
                selected.append(group.pop())
                if len(selected) >= min(limit, len(episode_ids)):
                    break
    return np.asarray(selected, dtype=np.int64)


def _build_net(cfg):
    from src.muzero.networks import MuZeroNet
    keys = ("input_channels", "input_spatial", "hidden_channels", "hidden_spatial", "num_actions", "dyn_blocks", "pred_blocks")
    kwargs = {key: cfg[key] for key in keys}
    kwargs.update({key: tuple(cfg[key]) for key in ("value_support", "reward_support", "rep_blocks")})
    return MuZeroNet(**kwargs)


def _predict(net, obs, device, batch_size):
    import torch
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(obs), batch_size):
            tensor = torch.from_numpy(obs[start:start + batch_size]).to(device).float().div_(255.0)
            _, logits, _ = net.initial_inference(tensor)
            predictions.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return probabilities(np.concatenate(predictions))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition_report(target, prior, stored_prior, actions, episode_ids, seed, repeats):
    report = {
        "n_episodes": len(np.unique(episode_ids)), "n_correlated_states": len(target),
        "recomputed_float32_prior_fit": fit_metrics(target, prior),
        "stored_evaluator_prior_fit": fit_metrics(target, stored_prior),
        "state_information": state_information(target, prior, episode_ids, seed, repeats),
        "stored_vs_recomputed_prior": {
            "mean_total_variation": float((0.5 * np.abs(stored_prior - prior).sum(1)).mean()),
            "max_absolute_probability_difference": float(np.abs(stored_prior - prior).max()),
            "argmax_disagreement": float((stored_prior.argmax(1) != prior.argmax(1)).mean()),
        },
        "selected_action_marginal": (np.bincount(actions, minlength=target.shape[1]) / len(actions)).tolist(),
    }
    if target.shape[1] == 12:
        report["selected_button_group_frequency"] = {key: float(np.isin(actions, values).mean()) for key, values in BUTTON_GROUPS.items()}
    episodes = []
    for key in np.unique(episode_ids):
        mask = episode_ids == key
        fit = fit_metrics(target[mask], prior[mask])
        episodes.append({"episode_key": str(key), **{name: value for name, value in fit.items() if name not in ("target", "prior")}})
    report["per_episode_fit"] = episodes
    report["equal_episode_mean_kl_nats"] = float(np.mean([e["kl_target_to_prior_mean_nats"] for e in episodes]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--banks-dir", required=True)
    parser.add_argument("--out", required=True, help="output JSON file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=12, help="maximum episodes PER CONDITION")
    parser.add_argument("--states-per-episode", type=int, default=24)
    parser.add_argument("--search-states", type=int, default=64, help="total states across conditions; 0 skips re-search")
    parser.add_argument("--sims", nargs="+", type=int, default=[50], help="fixed budgets to compare, e.g. 50 200")
    parser.add_argument("--leaf-batches", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--tail-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-repeats", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-repeat-states", type=int, default=8)
    parser.add_argument("--noise-repeats", type=int, default=5)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this diagnostic inside a SLURM compute allocation.")
    if min(args.max_episodes, args.states_per_episode, args.batch_size, args.tail_steps, args.shuffle_repeats, *args.sims, *args.leaf_batches) < 1 or args.search_states < 0:
        parser.error("counts must be positive (search-states may be zero)")
    if args.noise_repeat_states < 0 or args.noise_repeats < 2:
        parser.error("noise-repeat-states must be nonnegative and noise-repeats at least two")

    import torch
    from src.checkpoint import load_checkpoint
    from src.muzero.mcts import MCTS

    torch.set_num_threads(1)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_hash = _sha256(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = checkpoint["cfg_snapshot"]
    if args.level not in cfg["env"]["levels"]:
        parser.error(f"{args.level} is not in the checkpoint's configured levels")
    files = choose_episode_files(args.banks_dir, args.max_episodes)
    if not files:
        parser.error("no episode_*_bank.npz files found")
    banks = [read_bank(path, checkpoint_hash, args.states_per_episode, args.seed + i, args.tail_steps) for i, path in enumerate(files)]
    episode_keys = [bank["metadata"]["episode_key"] for bank in banks]
    if len(set(episode_keys)) != len(episode_keys):
        parser.error("duplicate condition/episode identities across banks")
    metadata = [bank["metadata"] for bank in banks]
    obs, stored_priors, targets, actions = (np.concatenate([bank[key] for bank in banks]) for key in ("obs", "priors", "policies", "actions"))
    episode_ids = np.concatenate([np.repeat(bank["metadata"]["episode_key"], len(bank["obs"])) for bank in banks])
    conditions = np.concatenate([np.repeat(bank["metadata"]["condition"], len(bank["obs"])) for bank in banks])
    net = _build_net(cfg["model"]).to(device).eval()
    net.load_state_dict(checkpoint["online"], strict=True)
    del checkpoint
    priors = _predict(net, obs, device, args.batch_size)
    if targets.shape != priors.shape:
        parser.error("bank action space does not match checkpoint")
    report = {
        "schema_version": 1, "level": args.level,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "source": "fresh frozen-checkpoint evaluator search targets",
        "original_training_replay_available": False,
        "n_episodes": len(banks), "n_correlated_states": len(obs),
        "action_names": ACTION_NAMES if targets.shape[1] == 12 else list(range(targets.shape[1])),
        "configuration": vars(args), "checkpoint_mcts_configuration": cfg["mcts"],
        "sampling": "Per condition, evenly span episode files; within episodes mix elapsed steps, observed x progress and last saved 200-step window (configurable).",
        "inference": "Recomputed local float32 inference, eval mode. Stored-prior differences are reported separately.",
        "limitations": [
            "States within episodes are correlated; no state-level confidence intervals or independent-trial completion rates are inferred.",
            "These freshly generated targets do not measure fit to original training replay, and are themselves influenced by the trained prior.",
            "A sharper target or better target fit does not establish better real actions or level completion.",
            "State shuffles and constants are descriptive controls, not held-out generalization tests.",
            "Button groups overlap: right+jump contributes to both groups.",
            "Exact ties use exact stored probabilities; logarithms are floored at 1e-12.",
        ],
        "episodes": metadata, "conditions": {}, "research": [],
    }
    for condition in np.unique(conditions):
        mask = conditions == condition
        summary = _condition_report(targets[mask], priors[mask], stored_priors[mask], actions[mask], episode_ids[mask], args.seed, args.shuffle_repeats)
        report["conditions"][str(condition)] = summary
        fit = summary["recomputed_float32_prior_fit"]
        print(f"{condition}: {summary['n_episodes']} episodes / {mask.sum()} correlated states; CE={fit['cross_entropy_mean_nats']:.4f}, target H={fit['target_entropy_mean_nats']:.4f}, KL={fit['kl_target_to_prior_mean_nats']:.4f}", flush=True)

    selected = balanced_search_indices(episode_ids, args.search_states, args.seed)
    report["research_state_rows"] = selected.tolist()
    report["research_state_episode_keys"] = episode_ids[selected].tolist()
    search_targets = {}
    for sims in args.sims if len(selected) else []:
        for leaf_batch in args.leaf_batches:
            search = MCTS(
                discount=float(cfg["muzero"]["discount"]), num_simulations=sims,
                root_dirichlet_alpha=0.0, root_exploration_eps=0.0,
                pb_c_base=float(cfg["mcts"]["pb_c_base"]), pb_c_init=float(cfg["mcts"]["pb_c_init"]),
                device=device, leaf_batch=leaf_batch,
            )
            rows = []
            for index in selected:
                # Reset per state so earlier states' tie counts cannot shift later streams.
                np.random.seed((args.seed + int(index)) % (2**32))
                rows.append(search.run(obs[index].astype(np.float32) / 255.0, net, temperature=0.0, deterministic=True)[1])
            fresh = probabilities(np.stack(rows))
            search_targets[(sims, leaf_batch)] = fresh
            for condition in np.unique(conditions[selected]):
                mask = conditions[selected] == condition
                fit = fit_metrics(fresh[mask], priors[selected][mask])
                report["research"].append({
                    "condition": str(condition), "num_simulations": sims, "leaf_batch": leaf_batch,
                    "root_noise_eps": 0.0, "temperature": 0.0,
                    "n_episodes": len(np.unique(episode_ids[selected][mask])),
                    "prior_fit_to_fresh_search": fit,
                    "fresh_vs_stored_target_tv_mean": float((0.5 * np.abs(fresh[mask] - targets[selected][mask]).sum(1)).mean()),
                })
            print(f"Re-search complete: sims={sims}, leaf_batch={leaf_batch}, states={len(selected)}", flush=True)
    report["research_pairwise_sensitivity"] = []
    for sims in args.sims:
        if (sims, 1) not in search_targets or (sims, 4) not in search_targets:
            continue
        one, four = search_targets[(sims, 1)], search_targets[(sims, 4)]
        for condition in np.unique(conditions[selected]):
            mask = conditions[selected] == condition
            report["research_pairwise_sensitivity"].append({
                "condition": str(condition), "num_simulations": sims, "n_states": int(mask.sum()),
                "batch1_vs_batch4_tv_mean": float((0.5 * np.abs(one[mask] - four[mask]).sum(1)).mean()),
                "batch1_vs_batch4_argmax_disagreement": float((one[mask].argmax(1) != four[mask].argmax(1)).mean()),
                "interpretation": "Algorithm sensitivity; neither search is an action-quality oracle. Exact UCB tie breaking remains seeded and random.",
            })
    repeated_rows = balanced_search_indices(episode_ids, args.noise_repeat_states, args.seed + 17000)
    if len(repeated_rows):
        search = MCTS(
            discount=float(cfg["muzero"]["discount"]), num_simulations=50,
            root_dirichlet_alpha=0.25, root_exploration_eps=0.25,
            pb_c_base=float(cfg["mcts"]["pb_c_base"]), pb_c_init=float(cfg["mcts"]["pb_c_init"]),
            device=device, leaf_batch=4,
        )
        draws, search_seeds = [], []
        for index in repeated_rows:
            state_draws, state_seeds = [], []
            for repetition in range(args.noise_repeats):
                seed = (args.seed + 5100001 + int(index) * args.noise_repeats + repetition) % (2**32)
                np.random.seed(seed)
                state_seeds.append(seed)
                state_draws.append(search.run(obs[index].astype(np.float32) / 255.0, net,
                                             temperature=0.0, deterministic=False)[1])
            draws.append(state_draws)
            search_seeds.append(state_seeds)
        report["repeated_noisy_search"] = {
            **repeated_target_metrics(draws, priors[repeated_rows]),
            "state_rows": repeated_rows.tolist(), "episode_keys": episode_ids[repeated_rows].tolist(),
            "search_seeds": search_seeds, "num_simulations": 50, "leaf_batch": 4,
            "root_noise_eps": 0.25, "dirichlet_alpha": 0.25,
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
