"""Targeted lesions of a trained MuZeroNet, for region-specific planning work.

Ported from the Towers-of-Hanoi study (`~/region-specific-planning/Muzero-Hanoi`),
which uses ablations of MuZero's heads as stand-ins for focal brain lesions
(value -> PFC, policy -> cerebellar) and compares the resulting deficits against
human lesion data. Three conventions carry over unchanged, because results are
only comparable across the two studies if they do:

1. **A lesion is random re-initialisation** of the targeted parameters, not
   noise added to them and not zeroing. Here that means calling each leaf
   module's own ``reset_parameters()``, so a lesioned head gets exactly the
   distribution PyTorch would have given it at construction.
2. **Lesions are applied at evaluation time only.** Never retrain after
   lesioning for a main result.
3. **Lesioning one target leaves every other parameter bit-identical.**
   Enforced by ``tests/test_lesion.py::TestLesionContract``.

Where Mario differs from Hanoi, and it matters:

- Hanoi's policy/value/reward heads are standalone `nn.Linear` stacks hanging
  off the latent. Mario's ``PredictionNet`` puts policy and value behind a
  **shared residual trunk** (``prediction.blocks``), and the reward head sits
  inside ``DynamicsNet``. So "policy" here resets only the policy-specific
  conv/bn/fc, leaving the shared trunk intact — otherwise a policy lesion would
  silently damage value and break convention 3. The trunk is separately
  lesionable as ``pred_trunk``.
- Mario adds two targets Hanoi has no analogue for. ``transition`` is the latent
  forward model MuZero actually rolls out during search (``dynamics`` minus its
  reward head), and ``encoder`` is the perceptual front end. For a planning
  study ``transition`` is arguably the most interesting single target available:
  it damages imagination while leaving perception and both heads intact.

``projection_net`` is deliberately not a target — it is a learner-only
SimSiam head that no inference path touches, so lesioning it is a guaranteed
no-op and would only produce a misleading "no deficit" result.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

# Target name -> dotted module paths reset for that lesion. Every path must
# name a real submodule of MuZeroNet; validated in `resolve_targets`.
LESION_TARGETS: Dict[str, Tuple[str, ...]] = {
    # -- the three heads Hanoi ablates, same semantics -----------------------
    "policy": ("prediction.policy_conv", "prediction.policy_bn", "prediction.policy_fc"),
    "value": ("prediction.value_conv", "prediction.value_bn",
              "prediction.value_fc1", "prediction.value_fc2"),
    "reward": ("dynamics.reward_conv", "dynamics.reward_bn",
               "dynamics.reward_fc1", "dynamics.reward_fc2"),
    # -- Mario-specific targets ---------------------------------------------
    "transition": ("dynamics.conv_in", "dynamics.bn_in", "dynamics.blocks"),
    "encoder": ("representation",),
    "pred_trunk": ("prediction.blocks",),
}

# Condition label -> targets. The first eight are Hanoi's 2^3 head grid, kept in
# the same order so the two studies' figures line up; the rest are Mario-only.
CONDITIONS: Dict[str, Tuple[str, ...]] = {
    "intact": (),
    "policy": ("policy",),
    "value": ("value",),
    "reward": ("reward",),
    "policy+value": ("policy", "value"),
    "policy+reward": ("policy", "reward"),
    "value+reward": ("value", "reward"),
    "policy+value+reward": ("policy", "value", "reward"),
    "transition": ("transition",),
    "encoder": ("encoder",),
    "pred_trunk": ("pred_trunk",),
}

CONDITION_DISPLAY: Dict[str, str] = {
    "intact": "Intact",
    "policy": "Policy lesion\n(cerebellar)",
    "value": "Value lesion\n(PFC)",
    "reward": "Reward lesion",
    "policy+value": "Policy + Value",
    "policy+reward": "Policy + Reward",
    "value+reward": "Value + Reward",
    "policy+value+reward": "Policy + Value + Reward",
    "transition": "Transition lesion\n(forward model)",
    "encoder": "Encoder lesion\n(perceptual)",
    "pred_trunk": "Prediction trunk\n(shared by policy + value)",
}


def resolve_targets(net: nn.Module, targets: Iterable[str]) -> List[nn.Module]:
    """Modules named by `targets`, raising on an unknown or missing path."""
    named = dict(net.named_modules())
    out: List[nn.Module] = []
    for t in targets:
        if t not in LESION_TARGETS:
            raise KeyError(f"unknown lesion target {t!r}; known: {sorted(LESION_TARGETS)}")
        for path in LESION_TARGETS[t]:
            if path not in named:
                raise KeyError(
                    f"lesion target {t!r} names {path!r}, which this network does not "
                    f"have — the architecture changed and LESION_TARGETS is stale"
                )
            out.append(named[path])
    return out


def _reset_recursive(module: nn.Module) -> int:
    """Call reset_parameters() on every leaf that has one. Returns the count.

    Using each module's own initialiser (rather than a hand-rolled uniform)
    means a lesioned Conv2d/Linear/BatchNorm2d gets exactly its constructor-time
    distribution, which is what "random re-initialisation" should mean.
    """
    n = 0
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
            n += 1
    return n


def apply_lesion(net: nn.Module, targets: Sequence[str], seed: int = 0) -> dict:
    """Re-initialise `targets` in place. Returns a record of what was touched.

    `seed` makes the re-initialisation reproducible without disturbing global
    RNG state: the torch RNG is saved, seeded, and restored around the reset.
    Pass different seeds to sample several lesions of the same target — the
    deficit from a single random re-init is itself a random variable, so a main
    result should average over seeds rather than trusting one.
    """
    if not targets:
        return {"targets": [], "modules_reset": 0, "params_reset": 0, "seed": seed}

    modules = resolve_targets(net, targets)
    state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        n_mod = sum(_reset_recursive(m) for m in modules)
    finally:
        torch.set_rng_state(state)

    n_par = sum(p.numel() for m in modules for p in m.parameters())
    return {"targets": list(targets), "modules_reset": n_mod,
            "params_reset": n_par, "seed": seed}


def lesioned_parameter_names(net: nn.Module, targets: Sequence[str]) -> set:
    """Fully-qualified names of parameters and buffers a lesion would touch.

    Used by the contract test to assert everything *else* stays bit-identical.
    """
    named = dict(net.named_modules())
    prefixes = [p for t in targets for p in LESION_TARGETS[t]]
    out = set()
    for prefix in prefixes:
        mod = named[prefix]
        for n, _ in list(mod.named_parameters()) + list(mod.named_buffers()):
            out.add(f"{prefix}.{n}")
    return out
