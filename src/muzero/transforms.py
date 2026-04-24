"""Categorical value-support <-> scalar transforms (MuZero Atari style).

Forward: scalar -> signed_hyperbolic -> 2-hot distribution over the support grid.
Inverse: logits -> softmax -> expected scalar on support -> signed_parabolic.
"""
import torch


def signed_hyperbolic(x, eps=1e-3):
    """Forward reward/value transform (Pohlen et al. 2018)."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def signed_parabolic(x, eps=1e-3):
    """Inverse of signed_hyperbolic."""
    z = torch.sqrt(1.0 + 4.0 * eps * (eps + 1.0 + torch.abs(x))) / (2.0 * eps) - 1.0 / (2.0 * eps)
    return torch.sign(x) * (torch.square(z) - 1.0)


def scalar_to_support(x, support_min, support_max, support_size):
    """Map a batch of scalars to a 2-hot categorical distribution on the support.

    Args:
        x: tensor of shape (*,) containing scalars after `signed_hyperbolic`.
        support_min, support_max: inclusive endpoints of the support grid.
        support_size: number of bins.
    Returns:
        tensor of shape (*, support_size) summing to 1 along the last axis.
    """
    x = x.clamp(support_min, support_max)
    step = (support_max - support_min) / (support_size - 1)
    # Index in float into [0, support_size - 1]
    idx = (x - support_min) / step
    lower = idx.floor().long()
    upper = (lower + 1).clamp(max=support_size - 1)
    upper_w = idx - idx.floor()
    lower_w = 1.0 - upper_w
    out = torch.zeros(*x.shape, support_size, device=x.device, dtype=torch.float32)
    out.scatter_(-1, lower.unsqueeze(-1), lower_w.unsqueeze(-1))
    out.scatter_add_(-1, upper.unsqueeze(-1), upper_w.unsqueeze(-1))
    return out


def support_to_scalar(logits, support_min, support_max, support_size):
    """Map support-distribution logits back to a scalar (after signed_parabolic)."""
    probs = torch.softmax(logits, dim=-1)
    support = torch.linspace(support_min, support_max, support_size, device=logits.device)
    transformed = (probs * support).sum(dim=-1)
    return signed_parabolic(transformed)


def cross_entropy_on_support(pred_logits, target_scalar, support_min, support_max, support_size):
    """Cross-entropy loss between predicted support logits and a 2-hot target built
    from `signed_hyperbolic(target_scalar)`.

    Returns a per-sample loss vector (reduction='none').
    """
    target_transformed = signed_hyperbolic(target_scalar)
    target_dist = scalar_to_support(target_transformed, support_min, support_max, support_size)
    log_probs = torch.log_softmax(pred_logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1)
