import torch

from src.muzero.transforms import (
    cross_entropy_on_support,
    scalar_to_support,
    signed_hyperbolic,
    signed_parabolic,
    support_to_scalar,
)


def test_hyperbolic_parabolic_roundtrip():
    x = torch.linspace(-100.0, 100.0, 41)
    back = signed_parabolic(signed_hyperbolic(x))
    assert torch.allclose(back, x, atol=1e-2)


def test_scalar_to_support_2hot_sums_to_one():
    x = torch.tensor([-10.0, 0.0, 5.5, 29.9])
    dist = scalar_to_support(x, -30.0, 30.0, 61)
    assert dist.shape == (4, 61)
    assert torch.allclose(dist.sum(dim=-1), torch.ones(4), atol=1e-5)
    # 2-hot: at most 2 non-zero bins per row
    nonzero = (dist > 1e-6).sum(dim=-1)
    assert (nonzero <= 2).all()


def test_support_to_scalar_inverse():
    # Perfect-distribution: value at a bin produces back the same scalar
    # (after signed_hyperbolic/parabolic round-trip).
    xs = torch.tensor([-5.0, 0.0, 3.0, 9.0])
    xs_transformed = signed_hyperbolic(xs)
    dist = scalar_to_support(xs_transformed, -30.0, 30.0, 61)
    # Build logits from the distribution by taking log.
    logits = torch.log(dist + 1e-12)
    recovered = support_to_scalar(logits, -30.0, 30.0, 61)
    assert torch.allclose(recovered, xs, atol=0.1)


def test_cross_entropy_on_support_nonnegative():
    pred = torch.randn(8, 61)
    target = torch.randn(8) * 10.0
    loss = cross_entropy_on_support(pred, target, -30.0, 30.0, 61)
    assert loss.shape == (8,)
    assert (loss >= 0).all()
