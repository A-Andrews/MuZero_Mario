import torch

from src.muzero.networks import MuZeroNet


def _small_net():
    return MuZeroNet(
        input_channels=4,
        input_spatial=96,
        hidden_channels=32,
        hidden_spatial=6,
        num_actions=12,
        value_support=(-30.0, 30.0, 61),
        reward_support=(-10.0, 10.0, 21),
        rep_blocks=(1, 1, 1, 1),
        dyn_blocks=2,
        pred_blocks=1,
    )


def test_representation_shape():
    net = _small_net()
    obs = torch.rand(2, 4, 96, 96)
    h = net.representation(obs)
    assert h.shape == (2, 32, 6, 6)
    # Min-max normalised into [0, 1]
    assert h.min().item() >= -1e-5
    assert h.max().item() <= 1.0 + 1e-5


def test_initial_step_shapes():
    net = _small_net()
    obs = torch.rand(3, 4, 96, 96)
    h, pi, v = net.initial_step(obs)
    assert h.shape == (3, 32, 6, 6)
    assert pi.shape == (3, 12)
    assert v.shape == (3, 61)


def test_recurrent_step_shapes():
    net = _small_net()
    obs = torch.rand(3, 4, 96, 96)
    h, _, _ = net.initial_step(obs)
    a = torch.randint(0, 12, (3,))
    h2, r, pi, v = net.recurrent_step(h, a)
    assert h2.shape == (3, 32, 6, 6)
    assert r.shape == (3, 21)
    assert pi.shape == (3, 12)
    assert v.shape == (3, 61)


def test_backward_through_unroll():
    net = _small_net()
    obs = torch.rand(2, 4, 96, 96)
    h, pi, v = net.initial_step(obs)
    loss = pi.pow(2).mean() + v.pow(2).mean()
    for _ in range(3):
        a = torch.randint(0, 12, (2,))
        h, r, pi, v = net.recurrent_step(h, a)
        loss = loss + r.pow(2).mean() + pi.pow(2).mean() + v.pow(2).mean()
    loss.backward()
    # Some parameter has gradient (not all zeros)
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_initial_inference_returns_scalar_value():
    net = _small_net()
    net.eval()
    obs = torch.rand(1, 4, 96, 96)
    h, pi_logits, v = net.initial_inference(obs)
    assert v.shape == (1,)
    assert pi_logits.shape == (1, 12)


def test_project_shapes_and_gradients():
    net = _small_net()
    obs = torch.rand(4, 4, 96, 96)
    h = net.representation(obs)
    proj_target = net.project(h, with_prediction=False)
    proj_online = net.project(h, with_prediction=True)
    assert proj_target.shape == proj_online.shape
    assert proj_target.shape[0] == 4
    # SimSiam consistency loss backprops through the online branch.
    loss = -torch.nn.functional.cosine_similarity(
        proj_online, proj_target.detach(), dim=-1
    ).mean()
    loss.backward()
    pred_grads = [
        p.grad for p in net.projection_net.predictor.parameters() if p.grad is not None
    ]
    assert any(g.abs().sum().item() > 0 for g in pred_grads)
