"""Golden-equivalence tests for the batched inference server.

These pin the contract that `RemoteNetwork` (worker side) returns numerically
the same thing as calling the underlying `MuZeroNet` directly. They guard the
self-play speed refactors (removing np<->tensor ping-pong, payload dtype,
batch-size bucketing): any of those must not change the numbers a worker sees.
"""
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from src.muzero.mcts import MCTS
from src.muzero.networks import MuZeroNet
from src.selfplay.inference_server import (
    INITIAL,
    RECURRENT,
    InferenceServer,
    RemoteNetwork,
)

CFG_MODEL = {
    "input_channels": 4,
    "input_spatial": 96,
    "hidden_channels": 32,
    "hidden_spatial": 6,
    "num_actions": 12,
    "value_support": (-30.0, 30.0, 61),
    "reward_support": (-10.0, 10.0, 21),
    "rep_blocks": (1, 1, 1, 1),
    "dyn_blocks": 2,
    "pred_blocks": 1,
}


def _reference_net():
    net = MuZeroNet(
        input_channels=CFG_MODEL["input_channels"],
        input_spatial=CFG_MODEL["input_spatial"],
        hidden_channels=CFG_MODEL["hidden_channels"],
        hidden_spatial=CFG_MODEL["hidden_spatial"],
        num_actions=CFG_MODEL["num_actions"],
        value_support=CFG_MODEL["value_support"],
        reward_support=CFG_MODEL["reward_support"],
        rep_blocks=CFG_MODEL["rep_blocks"],
        dyn_blocks=CFG_MODEL["dyn_blocks"],
        pred_blocks=CFG_MODEL["pred_blocks"],
    )
    net.eval()
    return net


class _ServerHandle:
    def __init__(self, net, remote, server):
        self.net, self.remote, self.server = net, remote, server


@pytest.fixture
def make_server():
    """Factory: build a server+remote for a given wire dtype, auto-stopped."""
    handles = []

    def _make(wire_dtype="float32"):
        torch.manual_seed(0)
        net = _reference_net()
        ctx = mp.get_context("spawn")
        request_queue = ctx.Queue(maxsize=64)
        worker_conn, server_conn = ctx.Pipe(duplex=False)
        server = InferenceServer(
            cfg_model=CFG_MODEL,
            request_queue=request_queue,
            reply_conns=[server_conn],
            device=torch.device("cpu"),
            max_batch=8,
            max_wait_ms=5.0,
            use_amp=False,
            wire_dtype=wire_dtype,
        )
        server._apply_state_dict(net.state_dict())
        server.start()
        remote = RemoteNetwork(
            worker_id=0,
            request_queue=request_queue,
            reply_conn=worker_conn,
            wire_dtype=wire_dtype,
        )
        h = _ServerHandle(net, remote, server)
        handles.append(h)
        return h

    yield _make
    for h in handles:
        h.server.stop()


@pytest.mark.parametrize("wire_dtype,tol", [("float32", 1e-4), ("float16", 3e-3)])
def test_initial_inference_matches_direct_net(make_server, wire_dtype, tol):
    h = make_server(wire_dtype)
    net, remote = h.net, h.remote
    obs = torch.rand(1, 4, 96, 96)

    h_ref, pl_ref, v_ref = net.initial_inference(obs)
    h_rem, pl_rem, v_rem = remote.initial_inference(obs)

    assert np.asarray(h_rem).shape == tuple(h_ref.shape)
    assert pl_rem.shape == pl_ref.shape
    assert v_rem.shape == v_ref.shape == (1,)
    np.testing.assert_allclose(np.asarray(h_rem, dtype=np.float32), h_ref.numpy(), atol=tol)
    np.testing.assert_allclose(pl_rem.numpy(), pl_ref.numpy(), atol=1e-4)
    np.testing.assert_allclose(v_rem.numpy(), v_ref.numpy(), atol=1e-4)


@pytest.mark.parametrize("wire_dtype,tol", [("float32", 1e-4), ("float16", 3e-3)])
def test_recurrent_inference_matches_direct_net(make_server, wire_dtype, tol):
    h = make_server(wire_dtype)
    net, remote = h.net, h.remote
    obs = torch.rand(1, 4, 96, 96)
    h_ref, _, _ = net.initial_inference(obs)
    action = torch.tensor([3], dtype=torch.long)

    hn_ref, r_ref, pl_ref, v_ref = net.recurrent_inference(h_ref, action)
    # Feed the remote path the numpy hidden state it produces itself.
    hn_rem, r_rem, pl_rem, v_rem = remote.recurrent_inference(h_ref.numpy(), action)

    assert np.asarray(hn_rem).shape == tuple(hn_ref.shape)
    assert r_rem.shape == v_rem.shape == (1,)
    np.testing.assert_allclose(np.asarray(hn_rem, dtype=np.float32), hn_ref.numpy(), atol=tol)
    np.testing.assert_allclose(r_rem.numpy(), r_ref.numpy(), atol=1e-4)
    np.testing.assert_allclose(pl_rem.numpy(), pl_ref.numpy(), atol=1e-4)
    np.testing.assert_allclose(v_rem.numpy(), v_ref.numpy(), atol=1e-4)


def test_mcts_runs_end_to_end_through_server(make_server):
    """MCTS treats the hidden state opaquely; a full search through the remote
    network must still produce a valid visit-count policy."""
    remote = make_server("float32").remote
    mcts = MCTS(discount=0.99, num_simulations=12, root_dirichlet_alpha=0.0, device="cpu")
    obs = np.random.rand(4, 96, 96).astype(np.float32)
    action, pi, q = mcts.run(obs, remote, temperature=1.0, deterministic=False)
    assert 0 <= action < CFG_MODEL["num_actions"]
    assert pi.shape == (CFG_MODEL["num_actions"],)
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)
    assert pi.min() >= 0.0


def _unstarted_server(pad_batches):
    """Build (server, net, pipe) without starting the serve thread so
    `_run_batch` can be driven synchronously from the test."""
    torch.manual_seed(0)
    net = _reference_net()
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue(maxsize=64)
    worker_conn, server_conn = ctx.Pipe(duplex=False)
    server = InferenceServer(
        cfg_model=CFG_MODEL,
        request_queue=request_queue,
        reply_conns=[server_conn],
        device=torch.device("cpu"),
        max_batch=8,
        max_wait_ms=5.0,
        use_amp=False,
        pad_batches=pad_batches,
    )
    server._apply_state_dict(net.state_dict())
    return server, net, worker_conn


def test_bucket_size_rounds_to_pow2_capped_at_max_batch():
    server, _, _ = _unstarted_server(pad_batches=True)  # max_batch=8
    assert [server._bucket_size(n) for n in (1, 2, 3, 5, 7, 8)] == [1, 2, 4, 8, 8, 8]
    # Cap: next pow2 of 5 is 8 but max_batch=6 -> clamp to 6 (still >= n).
    server.max_batch = 6
    assert server._bucket_size(5) == 6
    off, _, _ = _unstarted_server(pad_batches=False)
    assert [off._bucket_size(n) for n in (1, 3, 5)] == [1, 3, 5]  # no-op when off


def test_padded_batch_is_numerically_identical_to_unpadded():
    """A mixed 3-initial / 2-recurrent batch (sizes 3 and 2 -> buckets 4 and 2)
    must yield the same replies with padding on vs off, up to float32 conv
    accumulation-order rounding (the real rows are mathematically unchanged;
    batch-dim only alters FP reduction order, ~1e-7)."""
    np.random.seed(1)
    obs = [np.random.rand(4, 96, 96).astype(np.float32) for _ in range(3)]
    # Seed hidden states from the reference net so they are in-distribution.
    base_net = _reference_net()
    with torch.no_grad():
        h0, _, _ = base_net.initial_inference(torch.from_numpy(obs[0]).unsqueeze(0))
    h_payloads = [
        (h0[0].numpy().astype(np.float32), a) for a in (2, 7)
    ]
    batch = [(0, INITIAL, o) for o in obs] + [(0, RECURRENT, hp) for hp in h_payloads]

    def run(pad):
        server, _, conn = _unstarted_server(pad_batches=pad)
        server._run_batch(list(batch))
        return [conn.recv() for _ in range(len(batch))]

    off, on = run(False), run(True)
    for ro, rn in zip(off, on):
        # Replies are tuples mixing arrays and python floats.
        for a, b in zip(ro, rn):
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-5
            )
