"""Benchmark: per-worker local MCTS inference vs central batched inference server.

Simulates N concurrent self-play workers each running a fixed number of MCTS
decisions against a dummy (all-zero-obs) env. Reports wall-clock and
decisions/sec for both configurations.

Runs on CPU or GPU depending on `--device`. Threads are used to emulate
workers — acceptable here because PyTorch ops release the GIL during kernel
execution, so contention roughly matches the real multi-process worker setup
for the purpose of comparing the two inference topologies.

Usage:
    python scripts/benchmark_inference.py --workers 12 --decisions 20 \\
        --num-simulations 50 --device cuda
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.muzero.mcts import MCTS
from src.muzero.networks import MuZeroNet
from src.selfplay.inference_server import InferenceServer, RemoteNetwork


def build_model_cfg():
    # Matches conf/model/muzero_atari.yaml
    return dict(
        input_channels=4,
        input_spatial=96,
        hidden_channels=256,
        hidden_spatial=6,
        num_actions=12,
        value_support=[-300.0, 300.0, 601],
        reward_support=[-30.0, 30.0, 61],
        rep_blocks=[2, 3, 3, 3],
        dyn_blocks=15,
        pred_blocks=2,
    )


def build_net(cfg_model, device):
    net = MuZeroNet(
        input_channels=cfg_model["input_channels"],
        input_spatial=cfg_model["input_spatial"],
        hidden_channels=cfg_model["hidden_channels"],
        hidden_spatial=cfg_model["hidden_spatial"],
        num_actions=cfg_model["num_actions"],
        value_support=tuple(cfg_model["value_support"]),
        reward_support=tuple(cfg_model["reward_support"]),
        rep_blocks=tuple(cfg_model["rep_blocks"]),
        dyn_blocks=cfg_model["dyn_blocks"],
        pred_blocks=cfg_model["pred_blocks"],
    )
    return net.to(device).eval()


class _LocalNetAdapter:
    """Wraps a MuZeroNet so MCTS can call .initial_inference / .recurrent_inference."""

    def __init__(self, net):
        self.net = net

    def eval(self):
        self.net.eval()
        return self

    def initial_inference(self, obs):
        with torch.inference_mode():
            return self.net.initial_inference(obs)

    def recurrent_inference(self, h, a):
        with torch.inference_mode():
            return self.net.recurrent_inference(h, a)


def worker_loop_local(cfg_model, device, num_simulations, decisions, barrier, results, idx):
    net = build_net(cfg_model, device)
    adapter = _LocalNetAdapter(net)
    mcts = MCTS(discount=0.997, num_simulations=num_simulations,
                root_dirichlet_alpha=0.0, root_exploration_eps=0.0, device=device)
    obs = np.zeros((cfg_model["input_channels"], cfg_model["input_spatial"], cfg_model["input_spatial"]), dtype=np.float32)
    barrier.wait()
    t0 = time.time()
    for _ in range(decisions):
        mcts.run(obs, adapter, temperature=1.0, deterministic=False)
    results[idx] = time.time() - t0


def worker_loop_remote(request_q, reply_conn, worker_id, cfg_model, num_simulations, decisions, barrier, results, idx):
    rn = RemoteNetwork(worker_id=worker_id, request_queue=request_q, reply_conn=reply_conn)
    mcts = MCTS(discount=0.997, num_simulations=num_simulations,
                root_dirichlet_alpha=0.0, root_exploration_eps=0.0, device="cpu")
    obs = np.zeros((cfg_model["input_channels"], cfg_model["input_spatial"], cfg_model["input_spatial"]), dtype=np.float32)
    barrier.wait()
    t0 = time.time()
    for _ in range(decisions):
        mcts.run(obs, rn, temperature=1.0, deterministic=False)
    results[idx] = time.time() - t0


def bench_local(cfg_model, device, n_workers, decisions, num_simulations):
    results = [0.0] * n_workers
    barrier = threading.Barrier(n_workers + 1)
    threads = [
        threading.Thread(
            target=worker_loop_local,
            args=(cfg_model, device, num_simulations, decisions, barrier, results, i),
            daemon=True,
        )
        for i in range(n_workers)
    ]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.time()
    for t in threads:
        t.join()
    wall = time.time() - t0
    total_decisions = n_workers * decisions
    return wall, total_decisions, sum(results) / len(results)


def bench_remote(cfg_model, device, n_workers, decisions, num_simulations, max_batch, max_wait_ms, use_amp):
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    reply_pairs = [ctx.Pipe(duplex=False) for _ in range(n_workers)]  # (reader, writer)
    worker_readers = [r for (r, _w) in reply_pairs]
    server_writers = [w for (_r, w) in reply_pairs]

    server = InferenceServer(
        cfg_model=cfg_model,
        request_queue=req_q,
        reply_conns=server_writers,
        device=device,
        max_batch=max_batch,
        max_wait_ms=max_wait_ms,
        use_amp=use_amp,
    )
    server.start()

    results = [0.0] * n_workers
    barrier = threading.Barrier(n_workers + 1)
    threads = [
        threading.Thread(
            target=worker_loop_remote,
            args=(req_q, worker_readers[i], i, cfg_model, num_simulations, decisions, barrier, results, i),
            daemon=True,
        )
        for i in range(n_workers)
    ]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.time()
    for t in threads:
        t.join()
    wall = time.time() - t0
    server.stop()

    total_decisions = n_workers * decisions
    return wall, total_decisions, sum(results) / len(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--decisions", type=int, default=10)
    ap.add_argument("--num-simulations", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-batch", type=int, default=48)
    ap.add_argument("--max-wait-ms", type=float, default=1.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--skip-local", action="store_true",
                    help="Skip per-worker-local benchmark (it is very slow on CPU with full net).")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg_model = build_model_cfg()
    print(f"[bench] device={device} workers={args.workers} decisions={args.decisions} "
          f"sims={args.num_simulations} cuda_avail={torch.cuda.is_available()}", flush=True)

    if not args.skip_local:
        print("[bench] running LOCAL per-worker inference ...", flush=True)
        wall, n, avg = bench_local(cfg_model, device, args.workers, args.decisions, args.num_simulations)
        print(f"[bench] LOCAL   : wall={wall:.2f}s  total_decisions={n}  "
              f"throughput={n/wall:.2f} dec/s  avg_worker_time={avg:.2f}s", flush=True)

    print("[bench] running REMOTE batched inference server ...", flush=True)
    wall, n, avg = bench_remote(
        cfg_model, device, args.workers, args.decisions, args.num_simulations,
        max_batch=args.max_batch, max_wait_ms=args.max_wait_ms, use_amp=not args.no_amp,
    )
    print(f"[bench] REMOTE  : wall={wall:.2f}s  total_decisions={n}  "
          f"throughput={n/wall:.2f} dec/s  avg_worker_time={avg:.2f}s", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
