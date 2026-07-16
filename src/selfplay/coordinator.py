"""Self-play coordinator.

Spawns workers and hosts a central GPU-batched inference server thread.
Workers send MCTS inference requests through a shared mp.Queue and receive
replies on per-worker mp.Pipes. Weight updates are applied by the server in
place (no pickle through pipes).
"""
from __future__ import annotations

import pickle
from ctypes import c_double
from typing import Dict, List, Sequence

import torch
import torch.multiprocessing as mp

from src.selfplay.inference_server import InferenceServer
from src.selfplay.worker import selfplay_worker


class SelfPlayCoordinator:
    def __init__(
        self,
        cfg: dict,
        num_workers: int,
        levels: List[str],
        device: torch.device,
        initial_state_dict=None,
    ):
        self.cfg = cfg
        self.num_workers = num_workers
        self.levels = list(levels)
        self.device = device
        self.ctx = mp.get_context("spawn")
        self.traj_queue = self.ctx.Queue(maxsize=int(cfg["selfplay"]["max_queue_size"]))
        self.status_queue = self.ctx.Queue()
        self.train_step = self.ctx.Value("i", 0)
        self._processes = []

        # Shared sampling weights per level (uniform until the learner pushes
        # the first inverse-length update). Written by the learner via
        # update_level_weights(); read by workers at each episode boundary.
        n_levels = len(self.levels)
        uniform = 1.0 / max(1, n_levels)
        self.level_weights = self.ctx.Array(c_double, [uniform] * n_levels, lock=True)

        # --- inference plumbing -----------------------------------------------
        self.request_queue = self.ctx.Queue(maxsize=num_workers * 8)
        self._server_reply_conns = []
        worker_reply_conns = []
        for _ in range(num_workers):
            # Pipe(duplex=False) → (reader, writer). Server writes replies,
            # worker reads them.
            worker_conn, server_conn = self.ctx.Pipe(duplex=False)
            self._server_reply_conns.append(server_conn)
            worker_reply_conns.append(worker_conn)

        self.inference_server = InferenceServer(
            cfg_model=cfg["model"],
            request_queue=self.request_queue,
            reply_conns=self._server_reply_conns,
            device=device,
            max_batch=int(cfg.get("inference_server", {}).get("max_batch", num_workers * 4)),
            max_wait_ms=float(cfg.get("inference_server", {}).get("max_wait_ms", 1.0)),
            use_amp=bool(cfg.get("inference_server", {}).get("use_amp", True)),
            wire_dtype=str(cfg.get("inference_server", {}).get("wire_dtype", "float32")),
            pad_batches=bool(cfg.get("inference_server", {}).get("pad_batches", False)),
            compile_mode=str(cfg.get("inference_server", {}).get("compile", "off")),
        )
        if initial_state_dict is not None:
            # Pre-load learner weights so workers never see the server's
            # freshly-initialised random weights.
            self.inference_server._apply_state_dict(initial_state_dict)
        self.inference_server.start()

        # --- workers ---------------------------------------------------------
        cfg_pkl = pickle.dumps(cfg)
        autocurriculum_enabled = bool(
            (cfg.get("autocurriculum") or {}).get("enabled", False)
        ) and len(self.levels) > 1
        for i in range(num_workers):
            # When autocurriculum is on, the worker picks its level from the
            # shared weights at every episode boundary; the initial_level is
            # just where it starts before the first sample.
            initial_level = self.levels[i % len(self.levels)]
            p = self.ctx.Process(
                target=selfplay_worker,
                args=(
                    i,
                    initial_level,
                    list(self.levels),
                    self.level_weights,
                    autocurriculum_enabled,
                    cfg_pkl,
                    self.request_queue,
                    worker_reply_conns[i],
                    self.traj_queue,
                    self.status_queue,
                    self.train_step,
                ),
                daemon=True,
            )
            p.start()
            # Worker end stays with the child process; drop the parent-side
            # handle to that particular conn so only the child holds it.
            worker_reply_conns[i].close()
            self._processes.append(p)

    # -- public API -----------------------------------------------------------

    def set_train_step(self, step: int):
        self.train_step.value = int(step)

    def drain_trajectories(self, max_items: int = 64):
        out = []
        for _ in range(max_items):
            try:
                out.append(self.traj_queue.get_nowait())
            except Exception:
                break
        return out

    def drain_status(self, max_items: int = 256):
        out = []
        for _ in range(max_items):
            try:
                out.append(self.status_queue.get_nowait())
            except Exception:
                break
        return out

    def broadcast_weights(self, state_dict):
        """Push the latest weights into the inference server (no IPC)."""
        self.inference_server.update_weights(state_dict)

    def update_level_weights(self, weights: Dict[str, float]) -> None:
        """Write a new per-level sampling distribution into shared memory."""
        with self.level_weights.get_lock():
            for i, lv in enumerate(self.levels):
                w = float(weights.get(lv, 0.0))
                if w < 0.0 or w != w:  # NaN guard
                    w = 0.0
                self.level_weights[i] = w

    def stop(self):
        try:
            self.inference_server.stop()
        except Exception:
            pass
        # Belt-and-braces: if this process ever put anything on these queues,
        # never let interpreter exit block on flushing it. Workers stuck
        # mid-put get terminated below and can leave the queues' shared
        # write-locks held by a dead process, which would deadlock the
        # feeder-thread join that mp runs at exit.
        for q in (self.request_queue, self.traj_queue, self.status_queue):
            try:
                q.cancel_join_thread()
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        # Reap terminated workers so none linger as zombies into interpreter
        # shutdown.
        for p in self._processes:
            if p.is_alive():
                p.join(timeout=5)
