"""Self-play coordinator. Spawns workers and shuttles trajectories + weights."""
from __future__ import annotations

import pickle
from typing import List

import torch
import torch.multiprocessing as mp

from src.selfplay.worker import selfplay_worker


class SelfPlayCoordinator:
    def __init__(self, cfg: dict, num_workers: int, levels: List[str]):
        self.cfg = cfg
        self.num_workers = num_workers
        self.levels = list(levels)
        self.ctx = mp.get_context("spawn")
        self.traj_queue = self.ctx.Queue(maxsize=int(cfg["selfplay"]["max_queue_size"]))
        self.status_queue = self.ctx.Queue()
        self.train_step = self.ctx.Value("i", 0)
        self._weight_conns = []
        self._processes = []

        cfg_pkl = pickle.dumps(cfg)
        for i in range(num_workers):
            level = self.levels[i % len(self.levels)]
            parent_conn, child_conn = self.ctx.Pipe()
            p = self.ctx.Process(
                target=selfplay_worker,
                args=(
                    i,
                    level,
                    cfg_pkl,
                    child_conn,
                    self.traj_queue,
                    self.status_queue,
                    self.train_step,
                ),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._weight_conns.append(parent_conn)
            self._processes.append(p)

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
        """Send a CPU-tensor state_dict (pickleable through Pipe) to every worker."""
        cpu_sd = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        for conn in self._weight_conns:
            try:
                conn.send(cpu_sd)
            except Exception:
                pass

    def stop(self):
        for conn in self._weight_conns:
            try:
                conn.send(None)  # shutdown sentinel
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
