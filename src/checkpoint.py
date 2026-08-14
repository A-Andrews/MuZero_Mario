"""Atomic checkpoint save/load."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def save_checkpoint(
    path,
    online_state_dict: Dict[str, torch.Tensor],
    optimizer_state: Dict[str, Any],
    scheduler_state: Dict[str, Any] = None,
    training_step: int = 0,
    env_step: int = 0,
    cfg_snapshot: Dict[str, Any] = None,
):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "online": online_state_dict,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "training_step": training_step,
        "env_step": env_step,
        "cfg_snapshot": cfg_snapshot,
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        # A failed write (ENOSPC/EDQUOT, SIGTERM mid-save, ...) must not leave
        # a partial .tmp behind — on a quota'd filesystem the stale tmps
        # themselves eat the budget.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_checkpoint(path, map_location="cpu") -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=False)


def restore_rng(rng_state: Dict[str, Any]):
    cpu_state = rng_state["torch_cpu"]
    if isinstance(cpu_state, torch.Tensor):
        cpu_state = cpu_state.cpu()
    torch.set_rng_state(cpu_state)
    if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in rng_state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])


def rotate_checkpoints(ckpt_dir, keep: int = 10):
    """Delete the oldest `step_*.pt` files beyond `keep`, sorted by step number."""
    ckpt_dir = Path(ckpt_dir)
    ckpts = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for p in ckpts[:-keep]:
        try:
            p.unlink()
        except OSError:
            pass
