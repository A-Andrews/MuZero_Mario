"""MuZero-Mario training entrypoint."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

# Ensure `src` is importable no matter where Hydra runs us from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.logs.wandb_logger import WandbLogger
from src.muzero.buffer import TrajectoryBuffer
from src.muzero.muzero import MuzeroLearner
from src.muzero.networks import MuZeroNet
from src.selfplay.coordinator import SelfPlayCoordinator


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(spec: str) -> torch.device:
    if spec.startswith("cuda") and not torch.cuda.is_available():
        print(f"[train] requested {spec} but CUDA unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(spec)


def _build_network(model_cfg) -> MuZeroNet:
    return MuZeroNet(
        input_channels=model_cfg.input_channels,
        input_spatial=model_cfg.input_spatial,
        hidden_channels=model_cfg.hidden_channels,
        hidden_spatial=model_cfg.hidden_spatial,
        num_actions=model_cfg.num_actions,
        value_support=tuple(model_cfg.value_support),
        reward_support=tuple(model_cfg.reward_support),
        rep_blocks=tuple(model_cfg.rep_blocks),
        dyn_blocks=model_cfg.dyn_blocks,
        pred_blocks=model_cfg.pred_blocks,
    )


@hydra.main(config_path="../conf", config_name="muzero", version_base=None)
def main(cfg: DictConfig):
    # Hydra changes the cwd to the run dir; anchor int_path to it early.
    out_dir = Path(os.getcwd())
    print(f"[train] run dir: {out_dir}")

    # Limit CPU thread pool so single-threaded PyTorch ops don't spin up
    # dozens of competing threads — this is critical on HPC login nodes.
    learner_threads = int(cfg.learner_torch_threads)
    torch.set_num_threads(learner_threads)
    print(f"[train] torch threads: {torch.get_num_threads()}", flush=True)

    _seed_everything(int(cfg.seed))
    device = _pick_device(str(cfg.device))
    print(f"[train] device: {device}")

    # Resolve int_path relative to the original repo root (Hydra cwd-jumped us).
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    env_cfg["int_path"] = str(Path(env_cfg["int_path"]).resolve())
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["env"] = env_cfg  # ensure resolved int_path propagates

    # --- wandb ----------------------------------------------------------------
    wandb_logger = WandbLogger(
        project=cfg_dict["wandb"]["project"],
        entity=cfg_dict["wandb"].get("entity"),
        name=None,
        config=cfg_dict,
        mode=cfg_dict["wandb"].get("mode", "online"),
        dir=str(out_dir),
    )

    # --- networks + optimizer -------------------------------------------------
    online_net = _build_network(cfg.model).to(device)

    optimizer = torch.optim.Adam(
        online_net.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.training.lr_decay_steps),
        gamma=float(cfg.training.lr_decay_factor),
    )

    # --- buffer ---------------------------------------------------------------
    buffer = TrajectoryBuffer(
        capacity_transitions=int(cfg.buffer.capacity_transitions),
        unroll_K=int(cfg.muzero.unroll_K),
        num_actions=int(cfg.model.num_actions),
        priority_alpha=float(cfg.buffer.priority_alpha),
        priority_beta_start=float(cfg.buffer.priority_beta_start),
        priority_beta_end=float(cfg.buffer.priority_beta_end),
        priority_beta_anneal_steps=int(cfg.buffer.priority_beta_anneal_steps),
        eps_priority=float(cfg.buffer.eps_priority),
    )

    # --- self-play ------------------------------------------------------------
    coordinator = SelfPlayCoordinator(
        cfg=cfg_dict,
        num_workers=int(cfg.selfplay.num_workers),
        levels=list(cfg.env.levels),
    )

    # --- trainer --------------------------------------------------------------
    trainer = MuzeroLearner(
        cfg=cfg_dict,
        network=online_net,
        optimizer=optimizer,
        scheduler=scheduler,
        buffer=buffer,
        coordinator=coordinator,
        device=device,
        out_dir=out_dir,
        wandb_logger=wandb_logger,
    )

    start_step = 0
    if cfg.resume.path is not None:
        start_step = trainer.load(cfg.resume.path)
        print(f"[train] resumed at training_step={start_step}")

    print("[train] trainer created, broadcasting weights...", flush=True)
    # Send initial (or resumed) weights to workers before training begins.
    coordinator.broadcast_weights(online_net.state_dict())
    coordinator.set_train_step(start_step)
    print("[train] weights broadcast done, starting training loop", flush=True)

    try:
        trainer.training_loop(start_step=start_step)
    finally:
        coordinator.stop()
        wandb_logger.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
