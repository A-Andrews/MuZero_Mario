"""MuZero Learner: owns the online network, optimizer, and training loop.
Adapts `_update` from Muzero-Hanoi for categorical-support losses and a
K-step unroll over image observations.

Self-play happens elsewhere (SelfPlayCoordinator); this class just pulls
trajectories from the coordinator's queue and trains.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.checkpoint import (
    load_checkpoint,
    restore_rng,
    rotate_checkpoints,
    save_checkpoint,
)
from src.logs.video import save_video
from src.muzero.buffer import TrajectoryBuffer
from src.muzero.networks import MuZeroNet
from src.muzero.transforms import cross_entropy_on_support, support_to_scalar
from src.selfplay.coordinator import SelfPlayCoordinator
from src.selfplay.replay_eval import run_replay_rollout


class MuzeroLearner:
    def __init__(
        self,
        cfg: Dict[str, Any],
        network: MuZeroNet,
        optimizer: torch.optim.Optimizer,
        scheduler,
        buffer: TrajectoryBuffer,
        coordinator: SelfPlayCoordinator,
        device: torch.device,
        out_dir: Path,
        wandb_logger,
    ):
        self.cfg = cfg
        self.net = network
        self.opt = optimizer
        self.scheduler = scheduler
        self.buffer = buffer
        self.coord = coordinator
        self.device = device
        self.out_dir = Path(out_dir)
        self.wandb = wandb_logger

        self.training_step = 0
        self.env_step = 0

        tr = cfg["training"]
        self.min_replay = int(tr["min_replay_transitions"])
        self.batch_size = int(tr["batch_size"])
        self.grad_clip = float(tr["grad_clip"])
        self.weight_broadcast_every = int(tr["weight_broadcast_every"])
        self.save_every = int(tr["save_every_train_steps"])
        self.log_every = int(tr["log_every_train_steps"])
        self.total_env_steps = int(tr["total_env_steps"])

        m = cfg["muzero"]
        self.K = int(m["unroll_K"])
        self.discount = float(m["discount"])
        self.loss_w_v = float(m["loss_weight_value"])
        self.loss_w_r = float(m["loss_weight_reward"])
        self.loss_w_p = float(m["loss_weight_policy"])
        self.value_support = (float(m["value_support_min"]), float(m["value_support_max"]), int(m["value_support_size"]))
        self.reward_support = (float(m["reward_support_min"]), float(m["reward_support_max"]), int(m["reward_support_size"]))

        self.ckpt_dir = self.out_dir / "checkpoints"
        self.video_dir = self.out_dir / "videos"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self._recent_losses = deque(maxlen=100)

    # -------------------------------------------------------------------------

    def load(self, ckpt_path):
        state = load_checkpoint(ckpt_path, map_location=self.device)
        self.net.load_state_dict(state["online"], strict=True)
        self.opt.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.training_step = int(state.get("training_step", 0))
        self.env_step = int(state.get("env_step", 0))
        if state.get("rng") is not None:
            restore_rng(state["rng"])
        return self.training_step

    # -- main loop ------------------------------------------------------------

    def training_loop(self, start_step: int = 0):
        self.training_step = start_step
        last_log_time = time.time()
        last_traj_count = 0

        while self.env_step < self.total_env_steps:
            # 1) Drain trajectories + status from workers
            for traj in self.coord.drain_trajectories(max_items=32):
                self.env_step += int(traj.length)
                self.buffer.add(traj)
            for status in self.coord.drain_status(max_items=256):
                self.wandb.log(
                    {
                        f"selfplay/episode_return/{status['level']}": status["episode_return"],
                        f"selfplay/episode_length/{status['level']}": status["episode_length"],
                        f"selfplay/final_x_pos/{status['level']}": status["final_x_pos"],
                        f"selfplay/mcts_root_q_mean/{status['level']}": status["mcts_root_q_mean"],
                    },
                    step=self.training_step,
                )

            # 2) Gradient step if warm enough
            if self.buffer.size_transitions() < self.min_replay:
                time.sleep(0.05)
                continue

            loss_scalars = self._train_step()
            self.training_step += 1
            self._recent_losses.append(loss_scalars)

            # 3) Weight broadcast / log / checkpoint cadences
            if self.weight_broadcast_every > 0 and self.training_step % self.weight_broadcast_every == 0:
                self.coord.broadcast_weights(self.net.state_dict())
                self.coord.set_train_step(self.training_step)

            if self.log_every > 0 and self.training_step % self.log_every == 0:
                now = time.time()
                traj_count = self.buffer.size_trajectories()
                rate = (traj_count - last_traj_count) / max(1e-6, (now - last_log_time))
                self._log_training(rate)
                last_log_time = now
                last_traj_count = traj_count

            if self.save_every > 0 and self.training_step % self.save_every == 0:
                self._checkpoint_and_replay()

            if self.scheduler is not None:
                self.scheduler.step()

    # -- one gradient step ----------------------------------------------------

    def _train_step(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.training_step)
        obs = torch.from_numpy(batch["obs"]).to(self.device)                        # (B, C, H, W)
        actions = torch.from_numpy(batch["actions"]).to(self.device)                # (B, K)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)                # (B, K)
        policies = torch.from_numpy(batch["policies"]).to(self.device)              # (B, K+1, A)
        returns = torch.from_numpy(batch["returns"]).to(self.device)                # (B, K+1)
        is_w = torch.from_numpy(batch["is_weights"]).to(self.device)                # (B,)

        # Initial step: representation + prediction
        h, policy_logits, value_logits = self.net.initial_step(obs)
        v_pred_initial = support_to_scalar(value_logits, *self.value_support).detach()

        # Step 0 contributes value + policy at full weight (no reward at initial step).
        value_loss_init = cross_entropy_on_support(
            value_logits, returns[:, 0], *self.value_support
        )
        policy_loss_init = -(policies[:, 0] * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1)

        # Recurrent steps are scaled by 1/K so total loss scale is independent
        # of the unroll length (DeepMind MuZero, Appendix B).
        value_loss_recur = torch.zeros_like(value_loss_init)
        reward_loss_recur = torch.zeros_like(value_loss_init)
        policy_loss_recur = torch.zeros_like(value_loss_init)

        for k in range(self.K):
            h, reward_logits, policy_logits, value_logits = self.net.recurrent_step(h, actions[:, k])
            # Gradient scaling 0.5x on the dynamics hidden output
            h.register_hook(lambda grad: grad * 0.5)

            reward_loss_recur = reward_loss_recur + cross_entropy_on_support(
                reward_logits, rewards[:, k], *self.reward_support
            )
            value_loss_recur = value_loss_recur + cross_entropy_on_support(
                value_logits, returns[:, k + 1], *self.value_support
            )
            policy_loss_recur = policy_loss_recur + -(
                policies[:, k + 1] * F.log_softmax(policy_logits, dim=-1)
            ).sum(dim=-1)

        recur_scale = 1.0 / max(1, self.K)
        value_loss_total = value_loss_init + recur_scale * value_loss_recur
        reward_loss_total = recur_scale * reward_loss_recur
        policy_loss_total = policy_loss_init + recur_scale * policy_loss_recur

        loss_per_sample = (
            self.loss_w_v * value_loss_total
            + self.loss_w_r * reward_loss_total
            + self.loss_w_p * policy_loss_total
        )
        loss = (loss_per_sample * is_w).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip).item()
        self.opt.step()

        # Priority update: use |v_pred_initial - return_at_step_0|
        with torch.no_grad():
            target_return_0 = returns[:, 0]
            new_priorities = (v_pred_initial - target_return_0).abs().cpu().numpy()
        self.buffer.update_priorities(batch["sample_locations"], new_priorities)

        return {
            "value_loss": float(value_loss_total.mean().item()),
            "reward_loss": float(reward_loss_total.mean().item()),
            "policy_loss": float(policy_loss_total.mean().item()),
            "total_loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "is_weight_mean": float(is_w.mean().item()),
        }

    # -- housekeeping ---------------------------------------------------------

    def _log_training(self, traj_rate):
        if not self._recent_losses:
            return
        agg = defaultdict(list)
        for d in self._recent_losses:
            for k, v in d.items():
                agg[k].append(v)
        metrics = {f"train/{k}": float(np.mean(v)) for k, v in agg.items()}
        metrics["train/buffer_transitions"] = self.buffer.size_transitions()
        metrics["train/buffer_trajectories"] = self.buffer.size_trajectories()
        metrics["train/env_step"] = self.env_step
        metrics["train/lr"] = self.opt.param_groups[0]["lr"]
        metrics["train/trajectories_per_sec"] = float(traj_rate)
        self.wandb.log(metrics, step=self.training_step)

    def _checkpoint_and_replay(self):
        ckpt_path = self.ckpt_dir / f"step_{self.training_step}.pt"
        save_checkpoint(
            path=ckpt_path,
            online_state_dict=self.net.state_dict(),
            optimizer_state=self.opt.state_dict(),
            scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
            training_step=self.training_step,
            env_step=self.env_step,
            cfg_snapshot=self.cfg,
        )
        latest = self.ckpt_dir / "latest.pt"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(ckpt_path.name)
        except OSError:
            pass
        rotate_checkpoints(self.ckpt_dir, keep=10)

        if not self.cfg["wandb"].get("log_videos_every_ckpt", True):
            return

        video_step_dir = self.video_dir / f"step_{self.training_step}"
        video_step_dir.mkdir(parents=True, exist_ok=True)

        env_cfg = self.cfg["env"]
        model_cfg = self.cfg["model"]
        mcts_cfg = self.cfg["mcts"]
        pad_to = int(model_cfg["input_spatial"]) if env_cfg["pad_to_input_spatial"] else None

        frame_skip = int(env_cfg["frame_skip"])
        # We save one RGB frame per env.step(); env.step internally advances
        # `frame_skip` emulator frames at native 60 Hz, so playback fps is
        # 60 / frame_skip.
        video_fps = max(1, 60 // frame_skip)
        for level in env_cfg["levels"]:
            try:
                frames, total_return, n_steps = run_replay_rollout(
                    level=level,
                    int_path=env_cfg["int_path"],
                    network=self.net,
                    device=self.device,
                    num_simulations=int(mcts_cfg["num_simulations"]),
                    discount=float(self.cfg["muzero"]["discount"]),
                    pb_c_base=float(mcts_cfg["pb_c_base"]),
                    pb_c_init=float(mcts_cfg["pb_c_init"]),
                    n_frame_stack=int(env_cfg["n_frame_stack"]),
                    frame_skip=frame_skip,
                    pad_to=pad_to,
                    max_steps=int(self.cfg["selfplay"]["max_trajectory_length"]),
                    seed=int(self.cfg["seed"]) + 777,
                )
                mp4_path = video_step_dir / f"{level}.mp4"
                save_video(mp4_path, frames, fps=video_fps)
                self.wandb.log_video(
                    key=f"replay/{level}",
                    path=mp4_path,
                    fps=video_fps,
                    step=self.training_step,
                    extra={
                        f"replay/{level}_return": total_return,
                        f"replay/{level}_length": n_steps,
                    },
                )
            except Exception as e:
                self.wandb.log(
                    {f"replay/{level}_error": 1.0}, step=self.training_step
                )
                print(f"[replay] {level} failed: {e}")
        self.net.train()
