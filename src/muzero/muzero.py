"""MuZero Learner: owns the online network, optimizer, and training loop.
Adapts `_update` from Muzero-Hanoi for categorical-support losses and a
K-step unroll over image observations.

Self-play happens elsewhere (SelfPlayCoordinator); this class just pulls
trajectories from the coordinator's queue and trains.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional

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
from src.selfplay.autocurriculum import LevelSampler
from src.selfplay.coordinator import SelfPlayCoordinator
from src.selfplay.replay_eval import run_replay_rollout


class _NullCtx:
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False


class _BatchPrefetcher:
    """One-deep background batch sampler. Overlaps CPU buffer.sample with GPU step."""

    def __init__(self, buffer, batch_size, pin):
        self._buffer = buffer
        self._batch_size = batch_size
        self._pin = pin
        self._cv = threading.Condition()
        self._ready = None  # batch_dict_with_pinned_tensors
        self._error: Optional[BaseException] = None
        self._request_step = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request(self, train_step: int):
        with self._cv:
            self._request_step = train_step
            self._cv.notify_all()

    def get(self):
        with self._cv:
            while self._ready is None and self._error is None and not self._stop:
                self._cv.wait()
            if self._error is not None:
                err = self._error
                self._error = None
                raise err
            batch = self._ready
            self._ready = None
            self._cv.notify_all()
            return batch

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def _run(self):
        while True:
            with self._cv:
                while not self._stop and (
                    self._request_step is None
                    or self._ready is not None
                    or self._error is not None
                ):
                    self._cv.wait()
                if self._stop:
                    return
                step = self._request_step
                self._request_step = None
            try:
                batch_np = self._buffer.sample(self._batch_size, step)
                tensors = {
                    "obs": torch.from_numpy(batch_np["obs"]),
                    "actions": torch.from_numpy(batch_np["actions"]),
                    "rewards": torch.from_numpy(batch_np["rewards"]),
                    "policies": torch.from_numpy(batch_np["policies"]),
                    "returns": torch.from_numpy(batch_np["returns"]),
                    "is_weights": torch.from_numpy(batch_np["is_weights"]),
                }
                if self._pin:
                    for k in tensors:
                        tensors[k] = tensors[k].pin_memory()
                tensors["sample_locations"] = batch_np["sample_locations"]
            except BaseException as e:
                with self._cv:
                    self._error = e
                    self._cv.notify_all()
                continue
            with self._cv:
                self._ready = tensors
                self._cv.notify_all()


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
        self.replay_every = self._resolve_replay_every(
            tr.get("replay_every_train_steps", 0), self.save_every
        )
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

        self.use_amp = (device.type == "cuda")
        self.amp_dtype = torch.bfloat16
        self._prefetcher = _BatchPrefetcher(
            buffer=self.buffer,
            batch_size=self.batch_size,
            pin=(device.type == "cuda"),
        )

        self.ckpt_dir = self.out_dir / "checkpoints"
        self.video_dir = self.out_dir / "videos"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self._recent_losses = deque(maxlen=100)

        # --- autocurriculum: inverse-length per-level sampling ---------------
        ac = cfg.get("autocurriculum", {}) or {}
        self._ac_enabled = bool(ac.get("enabled", False)) and len(cfg["env"]["levels"]) > 1
        self._ac_update_every = int(ac.get("update_every_train_steps", 200))
        self._ac_warmup_per_level = int(ac.get("warmup_episodes_per_level", 3))
        self._level_sampler = LevelSampler(
            levels=list(cfg["env"]["levels"]),
            history_size=int(ac.get("history_size", 50)),
            exponent=float(ac.get("exponent", 1.0)),
            min_weight=float(ac.get("min_weight", 0.02)),
        )
        self._ac_episodes_seen = {lv: 0 for lv in cfg["env"]["levels"]}

        # Prefetch is primed once on the first post-warmup step; every
        # _train_step then requests the next batch itself.
        self._prefetch_primed = False
        # Replay runs on a private network copy in a background thread so it
        # never blocks the learner and never toggles the training net's
        # BatchNorm into eval().
        self._replay_net: Optional[MuZeroNet] = None
        self._replay_thread: Optional[threading.Thread] = None

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
                lv = status["level"]
                self._level_sampler.record_episode(lv, status["episode_length"])
                if lv in self._ac_episodes_seen:
                    self._ac_episodes_seen[lv] += 1
                self.wandb.log(
                    {
                        f"selfplay/episode_return/{lv}": status["episode_return"],
                        f"selfplay/episode_length/{lv}": status["episode_length"],
                        f"selfplay/final_x_pos/{lv}": status["final_x_pos"],
                        f"selfplay/mcts_root_q_mean/{lv}": status["mcts_root_q_mean"],
                    },
                    step=self.training_step,
                )

            # 2) Gradient step if warm enough
            if self.buffer.size_transitions() < self.min_replay:
                time.sleep(0.05)
                continue

            # Prime the prefetch exactly once (first post-warmup step). After
            # that, each _train_step eagerly requests the following batch, so
            # re-requesting here would just trigger a redundant buffer.sample.
            if not self._prefetch_primed:
                self._prefetcher.request(self.training_step)
                self._prefetch_primed = True
            loss_scalars = self._train_step()
            self.training_step += 1
            self._recent_losses.append(loss_scalars)

            # 3) Weight broadcast / log / checkpoint cadences
            if self.weight_broadcast_every > 0 and self.training_step % self.weight_broadcast_every == 0:
                self.coord.broadcast_weights(self.net.state_dict())
                self.coord.set_train_step(self.training_step)

            if (
                self._ac_enabled
                and self._ac_update_every > 0
                and self.training_step % self._ac_update_every == 0
            ):
                self._push_curriculum_weights()

            if self.log_every > 0 and self.training_step % self.log_every == 0:
                now = time.time()
                traj_count = self.buffer.size_trajectories()
                rate = (traj_count - last_traj_count) / max(1e-6, (now - last_log_time))
                self._log_training(rate)
                last_log_time = now
                last_traj_count = traj_count

            if self.save_every > 0 and self.training_step % self.save_every == 0:
                self._save_checkpoint()

            if self.replay_every > 0 and self.training_step % self.replay_every == 0:
                self._maybe_launch_replay()

            if self.scheduler is not None:
                self.scheduler.step()

    def close(self):
        try:
            self._prefetcher.stop()
        except Exception:
            pass
        if self._replay_thread is not None and self._replay_thread.is_alive():
            try:
                self._replay_thread.join(timeout=30)
            except Exception:
                pass

    # -- one gradient step ----------------------------------------------------

    def _train_step(self) -> Dict[str, float]:
        batch = self._prefetcher.get()
        # Immediately queue the next batch so CPU sampling overlaps GPU compute.
        self._prefetcher.request(self.training_step + 1)
        pin = (self.device.type == "cuda")
        obs = batch["obs"].to(self.device, non_blocking=pin)
        if self.use_amp:
            obs = obs.contiguous(memory_format=torch.channels_last)
        actions = batch["actions"].to(self.device, non_blocking=pin)
        rewards = batch["rewards"].to(self.device, non_blocking=pin)
        policies = batch["policies"].to(self.device, non_blocking=pin)
        returns = batch["returns"].to(self.device, non_blocking=pin)
        is_w = batch["is_weights"].to(self.device, non_blocking=pin)
        sample_locations = batch["sample_locations"]

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.use_amp
            else _NullCtx()
        )

        with amp_ctx:
            # Initial step: representation + prediction
            h, policy_logits, value_logits = self.net.initial_step(obs)
            # support_to_scalar runs in fp32 for a stable priority signal.
            v_pred_initial = support_to_scalar(
                value_logits.float(), *self.value_support
            ).detach()

            value_loss_init = cross_entropy_on_support(
                value_logits.float(), returns[:, 0], *self.value_support
            )
            policy_loss_init = -(policies[:, 0] * F.log_softmax(policy_logits.float(), dim=-1)).sum(dim=-1)

            value_loss_recur = torch.zeros_like(value_loss_init)
            reward_loss_recur = torch.zeros_like(value_loss_init)
            policy_loss_recur = torch.zeros_like(value_loss_init)

            for k in range(self.K):
                h, reward_logits, policy_logits, value_logits = self.net.recurrent_step(h, actions[:, k])
                # Gradient scaling 0.5x on the dynamics hidden output
                h.register_hook(lambda grad: grad * 0.5)

                reward_loss_recur = reward_loss_recur + cross_entropy_on_support(
                    reward_logits.float(), rewards[:, k], *self.reward_support
                )
                value_loss_recur = value_loss_recur + cross_entropy_on_support(
                    value_logits.float(), returns[:, k + 1], *self.value_support
                )
                policy_loss_recur = policy_loss_recur + -(
                    policies[:, k + 1] * F.log_softmax(policy_logits.float(), dim=-1)
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
        self.buffer.update_priorities(sample_locations, new_priorities)

        return {
            "value_loss": float(value_loss_total.mean().item()),
            "reward_loss": float(reward_loss_total.mean().item()),
            "policy_loss": float(policy_loss_total.mean().item()),
            "total_loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "is_weight_mean": float(is_w.mean().item()),
        }

    # -- housekeeping ---------------------------------------------------------

    def _push_curriculum_weights(self) -> None:
        """Recompute inverse-length weights and push them to workers.

        Stays uniform until every level has at least ``warmup_episodes_per_level``
        completed episodes; this avoids a "first level to finish wins all
        worker slots" pathology before there is any signal to act on.
        """
        ready = all(
            n >= self._ac_warmup_per_level for n in self._ac_episodes_seen.values()
        )
        levels = self._level_sampler.levels
        if ready:
            weights = self._level_sampler.compute_weights()
        else:
            uniform = 1.0 / len(levels)
            weights = {lv: uniform for lv in levels}
        self.coord.update_level_weights(weights)
        means = self._level_sampler.mean_lengths()
        metrics = {}
        for lv in levels:
            metrics[f"autocurriculum/weight/{lv}"] = float(weights[lv])
            m = means[lv]
            if m == m:  # not NaN
                metrics[f"autocurriculum/mean_length/{lv}"] = float(m)
        metrics["autocurriculum/active"] = 1.0 if ready else 0.0
        self.wandb.log(metrics, step=self.training_step)

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

    @staticmethod
    def _resolve_replay_every(replay_every_cfg, save_every: int) -> int:
        """Resolve the replay cadence.

        A replay rollout is a full greedy-MCTS episode on the GPU and contends
        with the learner, so it runs on its own (typically sparser) cadence
        rather than every checkpoint. A non-positive / unset config value falls
        back to ``save_every`` so existing configs keep their behavior.
        """
        replay_every = int(replay_every_cfg)
        if replay_every <= 0:
            return save_every
        return replay_every

    def _save_checkpoint(self):
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

    def _maybe_launch_replay(self):
        if not self.cfg["wandb"].get("log_videos_every_ckpt", True):
            return

        if self._replay_thread is not None and self._replay_thread.is_alive():
            print(
                f"[replay] previous replay still running; "
                f"skipping replay at step {self.training_step}"
            )
            return

        video_step_dir = self.video_dir / f"step_{self.training_step}"
        video_step_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot weights to CPU on the training thread so the background
        # replay sees a consistent set of weights and never races the
        # optimizer. Replay runs on a separate network instance, so the
        # training net is never switched to eval().
        cpu_sd = {
            k: v.detach().to("cpu", copy=True)
            for k, v in self.net.state_dict().items()
        }
        self._replay_thread = threading.Thread(
            target=self._run_replay,
            args=(cpu_sd, video_step_dir),
            name=f"replay-step-{self.training_step}",
            daemon=True,
        )
        self._replay_thread.start()

    def _build_replay_net(self) -> MuZeroNet:
        m = self.cfg["model"]
        net = MuZeroNet(
            input_channels=m["input_channels"],
            input_spatial=m["input_spatial"],
            hidden_channels=m["hidden_channels"],
            hidden_spatial=m["hidden_spatial"],
            num_actions=m["num_actions"],
            value_support=tuple(m["value_support"]),
            reward_support=tuple(m["reward_support"]),
            rep_blocks=tuple(m["rep_blocks"]),
            dyn_blocks=m["dyn_blocks"],
            pred_blocks=m["pred_blocks"],
        ).to(self.device)
        if self.device.type == "cuda":
            net = net.to(memory_format=torch.channels_last)
        return net

    def _run_replay(self, cpu_sd, video_step_dir):
        """Background replay: greedy MCTS rollouts on a private network copy.

        Runs off the training thread so the learner keeps stepping, and uses a
        dedicated `self._replay_net` (never the training net) so BatchNorm is
        never toggled into eval() under the learner.
        """
        try:
            if self._replay_net is None:
                self._replay_net = self._build_replay_net()
            self._replay_net.load_state_dict(
                {k: v.to(self.device) for k, v in cpu_sd.items()}, strict=True
            )
            self._replay_net.eval()

            env_cfg = self.cfg["env"]
            model_cfg = self.cfg["model"]
            mcts_cfg = self.cfg["mcts"]
            pad_to = (
                int(model_cfg["input_spatial"])
                if env_cfg["pad_to_input_spatial"]
                else None
            )
            frame_skip = int(env_cfg["frame_skip"])
            # One RGB frame per env.step(); env.step advances `frame_skip`
            # emulator frames at 60 Hz, so playback fps is 60 / frame_skip.
            video_fps = max(1, 60 // frame_skip)
            for level in env_cfg["levels"]:
                try:
                    frames, total_return, n_steps = run_replay_rollout(
                        level=level,
                        int_path=env_cfg["int_path"],
                        network=self._replay_net,
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
                        bk2_path=video_step_dir / f"{level}.bk2",
                    )
                    mp4_path = video_step_dir / f"{level}.mp4"
                    save_video(mp4_path, frames, fps=video_fps)
                    # step read live to keep wandb's step monotonic; the
                    # training thread has advanced since the snapshot.
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
        except Exception as e:
            print(f"[replay] dispatch failed: {e}")
