"""MuZero Learner: owns the online network, optimizer, and training loop.
Adapts `_update` from Muzero-Hanoi for categorical-support losses and a
K-step unroll over image observations.

Self-play happens elsewhere (SelfPlayCoordinator); this class just pulls
trajectories from the coordinator's queue and trains.
"""
from __future__ import annotations

import json
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
from src.muzero.human_data import HumanEvalSet
from src.muzero.human_baseline import (
    HumanLevelStats,
    compare_metrics,
    compute_level_stats,
    load_action_eval_sets,
)
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
                # Convert every ndarray in the sample generically (the key set
                # depends on buffer options, e.g. reanalyze); non-arrays such
                # as sample_locations pass through untouched.
                tensors = {}
                for k, v in batch_np.items():
                    if isinstance(v, np.ndarray):
                        t = torch.from_numpy(v)
                        if self._pin:
                            t = t.pin_memory()
                        tensors[k] = t
                    else:
                        tensors[k] = v
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
        coordinator: Optional[SelfPlayCoordinator],
        device: torch.device,
        out_dir: Path,
        wandb_logger,
        human_eval: Optional[HumanEvalSet] = None,
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
        # Step of the last on-disk checkpoint; save_final_checkpoint() skips
        # the write when the loop happened to stop exactly on a save boundary.
        self._last_saved_step = -1

        tr = cfg["training"]
        self.min_replay = int(tr["min_replay_transitions"])
        self.batch_size = int(tr["batch_size"])
        self.grad_clip = float(tr["grad_clip"])
        self.weight_broadcast_every = int(tr["weight_broadcast_every"])
        self.save_every = int(tr["save_every_train_steps"])
        self.checkpoint_keep = int(tr.get("checkpoint_keep", 10))
        self.replay_every = self._resolve_replay_every(
            tr.get("replay_every_train_steps", 0), self.save_every
        )
        self.log_every = int(tr["log_every_train_steps"])
        self.total_env_steps = int(tr["total_env_steps"])
        # Cap on learner pace: at most `max_train_per_env_step` gradient steps
        # per collected env step (<=0 disables). Keeps the replay ratio sane
        # and leaves GPU headroom for the inference server when self-play is
        # the bottleneck.
        self.train_ratio = float(tr.get("max_train_per_env_step", 0.0))
        self.replay_max_steps = int(
            tr.get("replay_max_steps", 0)
            or cfg["selfplay"]["max_trajectory_length"]
        )

        m = cfg["muzero"]
        self.K = int(m["unroll_K"])
        self.discount = float(m["discount"])
        self.loss_w_v = float(m["loss_weight_value"])
        self.loss_w_r = float(m["loss_weight_reward"])
        self.loss_w_p = float(m["loss_weight_policy"])
        self.loss_w_c = float(m.get("loss_weight_consistency", 0.0))
        self.value_support = (float(m["value_support_min"]), float(m["value_support_max"]), int(m["value_support_size"]))
        self.reward_support = (float(m["reward_support_min"]), float(m["reward_support_max"]), int(m["reward_support_size"]))
        # Value reanalyze: recompute n-step value targets at sample time with
        # a lagged copy of the online network instead of the root values
        # frozen at collection time. The target net is (re)synced from the
        # online net every `training.target_update_every` train steps.
        self.reanalyze = bool(m.get("reanalyze", False))
        self.target_update_every = max(1, int(tr.get("target_update_every", 200)))
        self._target_net: Optional[MuZeroNet] = None
        self._target_synced_at = -(10 ** 9)

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
        # Rolling completion flags over the last 100 self-play episodes
        # (across all levels) — the headline "is Mario finishing levels" metric.
        self._recent_completions = deque(maxlen=100)
        # Best-checkpoint tracking (rolling completion rate). best.pt is
        # outside the step_*.pt rotation window, so performance peaks survive
        # even when training later degrades. best.json carries the rate across
        # resumed chain legs so a worse leg never overwrites a better net.
        self._best_completion_rate = 0.0
        self._last_best_save_step = -(10**9)
        best_meta = self.ckpt_dir / "best.json"
        if best_meta.exists():
            try:
                self._best_completion_rate = float(
                    json.loads(best_meta.read_text()).get("completion_rate_100ep", 0.0)
                )
            except (ValueError, OSError):
                pass

        # Completion-gated root-Dirichlet anneal. The origin lives in a sidecar
        # rather than in the checkpoint so enabling the gate doesn't invalidate
        # existing checkpoints, and so a chained leg that resumes after the
        # unlock doesn't re-arm the gate and jump eps back to its start value.
        self._eps_gate_on_completion = bool(
            (cfg.get("mcts", {}) or {}).get(
                "root_exploration_eps_gate_on_completion", False
            )
        )
        self._first_completion_step = -1
        fc_meta = self.ckpt_dir / "first_completion.json"
        if fc_meta.exists():
            try:
                self._first_completion_step = int(
                    json.loads(fc_meta.read_text()).get("rl_train_step", -1)
                )
            except (ValueError, OSError):
                pass

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

        # --- imitation: human-data pretrain + constant batch mix -------------
        im = cfg.get("imitation") or {}
        self._imitation_enabled = bool(im.get("enabled", False))
        self._pretrain_steps = (
            int(im.get("pretrain_steps", 0)) if self._imitation_enabled else 0
        )
        # Pretrain steps share the global training_step counter (LR warmup,
        # checkpoints, resume all unchanged), so the RL-phase pacing — the
        # replay-ratio gate and the worker temperature schedule — must see
        # RL-only steps: training_step - _rl_offset.
        self._rl_offset = self._pretrain_steps
        self._human_eval = human_eval
        self._bc_eval_obs: Optional[torch.Tensor] = None
        self._bc_eval_actions: Optional[torch.Tensor] = None

        # Human comparison at checkpoint-replay time. Loaded lazily inside the
        # background replay thread (the action-eval obs are hundreds of MB and
        # take seconds to decompress) so learner startup is untouched.
        self._human_cmp_cfg = cfg.get("human_compare") or {}
        self._human_stats: Optional[Dict[str, HumanLevelStats]] = None
        self._human_action_eval: Dict[str, HumanEvalSet] = {}
        self._human_cmp_tensors: Dict[str, Any] = {}

        # Prefetch is primed once on the first post-warmup step; every
        # _train_step then requests the next batch itself.
        self._prefetch_primed = False
        # Replay runs on a private network copy in a background thread so it
        # never blocks the learner and never toggles the training net's
        # BatchNorm into eval().
        self._replay_net: Optional[MuZeroNet] = None
        self._replay_thread: Optional[threading.Thread] = None

        # Normally the coordinator is attached later via set_coordinator()
        # (after any imitation pretrain); this covers a caller that passes one
        # straight to the constructor, since the sidecar is read above.
        self._push_anneal_origin()

    # -------------------------------------------------------------------------

    def set_coordinator(self, coordinator: SelfPlayCoordinator):
        """Attach the self-play coordinator (constructed after pretrain so
        workers start with the BC-pretrained weights)."""
        self.coord = coordinator
        self._push_anneal_origin()

    def _push_anneal_origin(self) -> None:
        """Seed the workers' shared anneal origin from the resumed sidecar."""
        if self.coord is None or self._first_completion_step < 0:
            return
        try:
            self.coord.set_anneal_origin(self._first_completion_step)
        except AttributeError:
            # Loud on purpose: the failure mode is silent otherwise — the run
            # would anneal exactly like an ungated one and only look wrong 15M
            # env steps later, which is the mistake this gate exists to stop.
            print(
                "[train] WARNING: coordinator has no set_anneal_origin(); the "
                "completion-gated eps anneal is NOT active for this run",
                flush=True,
            )

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

    # -- imitation pretrain -----------------------------------------------------

    def pretrain(self, start_step: int = 0) -> int:
        """Supervised pretraining on 100% human batches before self-play starts.

        Reuses `_train_step` unchanged: policy loss against the stored one-hot
        human actions is behavioral cloning, and value/reward/consistency
        losses all apply. Steps count into the global `training_step`, so a
        resumed run past `pretrain_steps` skips this entirely.

        Returns the training step to hand to `training_loop`.
        """
        if not self._imitation_enabled or start_step >= self._pretrain_steps:
            return start_step
        assert hasattr(self.buffer, "force_mix_ratio"), (
            "imitation pretrain requires the learner to hold a MixedBuffer"
        )
        self.training_step = start_step
        print(
            f"[pretrain] human-data pretraining: steps {start_step}..{self._pretrain_steps} "
            f"({self.buffer.human_transitions()} human transitions)",
            flush=True,
        )
        # Reanalyze bootstraps value targets from a lagged copy of the online
        # net — during pretrain that net is (near-)random, so its bootstraps
        # are noise. Train against the stored human n-step returns instead.
        # The buffers keep emitting the reanalyze keys throughout, so batch
        # shapes never change across the phase boundary.
        saved_reanalyze = self.reanalyze
        self.reanalyze = False
        self.buffer.force_mix_ratio(1.0)

        self._prefetcher.request(self.training_step)
        while self.training_step < self._pretrain_steps:
            loss_scalars = self._train_step()
            self.training_step += 1
            self._recent_losses.append(loss_scalars)
            if self.log_every > 0 and self.training_step % self.log_every == 0:
                self._log_pretrain()
            if self.save_every > 0 and self.training_step % self.save_every == 0:
                self._save_checkpoint()
            if self.scheduler is not None:
                self.scheduler.step()

        # _train_step eagerly requested one more (human-only, reanalyze-less)
        # batch; consume and discard it so the flag flips below happen while
        # the prefetch thread is idle. _prefetch_primed stays False, so
        # training_loop re-primes with post-pretrain settings.
        self._prefetcher.get()
        self.reanalyze = saved_reanalyze
        self.buffer.force_mix_ratio(None)  # back to constant / schedule
        self._recent_losses.clear()
        self._save_checkpoint()  # phase boundary (may not align with save_every)
        print(f"[pretrain] done at training_step={self.training_step}", flush=True)
        return self.training_step

    # -- main loop ------------------------------------------------------------

    def training_loop(self, start_step: int = 0):
        assert self.coord is not None, "set_coordinator() before training_loop()"
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
                completed = bool(status.get("completed", False))
                self._level_sampler.record_episode(
                    lv, status["episode_length"], completed=completed
                )
                if lv in self._ac_episodes_seen:
                    self._ac_episodes_seen[lv] += 1
                self._recent_completions.append(1.0 if completed else 0.0)
                completion_rate = float(np.mean(self._recent_completions))
                if completed and self._first_completion_step < 0:
                    self._record_first_completion(lv)
                self.wandb.log(
                    {
                        f"selfplay/episode_return/{lv}": status["episode_return"],
                        f"selfplay/episode_length/{lv}": status["episode_length"],
                        f"selfplay/final_x_pos/{lv}": status["final_x_pos"],
                        f"selfplay/mcts_root_q_mean/{lv}": status["mcts_root_q_mean"],
                        # Search quality. Entropies are nats; compare against
                        # ln(12) = 2.485 for the 12-action Mario set — a prior
                        # entropy sitting near that means the policy head is
                        # near-uniform and root noise is doing the work.
                        f"selfplay/mcts_prior_entropy/{lv}": status.get("mcts_prior_entropy_mean", 0.0),
                        f"selfplay/mcts_visit_entropy/{lv}": status.get("mcts_visit_entropy_mean", 0.0),
                        f"selfplay/mcts_visit_max_frac/{lv}": status.get("mcts_visit_max_frac_mean", 0.0),
                        # Effective root-Dirichlet fraction (T2.1 anneal). Read
                        # the completion rate against this: a rate that holds up
                        # as eps decays is the policy head standing on its own.
                        "selfplay/root_exploration_eps": status.get(
                            "root_exploration_eps", 0.0
                        ),
                        f"selfplay/completed/{lv}": 1.0 if completed else 0.0,
                        "selfplay/completion_rate_100ep": completion_rate,
                    },
                    step=self.training_step,
                )
                # New rolling-completion high → refresh best.pt. Full window
                # only (early small-sample rates overshoot), +0.01 margin and
                # a step gap so a slow climb doesn't checkpoint every episode.
                if (
                    len(self._recent_completions) == self._recent_completions.maxlen
                    and completion_rate >= self._best_completion_rate + 0.01
                    and self.training_step - self._last_best_save_step >= 2000
                ):
                    self._save_best_checkpoint(completion_rate)

            # 2) Gradient step if warm enough
            if self.buffer.size_transitions() < self.min_replay:
                time.sleep(0.05)
                continue

            # Learner pace cap: don't run ahead of self-play data collection.
            # Pretrain steps don't count — without the offset a pretrained run
            # would sit at training_step >= env_step * ratio forever.
            rl_steps = self.training_step - self._rl_offset
            if self.train_ratio > 0.0 and rl_steps >= self.env_step * self.train_ratio:
                time.sleep(0.02)
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
                # Workers' temperature schedule is thresholded on train steps;
                # report RL-only steps so pretrain doesn't skip the schedule.
                self.coord.set_train_step(self.training_step - self._rl_offset)

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

        # Budget reached: persist the tail since the last save_every boundary
        # and mark the run complete so chained resume legs stand down.
        print(
            f"[train] env-step budget reached "
            f"({self.env_step}/{self.total_env_steps}) at "
            f"training_step={self.training_step}",
            flush=True,
        )
        self.save_final_checkpoint()
        self.write_complete_sentinel()

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

    def _maybe_refresh_target(self):
        """(Re)sync the reanalyze target net from the online net on cadence."""
        if not self.reanalyze:
            return
        if (
            self._target_net is not None
            and self.training_step - self._target_synced_at < self.target_update_every
        ):
            return
        if self._target_net is None:
            self._target_net = self._build_replay_net()
        self._target_net.load_state_dict(self.net.state_dict())
        self._target_net.eval()
        self._target_synced_at = self.training_step

    def _train_step(self) -> Dict[str, float]:
        self._maybe_refresh_target()
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
        if self.loss_w_c > 0.0:
            next_obs = batch["next_obs"].to(self.device, non_blocking=pin)
            next_obs_mask = batch["next_obs_mask"].to(self.device, non_blocking=pin)
        sample_locations = batch["sample_locations"]

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.use_amp
            else _NullCtx()
        )

        with amp_ctx:
            # Value reanalyze: replace the collection-time returns with fresh
            # n-step targets bootstrapped from the (lagged) target network:
            #   z_k = reward_window_k + factor_k * V_target(obs at bootstrap)
            # reward_window / factor come precomputed from the buffer with the
            # right terminal/truncation semantics; factor is 0 past terminals.
            if self.reanalyze:
                value_obs = batch["value_obs"].to(self.device, non_blocking=pin)
                value_obs_factor = batch["value_obs_factor"].to(self.device, non_blocking=pin)
                reward_window = batch["reward_window"].to(self.device, non_blocking=pin)
                B = value_obs.shape[0]
                flat_vobs = value_obs.reshape(B * (self.K + 1), *value_obs.shape[2:])
                flat_vobs = flat_vobs.float().div_(255.0)
                if self.use_amp:
                    flat_vobs = flat_vobs.contiguous(memory_format=torch.channels_last)
                with torch.no_grad():
                    h_boot = self._target_net.representation(flat_vobs)
                    _, v_boot_logits = self._target_net.prediction(h_boot)
                    v_boot = support_to_scalar(
                        v_boot_logits.float(), *self.value_support
                    ).reshape(B, self.K + 1)
                returns = reward_window + value_obs_factor * v_boot

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

            unrolled_h = [] if self.loss_w_c > 0.0 else None
            for k in range(self.K):
                h, reward_logits, policy_logits, value_logits = self.net.recurrent_step(h, actions[:, k])
                # Gradient scaling 0.5x on the dynamics hidden output
                h.register_hook(lambda grad: grad * 0.5)
                if unrolled_h is not None:
                    unrolled_h.append(h)

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

            # EfficientZero-style self-supervised consistency: pull the
            # dynamics-unrolled hidden state at step k toward the (stop-grad)
            # representation of the real observation at t+k.
            consistency_loss_total = torch.zeros_like(value_loss_init)
            if self.loss_w_c > 0.0:
                B = next_obs_mask.shape[0]
                flat_next = next_obs.reshape(B * self.K, *next_obs.shape[2:])
                flat_next = flat_next.float().div_(255.0)
                if self.use_amp:
                    flat_next = flat_next.contiguous(memory_format=torch.channels_last)
                with torch.no_grad():
                    h_target = self.net.representation(flat_next)
                    proj_target = self.net.project(h_target, with_prediction=False)
                h_dyn = torch.stack(unrolled_h, dim=1).reshape(
                    B * self.K, *unrolled_h[0].shape[1:]
                )
                proj_dyn = self.net.project(h_dyn, with_prediction=True)
                cos = F.cosine_similarity(
                    proj_dyn.float(), proj_target.detach().float(), dim=-1
                ).reshape(B, self.K)
                consistency_loss_total = (
                    -(cos * next_obs_mask).sum(dim=-1) * recur_scale
                )

            loss_per_sample = (
                self.loss_w_v * value_loss_total
                + self.loss_w_r * reward_loss_total
                + self.loss_w_p * policy_loss_total
                + self.loss_w_c * consistency_loss_total
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
            "consistency_loss": float(consistency_loss_total.mean().item()),
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
        completion = self._level_sampler.completion_rates()
        metrics = {}
        for lv in levels:
            metrics[f"autocurriculum/weight/{lv}"] = float(weights[lv])
            metrics[f"autocurriculum/completion_rate/{lv}"] = float(completion[lv])
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
        if self._imitation_enabled:
            metrics["train/human_buffer_transitions"] = self.buffer.human_transitions()
            metrics["train/human_frac"] = float(
                self.buffer.mix_ratio_at(self.training_step)
                if hasattr(self.buffer, "mix_ratio_at")
                else getattr(self.buffer, "mix_ratio", 0.0)
            )
            metrics.update(self._bc_eval_metrics())
        self.wandb.log(metrics, step=self.training_step)

    def _log_pretrain(self):
        if not self._recent_losses:
            return
        agg = defaultdict(list)
        for d in self._recent_losses:
            for k, v in d.items():
                agg[k].append(v)
        metrics = {f"pretrain/{k}": float(np.mean(v)) for k, v in agg.items()}
        metrics["pretrain/lr"] = self.opt.param_groups[0]["lr"]
        metrics.update(self._bc_eval_metrics())
        self.wandb.log(metrics, step=self.training_step)

    def _bc_eval_metrics(self) -> Dict[str, float]:
        """Top-1 accuracy + cross-entropy of the policy head against held-out
        human actions — the cheap "human-likeness" signal, logged in both
        phases. Runs synchronously on the learner thread, so flipping the net
        to eval() (keeps BatchNorm stats clean of human-only batches) is safe.
        """
        if self._human_eval is None or len(self._human_eval) == 0:
            return {}
        if self._bc_eval_obs is None:
            self._bc_eval_obs = torch.from_numpy(self._human_eval.obs)
            self._bc_eval_actions = torch.from_numpy(self._human_eval.actions)
        n = int(self._bc_eval_actions.shape[0])
        correct = 0
        ce_sum = 0.0
        self.net.eval()
        with torch.no_grad():
            for i in range(0, n, 512):
                obs = self._bc_eval_obs[i:i + 512].to(self.device).float().div_(255.0)
                if self.use_amp:
                    obs = obs.contiguous(memory_format=torch.channels_last)
                acts = self._bc_eval_actions[i:i + 512].to(self.device)
                _, policy_logits, _ = self.net.initial_step(obs)
                logp = F.log_softmax(policy_logits.float(), dim=-1)
                ce_sum += float(-logp.gather(1, acts.unsqueeze(1)).sum().item())
                correct += int((logp.argmax(dim=-1) == acts).sum().item())
        self.net.train()
        return {
            "imitation/bc_accuracy": correct / n,
            "imitation/bc_cross_entropy": ce_sum / n,
        }

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

    def save_final_checkpoint(self):
        """Checkpoint the current step unless it was already saved.

        Called at the natural end of training and on SIGTERM/interrupt so the
        steps trained since the last `save_every` boundary are never lost —
        without this, every resumed chain leg re-trained the same tail and
        wandb dropped the re-logged (non-monotonic) steps.
        """
        if self.training_step != self._last_saved_step:
            self._save_checkpoint()

    def write_complete_sentinel(self):
        """Mark the run as having reached its env-step budget.

        `train_muzero.py` and the SLURM chain scripts read this to skip /
        cancel now-pointless resume legs. Never written on interrupt — only
        when `env_step >= total_env_steps`.
        """
        sentinel = self.out_dir / "TRAINING_COMPLETE"
        sentinel.write_text(
            json.dumps(
                {
                    "env_step": int(self.env_step),
                    "total_env_steps": int(self.total_env_steps),
                    "training_step": int(self.training_step),
                }
            )
            + "\n"
        )
        print(f"[train] wrote {sentinel}", flush=True)

    def _record_first_completion(self, level: str) -> None:
        """Unlock the completion-gated anneal on the run's first completion.

        Recorded even when the gate is off — the sidecar is a cheap provenance
        record of when a run escaped, and it means flipping the gate on for a
        resumed leg starts from the right origin instead of re-arming.
        """
        rl_step = max(0, self.training_step - self._rl_offset)
        self._first_completion_step = rl_step
        self._push_anneal_origin()
        try:
            (self.ckpt_dir / "first_completion.json").write_text(
                json.dumps(
                    {
                        "rl_train_step": rl_step,
                        "training_step": self.training_step,
                        "env_step": self.env_step,
                        "level": level,
                    }
                )
            )
        except OSError as e:
            # Same reasoning as _save_checkpoint: never kill a run over a
            # sidecar write. The in-memory origin still holds for this leg.
            print(
                f"[train] WARNING: could not persist first_completion.json: {e}",
                flush=True,
            )
        if self._eps_gate_on_completion:
            print(
                f"[train] first completion on {level} at RL train step "
                f"{rl_step} — root-Dirichlet anneal unlocked",
                flush=True,
            )

    def _save_best_checkpoint(self, completion_rate: float):
        """Overwrite best.pt on a new rolling-completion-rate high.

        step_*.pt checkpoints rotate (keep=10), so without this the
        best-performing net is lost whenever training later degrades — the
        level1-1-diag-v1 0.72-completion peak rotated away exactly that way.
        wandb logs the event under train/best_completion_rate.
        """
        prev_best = self._best_completion_rate
        self._best_completion_rate = completion_rate
        self._last_best_save_step = self.training_step
        try:
            save_checkpoint(
                path=self.ckpt_dir / "best.pt",
                online_state_dict=self.net.state_dict(),
                optimizer_state=self.opt.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
                training_step=self.training_step,
                env_step=self.env_step,
                cfg_snapshot=self.cfg,
            )
        except OSError as e:
            # Roll the recorded best back so the next new-high (or even the
            # same rate again) retries the save instead of best.pt silently
            # freezing at an older net.
            self._best_completion_rate = prev_best
            print(
                f"[train] WARNING: best.pt save failed at step "
                f"{self.training_step}: {e} — continuing without saving",
                flush=True,
            )
            return
        (self.ckpt_dir / "best.json").write_text(
            json.dumps(
                {
                    "completion_rate_100ep": completion_rate,
                    "training_step": int(self.training_step),
                    "env_step": int(self.env_step),
                }
            )
            + "\n"
        )
        self.wandb.log(
            {"train/best_completion_rate": completion_rate}, step=self.training_step
        )
        print(
            f"[train] new best completion_rate_100ep={completion_rate:.2f} "
            f"at step {self.training_step} → best.pt",
            flush=True,
        )

    def _save_checkpoint(self):
        ckpt_path = self.ckpt_dir / f"step_{self.training_step}.pt"
        try:
            save_checkpoint(
                path=ckpt_path,
                online_state_dict=self.net.state_dict(),
                optimizer_state=self.opt.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
                training_step=self.training_step,
                env_step=self.env_step,
                cfg_snapshot=self.cfg,
            )
        except OSError as e:
            # ENOSPC/EDQUOT etc. A missed checkpoint is recoverable (the next
            # save boundary retries, and _last_saved_step stays stale so
            # save_final_checkpoint retries too); killing the run is not —
            # the 2026-08-12 T6 fleet lost 12x2h15 of training to a home-quota
            # EDQUOT raised here.
            print(
                f"[train] WARNING: checkpoint save failed at step "
                f"{self.training_step}: {e} — continuing without saving",
                flush=True,
            )
            return
        self._last_saved_step = self.training_step
        latest = self.ckpt_dir / "latest.pt"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(ckpt_path.name)
        except OSError:
            pass
        rotate_checkpoints(self.ckpt_dir, keep=self.checkpoint_keep)

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

    # ------------------------------------------------------------------
    # Human comparison (runs inside the background replay thread)
    # ------------------------------------------------------------------
    def _ensure_human_baseline(self, levels) -> None:
        """One-shot load of the per-level human reference + action-eval sets."""
        if self._human_stats is not None:
            return
        c = self._human_cmp_cfg
        data_dir = c.get("data_dir", "outputs/human_trajectories")
        subjects = list(c["subjects"]) if c.get("subjects") else None
        try:
            self._human_stats = compute_level_stats(
                data_dir, levels, subjects=subjects,
                use_cache=bool(c.get("cache_stats", True)),
            )
        except Exception as e:
            print(f"[human-compare] level stats unavailable: {e}")
            self._human_stats = {}
            return

        im = self.cfg.get("imitation") or {}
        # When imitation trained on this corpus, score agreement only on the
        # files it held out. The holdout is a seeded shuffle of the *filtered*
        # file list, so reproducing it means mirroring the imitation loader's
        # level and subject filters exactly — the replay level list and this
        # block's own `subjects` would shuffle a different list and silently
        # hand back files the net trained on.
        holdout_only = self._imitation_enabled
        split_levels = list(levels)
        split_subjects = subjects
        holdout_fraction = 0.0
        if holdout_only:
            lv = im.get("levels", "match_env")
            if lv == "match_env":
                split_levels = list(self.cfg["env"]["levels"])
            elif lv in ("all", None):
                split_levels = None
            else:
                split_levels = list(lv)
            split_subjects = list(im["subjects"]) if im.get("subjects") else None
            holdout_fraction = float(im.get("holdout_fraction", 0.0))
            if subjects is not None and subjects != split_subjects:
                print(
                    f"[human-compare] ignoring human_compare.subjects={subjects}: with "
                    f"imitation on, the action-eval split must mirror "
                    f"imitation.subjects={split_subjects} to stay leak-free"
                )
        try:
            self._human_action_eval = load_action_eval_sets(
                data_dir,
                levels,
                subjects=split_subjects,
                max_transitions_per_level=int(c.get("action_eval_transitions_per_level", 1024)),
                holdout_only=holdout_only,
                holdout_fraction=holdout_fraction,
                split_levels=split_levels,
                seed=int(self.cfg.get("seed", 0)),
            )
        except Exception as e:
            print(f"[human-compare] action-eval unavailable: {e}")
            self._human_action_eval = {}

    def _human_action_agreement(self, level: str) -> Dict[str, float]:
        """Top-1 agreement + cross-entropy of the *replay* net against the human's
        actual button presses on that level's held-out states."""
        eval_set = self._human_action_eval.get(level)
        if eval_set is None or len(eval_set) == 0:
            return {}
        cached = self._human_cmp_tensors.get(level)
        if cached is None:
            cached = (
                torch.from_numpy(eval_set.obs),
                torch.from_numpy(eval_set.actions),
            )
            self._human_cmp_tensors[level] = cached
        obs_all, act_all = cached

        n = int(act_all.shape[0])
        correct, ce_sum = 0, 0.0
        with torch.no_grad():
            for i in range(0, n, 256):
                obs = obs_all[i:i + 256].to(self.device).float().div_(255.0)
                if self.device.type == "cuda":
                    obs = obs.contiguous(memory_format=torch.channels_last)
                acts = act_all[i:i + 256].to(self.device)
                _, policy_logits, _ = self._replay_net.initial_step(obs)
                logp = F.log_softmax(policy_logits.float(), dim=-1)
                ce_sum += float(-logp.gather(1, acts.unsqueeze(1)).sum().item())
                correct += int((logp.argmax(dim=-1) == acts).sum().item())
        return {
            f"compare/{level}_action_agreement": correct / n,
            f"compare/{level}_action_cross_entropy": ce_sum / n,
        }

    def _human_compare_metrics(
        self, level: str, completed: bool, n_steps: int, total_return: float
    ) -> Dict[str, float]:
        """Performance comparison (pure math, see `human_baseline.compare_metrics`)
        plus the behavioural action-agreement pass over held-out human states."""
        m = compare_metrics(
            level,
            completed=completed,
            n_steps=n_steps,
            total_return=total_return,
            stats=(self._human_stats or {}).get(level),
            completion_bonus=float(self.cfg["env"].get("completion_bonus", 0.0)),
        )
        m.update(self._human_action_agreement(level))
        return m

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
            if self._human_cmp_cfg.get("enabled", False):
                self._ensure_human_baseline(list(env_cfg["levels"]))
            for level in env_cfg["levels"]:
                try:
                    rollout_info: Dict[str, Any] = {}
                    frames, total_return, n_steps, completed = run_replay_rollout(
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
                        max_steps=self.replay_max_steps,
                        seed=int(self.cfg["seed"]) + 777,
                        bk2_path=video_step_dir / f"{level}.bk2",
                        leaf_batch=int(mcts_cfg.get("leaf_batch", 1)),
                        info_out=rollout_info,
                        # keeps replay/<level>_return on the same reward scale
                        # as selfplay/episode_return/<level>
                        completion_bonus=float(env_cfg.get("completion_bonus", 100.0)),
                    )
                    metrics = {
                        f"replay/{level}_return": total_return,
                        f"replay/{level}_length": n_steps,
                        f"replay/{level}_completed": 1.0 if completed else 0.0,
                        f"replay/{level}_final_x": float(rollout_info.get("final_x", 0)),
                        f"replay/{level}_timed_out": float(
                            rollout_info.get("timed_out", False)
                        ),
                    }
                    if self._human_cmp_cfg.get("enabled", False):
                        metrics.update(
                            self._human_compare_metrics(
                                level, completed, n_steps, total_return
                            )
                        )
                    # The mp4 is the *least* important product here, and its
                    # encoder is the flakiest step (ffmpeg dies under CPU
                    # contention, leaving a 0-byte file). Keep video failure
                    # from taking the scalars — including the human comparison
                    # — down with it. step is read live to keep wandb's step
                    # monotonic; the training thread has advanced since the
                    # weight snapshot.
                    mp4_path = video_step_dir / f"{level}.mp4"
                    try:
                        save_video(mp4_path, frames, fps=video_fps)
                        self.wandb.log_video(
                            key=f"replay/{level}",
                            path=mp4_path,
                            fps=video_fps,
                            step=self.training_step,
                            extra=metrics,
                        )
                    except Exception as e:
                        metrics[f"replay/{level}_video_error"] = 1.0
                        self.wandb.log(metrics, step=self.training_step)
                        print(f"[replay] {level} video failed (metrics kept): {e}")
                except Exception as e:
                    self.wandb.log(
                        {f"replay/{level}_error": 1.0}, step=self.training_step
                    )
                    print(f"[replay] {level} failed: {e}")
        except Exception as e:
            print(f"[replay] dispatch failed: {e}")
