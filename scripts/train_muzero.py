"""MuZero-Mario training entrypoint."""
from __future__ import annotations

import json
import math
import random
import signal
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Ensure `src` is importable no matter where Hydra runs us from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.logs.wandb_logger import WandbLogger
from src.muzero.buffer import MixedBuffer, TrajectoryBuffer
from src.muzero.human_data import load_human_buffer
from src.muzero.muzero import MuzeroLearner
from src.muzero.networks import MuZeroNet
from src.selfplay.coordinator import SelfPlayCoordinator


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_fast_math():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


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
    # With version_base=None, Hydra ≥1.2 does NOT chdir into the run dir, so
    # os.getcwd() is the repo root. Pull the run dir directly from HydraConfig
    # so checkpoints + videos land next to the Hydra log under outputs/.
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run dir: {out_dir}")

    # Limit CPU thread pool so single-threaded PyTorch ops don't spin up
    # dozens of competing threads — this is critical on HPC login nodes.
    learner_threads = int(cfg.learner_torch_threads)
    torch.set_num_threads(learner_threads)
    print(f"[train] torch threads: {torch.get_num_threads()}", flush=True)

    _seed_everything(int(cfg.seed))
    device = _pick_device(str(cfg.device))
    if device.type == "cuda":
        _enable_fast_math()
    print(f"[train] device: {device}")

    # Resolve int_path relative to the original repo root (Hydra cwd-jumped us).
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    env_cfg["int_path"] = str(Path(env_cfg["int_path"]).resolve())
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["env"] = env_cfg  # ensure resolved int_path propagates

    # --- resume-from-latest auto-detection -----------------------------------
    # If cfg.run_name is set, treat out_dir/checkpoints/latest.pt as the
    # canonical resume point: the same SLURM script can be resubmitted with
    # --dependency=afterany and will pick up where the previous job stopped.
    # Explicit cfg.resume.path always wins so users can still pin a specific
    # checkpoint.
    run_name = cfg_dict.get("run_name")
    auto_resume_path = None
    if cfg.resume.path is None and run_name:
        candidate = out_dir / "checkpoints" / "latest.pt"
        if candidate.exists():
            auto_resume_path = candidate
            print(f"[train] auto-resume: found {candidate}")

    # --- completed-run fast exit ----------------------------------------------
    # The learner writes TRAINING_COMPLETE when env_step reaches
    # training.total_env_steps. Chained SLURM resume legs land here; bail out
    # before wandb / human-data / coordinator setup unless the budget was
    # raised past what the sentinel records.
    sentinel = out_dir / "TRAINING_COMPLETE"
    if sentinel.exists():
        try:
            done = json.loads(sentinel.read_text())
        except (ValueError, OSError):
            done = {}
        if int(cfg.training.total_env_steps) <= int(done.get("env_step", 0)):
            print(
                f"[train] {sentinel} present: run already reached "
                f"env_step={done.get('env_step')} >= "
                f"total_env_steps={int(cfg.training.total_env_steps)}; nothing to do. "
                f"(Raise training.total_env_steps or remove the sentinel to continue.)",
                flush=True,
            )
            return

    # --- wandb ----------------------------------------------------------------
    w = cfg_dict["wandb"]
    # When run_name is set, use it as a stable wandb id so re-submitted SLURM
    # jobs append to the original run rather than spawning a fresh one. Without
    # run_name, fall back to wandb's auto-generated id (original behavior).
    wandb_id = run_name if run_name else None
    wandb_resume = "allow" if run_name else None
    wandb_logger = WandbLogger(
        project=w["project"],
        entity=w.get("entity"),
        name=w.get("name") or run_name,
        config=cfg_dict,
        mode=w.get("mode", "online"),
        dir=str(out_dir),
        tags=w.get("tags"),
        notes=w.get("notes"),
        group=w.get("group"),
        job_type=w.get("job_type"),
        id=wandb_id,
        resume=wandb_resume,
    )

    # --- networks + optimizer -------------------------------------------------
    online_net = _build_network(cfg.model).to(device)
    if device.type == "cuda":
        online_net = online_net.to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(
        online_net.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    # Warmup + cosine decay to a floor, then constant at the floor. The
    # previous StepLR (x0.1 every lr_decay_steps) silently drove the LR to
    # ~1e-9 on long runs, killing learning entirely.
    lr_max = float(cfg.training.lr)
    lr_min = float(cfg.training.get("lr_min", lr_max * 0.1))
    warmup = max(1, int(cfg.training.get("lr_warmup_steps", 1000)))
    decay_steps = max(1, int(cfg.training.lr_decay_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        frac = min(1.0, (step - warmup) / decay_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))
        return (lr_min + (lr_max - lr_min) * cos) / lr_max

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

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
        reanalyze=bool(cfg.muzero.get("reanalyze", False)),
        n_step=int(cfg.muzero.n_step),
        discount=float(cfg.muzero.discount),
    )

    # --- imitation: pinned human buffer + sample-time mixing -------------------
    im_cfg = cfg_dict.get("imitation") or {}
    human_eval = None
    pretrain_steps = 0
    if im_cfg.get("enabled", False):
        pretrain_steps = int(im_cfg.get("pretrain_steps", 0))
        lv = im_cfg.get("levels", "match_env")
        if lv == "match_env":
            im_levels = list(cfg.env.levels)
        elif lv == "all" or lv is None:
            im_levels = None
        else:
            im_levels = list(lv)
        human_buffer, human_eval = load_human_buffer(
            im_cfg.get("data_dir", "outputs/human_trajectories"),
            unroll_K=int(cfg.muzero.unroll_K),
            num_actions=int(cfg.model.num_actions),
            levels=im_levels,
            subjects=list(im_cfg["subjects"]) if im_cfg.get("subjects") else None,
            completed_only=bool(im_cfg.get("completed_only", False)),
            max_transitions=int(im_cfg.get("max_transitions", 0)),
            holdout_fraction=float(im_cfg.get("holdout_fraction", 0.0)),
            holdout_max_transitions=int(im_cfg.get("holdout_max_transitions", 4096)),
            buffer_kwargs=dict(
                priority_alpha=float(cfg.buffer.priority_alpha),
                priority_beta_start=float(cfg.buffer.priority_beta_start),
                priority_beta_end=float(cfg.buffer.priority_beta_end),
                priority_beta_anneal_steps=int(cfg.buffer.priority_beta_anneal_steps),
                eps_priority=float(cfg.buffer.eps_priority),
                reanalyze=bool(cfg.muzero.get("reanalyze", False)),
                n_step=int(cfg.muzero.n_step),
                discount=float(cfg.muzero.discount),
            ),
            seed=int(cfg.seed),
        )
        # Schedule thresholds are written in RL-phase training steps; shift
        # them past the pretrain phase (which shares the global step counter)
        # so the anneal starts when self-play does.
        mix_schedule = im_cfg.get("mix_ratio_schedule")
        if mix_schedule:
            mix_schedule = [
                [int(s) + pretrain_steps, float(r)] for s, r in mix_schedule
            ]
        buffer = MixedBuffer(
            buffer,
            human_buffer,
            float(im_cfg.get("mix_ratio", 0.0)),
            mix_schedule=mix_schedule,
        )

    # --- trainer --------------------------------------------------------------
    # The coordinator is constructed *after* the (optional) imitation pretrain
    # so self-play workers start from the BC-pretrained weights.
    trainer = MuzeroLearner(
        cfg=cfg_dict,
        network=online_net,
        optimizer=optimizer,
        scheduler=scheduler,
        buffer=buffer,
        coordinator=None,
        device=device,
        out_dir=out_dir,
        wandb_logger=wandb_logger,
        human_eval=human_eval,
    )

    start_step = 0
    resume_path = cfg.resume.path if cfg.resume.path is not None else auto_resume_path
    if resume_path is not None:
        start_step = trainer.load(resume_path)
        print(f"[train] resumed at training_step={start_step} from {resume_path}")
        if 0 < start_step < pretrain_steps:
            print(
                f"[train] WARNING: resuming with imitation enabled at step {start_step} "
                f"< pretrain_steps={pretrain_steps}: pretrain will continue. If this "
                f"checkpoint came from a run WITHOUT imitation, you are about to "
                f"BC-pretrain an already-trained net — use a fresh run_name instead.",
                flush=True,
            )

    # SLURM sends SIGTERM ~32s before the wall. Re-raise as KeyboardInterrupt
    # so the try/finally below unwinds and wandb.finish() runs — otherwise the
    # run is mislabelled "crashed" on wandb instead of "finished (truncated)".
    # Installed before pretrain so the pretrain phase is also covered.
    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    coordinator = None
    try:
        # Backstop for run dirs from before the sentinel existed: a resumed
        # checkpoint at/over budget means there is nothing left to train —
        # don't pretrain or spawn 30+ worker processes just to exit.
        if trainer.env_step >= trainer.total_env_steps:
            print(
                f"[train] resumed checkpoint already at env-step budget "
                f"({trainer.env_step}/{trainer.total_env_steps}); exiting.",
                flush=True,
            )
            trainer.write_complete_sentinel()
            return

        # No-op unless imitation is enabled and start_step < pretrain_steps.
        start_step = trainer.pretrain(start_step)

        # --- self-play --------------------------------------------------------
        coordinator = SelfPlayCoordinator(
            cfg=cfg_dict,
            num_workers=int(cfg.selfplay.num_workers),
            levels=list(cfg.env.levels),
            device=device,
            initial_state_dict=online_net.state_dict(),
        )
        trainer.set_coordinator(coordinator)

        print("[train] broadcasting weights...", flush=True)
        # Send initial (resumed / pretrained) weights to workers before training.
        coordinator.broadcast_weights(online_net.state_dict())
        coordinator.set_train_step(max(0, start_step - pretrain_steps))
        print("[train] weights broadcast done, starting training loop", flush=True)

        trainer.training_loop(start_step=start_step)
    except KeyboardInterrupt as e:
        print(f"[train] interrupted ({e}); saving checkpoint + finalizing wandb", flush=True)
        # Persist the steps since the last save_every boundary so the next
        # chain leg resumes here instead of re-training them (which also made
        # wandb drop the re-logged, non-monotonic steps). SLURM's SIGTERM
        # leaves ~32s of grace — enough for one checkpoint write.
        try:
            trainer.save_final_checkpoint()
        except Exception as save_err:
            print(f"[train] final checkpoint save failed: {save_err}", flush=True)
    finally:
        trainer.close()
        if coordinator is not None:
            coordinator.stop()
        wandb_logger.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
